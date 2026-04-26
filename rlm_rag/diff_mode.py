"""PR / git-diff impact analysis.

Given a unified diff (or a git rev range), parse it to find changed files
and (best-effort) the changed symbols. For each impacted symbol, query the
indexed call graph for callers, then ask the LLM to assess whether the
change is safe.

This is the natural RLM use case: the diff is small and fits in the root
context; the codebase is large and lives in the index; the question is
"who else does this touch."
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from . import _rlm_helper as rlm_helper
from .store import ChunkStore


@dataclass
class DiffHunk:
    file_path: str          # path as it appears in the diff (b-side)
    added_lines: list[int]  # 1-based line numbers in the new file
    body: str               # the raw hunk text


@dataclass
class ChangedSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class DiffImpact:
    hunks: list[DiffHunk]
    changed_symbols: list[ChangedSymbol]
    callers_per_symbol: dict[str, list[dict]]   # symbol_name → [{file_path, caller}]
    summary: str = ""


# ---------- diff parsing ------------------------------------------------

_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.MULTILINE)


def parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    """Parse a unified diff into per-hunk records.

    Handles the common `git diff` output format. Captures only added lines
    (we want to know what's new); deleted lines are noted in the body for
    context but don't go into added_lines.
    """
    hunks: list[DiffHunk] = []
    # Split on file headers.
    sections = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    for section in sections:
        m = _FILE_RE.search(section)
        if not m:
            continue
        file_path = m.group(1)
        # Collect each hunk in this section.
        for h_match in _HUNK_RE.finditer(section):
            start_new = int(h_match.group(1))
            # Hunk body: from end of @@ line to next @@ or end of section.
            body_start = h_match.end()
            next_h = _HUNK_RE.search(section, body_start)
            body_end = next_h.start() if next_h else len(section)
            body = section[body_start:body_end]
            added: list[int] = []
            cur = start_new
            for line in body.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added.append(cur)
                    cur += 1
                elif line.startswith("-") and not line.startswith("---"):
                    pass  # removal: don't advance new-file counter
                else:
                    cur += 1
            hunks.append(DiffHunk(file_path=file_path,
                                  added_lines=added,
                                  body=body))
    return hunks


def diff_from_git(repo: Path, rev_spec: str) -> str:
    """Run `git diff <rev_spec>` in `repo` and return the unified diff."""
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--unified=3", rev_spec],
        check=True, capture_output=True, text=True,
    )
    return out.stdout


# ---------- symbol matching --------------------------------------------

def changed_symbols_from_hunks(
    hunks: list[DiffHunk], store: ChunkStore,
) -> list[ChangedSymbol]:
    """For each hunk, find the indexed symbols whose line ranges overlap
    the hunk's added lines.
    """
    out: list[ChangedSymbol] = []
    seen: set[tuple[str, str, int]] = set()

    for h in hunks:
        if not h.added_lines:
            continue
        # Pull all symbols indexed for this file, then filter by line overlap.
        rows = store.db.execute(
            """SELECT name, kind, start_line, end_line
               FROM symbols WHERE file_path = ?""",
            (h.file_path,),
        ).fetchall()
        for name, kind, start, end in rows:
            for ln in h.added_lines:
                if start <= ln <= end:
                    key = (h.file_path, name, start)
                    if key in seen:
                        break
                    seen.add(key)
                    out.append(ChangedSymbol(
                        file_path=h.file_path, name=name, kind=kind,
                        start_line=start, end_line=end,
                    ))
                    break
    return out


# ---------- LLM impact analysis ----------------------------------------

IMPACT_SYSTEM = (
    "You are a careful code reviewer. Given a diff hunk, the affected "
    "symbol's identity, and a list of callers from elsewhere in the "
    "codebase, decide whether the change is likely to break those callers. "
    "Be terse, specific, and grounded in the evidence shown."
)


def _impact_prompt(
    hunk: DiffHunk, sym: ChangedSymbol, callers: list[dict],
) -> str:
    callers_text = "\n".join(
        f"  - {c['file_path']}::{c['caller']}" for c in callers[:30]
    ) or "  (no callers found in index)"
    return textwrap.dedent(f"""\
        Changed symbol: `{sym.kind} {sym.name}` in `{sym.file_path}` (lines {sym.start_line}-{sym.end_line})

        Diff hunk:
        ```
        {hunk.body.strip()}
        ```

        Known callers (from indexed call graph; name-only matching, may include
        unrelated functions with the same leaf name):
        {callers_text}

        Assess the impact of this change. Specifically:
          1. Does the change appear to break the API contract (signature,
             behavior, return values, error semantics)?
          2. If yes, which listed callers are most likely to break, and how?
          3. If you can't tell from the evidence alone, say so plainly.""")


def analyze_diff(
    diff_text: str,
    store: ChunkStore,
    *,
    sub_model: str | None = None,
    concurrency: int = 5,
) -> DiffImpact:
    """Run the full impact pipeline. Returns a DiffImpact with everything."""
    hunks = parse_unified_diff(diff_text)
    syms = changed_symbols_from_hunks(hunks, store)

    callers_per_symbol: dict[str, list[dict]] = {}
    for s in syms:
        callers_per_symbol[s.name] = store.callers_of(s.name)

    if not syms:
        return DiffImpact(
            hunks=hunks, changed_symbols=[], callers_per_symbol={},
            summary="(diff did not overlap any indexed symbols)",
        )

    # Per-symbol impact analysis in parallel.
    prompts = []
    paired = []  # (hunk, sym) per prompt
    for sym in syms:
        # Pick the hunk in the same file that overlaps this symbol.
        for h in hunks:
            if h.file_path == sym.file_path and any(
                sym.start_line <= ln <= sym.end_line for ln in h.added_lines
            ):
                prompts.append(_impact_prompt(h, sym, callers_per_symbol[sym.name]))
                paired.append((h, sym))
                break

    impact_summaries = rlm_helper.llm_query_batch(
        prompts,
        system=IMPACT_SYSTEM,
        model=sub_model,
        concurrency=concurrency,
    )

    blocks = []
    for (_, sym), summary in zip(paired, impact_summaries):
        n_callers = len(callers_per_symbol[sym.name])
        blocks.append(
            f"### `{sym.file_path}` :: `{sym.kind} {sym.name}` "
            f"(callers: {n_callers})\n{summary.strip()}"
        )
    summary_text = "\n\n".join(blocks)

    return DiffImpact(
        hunks=hunks,
        changed_symbols=syms,
        callers_per_symbol=callers_per_symbol,
        summary=summary_text,
    )
