"""Iterative RLM root loop.

The simple `query()` in query.py does retrieve → triage → synthesize in one
shot. This loop lets the orchestrating model *react* to what it sees:
re-search with refined queries, navigate the call graph, fetch specific
definitions, and decide for itself when to finalize.

The loop is bounded (default 5 iterations) and budgeted (default 30 sub-LLM
calls) so it can't run away.

Protocol — the root LLM emits a JSON object on every turn:

    {"action": "search",   "query": "...", "top_k": 10}
    {"action": "callers",  "symbol": "authenticate"}
    {"action": "imports",  "file": "src/auth.py"}
    {"action": "files_importing", "module": "auth"}
    {"action": "fetch",    "file": "src/auth.py", "symbol": "authenticate"}
    {"action": "find",     "name": "User"}
    {"action": "grep",     "pattern": "validate"}
    {"action": "final",    "answer": "..."}

Every response from the loop describes what was done and surfaces compact,
bounded results — never raw chunk text dumped wholesale.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from . import _rlm_helper as rlm_helper
from .embedder import Embedder
from .models import ModelConfig, ModelTier, default_config
from .retrieval import Reranker, retrieve
from .store import ChunkStore, SearchHit


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code-research orchestrator. You are answering a user's question
    about a codebase you cannot see directly. Each turn you may take ONE action,
    formatted as a single JSON object inside a ```json fenced block.

    Available actions:
      {"action":"search",  "query":"...", "top_k":10}      retrieve top_k chunks
      {"action":"find",    "name":"X"}                     exact-match symbol lookup
      {"action":"grep",    "pattern":"X"}                  partial/case-insensitive symbol lookup
      {"action":"callers", "symbol":"X"}                   list (file, caller) tuples that call X
      {"action":"imports", "file":"path/to/file"}          modules that file imports
      {"action":"files_importing", "module":"X"}           files that import X
      {"action":"fetch",   "file":"path", "symbol":"name"} pull the source of a specific chunk
      {"action":"final",   "answer":"..."}                 commit to a final answer

    Strategy:
      - Start with `search` to get oriented.
      - Use `find`/`grep` to locate specific symbols by name when retrieval is fuzzy.
      - Use `callers` and `imports` to follow the call graph when impact matters.
      - Use `fetch` to read a specific chunk when its summary in a search hit
        wasn't enough.
      - Stop with `final` once you have enough evidence. Be specific in your
        answer — cite file paths and symbol names.

    Be terse. Don't restate the user's question. Don't narrate your plan in
    prose. Just emit the next action.
""")


@dataclass
class IterationLog:
    iteration: int
    action: str
    payload: dict
    summary: str           # short text the model sees on next turn
    raw: Any = None        # full data, persisted but not re-fed to the model


@dataclass
class IterativeResult:
    answer: str
    iterations: list[IterationLog] = field(default_factory=list)
    stopped_reason: str = "final"   # final | budget | iterations | parse_error
    tokens_in: int = 0
    tokens_out: int = 0


# ---------- action handlers ---------------------------------------------

def _do_search(store: ChunkStore, embedder: Embedder, reranker: Reranker | None,
               args: dict, mode: str = "hybrid") -> tuple[str, list[SearchHit]]:
    q = args.get("query", "").strip()
    k = int(args.get("top_k", 10))
    if not q:
        return "search: missing 'query'", []
    hits = retrieve(q, store, embedder, top_k=k, mode=mode,
                    rerank=reranker is not None, reranker=reranker)
    summary = "\n".join(
        f"  {h.score:+.3f}  {h.chunk.file_path}:{h.chunk.start_line}  "
        f"{h.chunk.symbol_kind} {h.chunk.symbol_name}"
        for h in hits
    ) or "  (no hits)"
    return f"search '{q}' top_k={k}:\n{summary}", hits


def _do_find(store: ChunkStore, args: dict) -> tuple[str, Any]:
    name = args.get("name", "").strip()
    if not name:
        return "find: missing 'name'", None
    rows = store.find_symbol(name)
    if not rows:
        return f"find '{name}': no matches", rows
    summary = "\n".join(
        f"  {r['file_path']}:{r['start_line']}  {r['kind']} {r['name']}"
        for r in rows[:30]
    )
    return f"find '{name}' ({len(rows)} matches):\n{summary}", rows


def _do_grep(store: ChunkStore, args: dict) -> tuple[str, Any]:
    pattern = args.get("pattern", "").strip()
    if not pattern:
        return "grep: missing 'pattern'", None
    rows = store.grep_symbol(pattern)
    if not rows:
        return f"grep '{pattern}': no matches", rows
    summary = "\n".join(
        f"  {r['file_path']}:{r['start_line']}  {r['kind']} {r['name']}"
        for r in rows[:30]
    )
    return f"grep '{pattern}' ({len(rows)} matches):\n{summary}", rows


def _do_callers(store: ChunkStore, args: dict) -> tuple[str, Any]:
    sym = args.get("symbol", "").strip()
    if not sym:
        return "callers: missing 'symbol'", None
    rows = store.callers_of(sym)
    if not rows:
        return f"callers of '{sym}': none found", rows
    summary = "\n".join(f"  {r['file_path']}  {r['caller']}" for r in rows[:50])
    return f"callers of '{sym}' ({len(rows)}):\n{summary}", rows


def _do_imports(store: ChunkStore, args: dict) -> tuple[str, Any]:
    f = args.get("file", "").strip()
    if not f:
        return "imports: missing 'file'", None
    mods = store.imports_of(f)
    return f"imports of {f}: {mods or '(none)'}", mods


def _do_files_importing(store: ChunkStore, args: dict) -> tuple[str, Any]:
    m = args.get("module", "").strip()
    if not m:
        return "files_importing: missing 'module'", None
    files = store.files_importing(m)
    return f"files importing '{m}' ({len(files)}):\n  " + "\n  ".join(files[:50]), files


def _do_fetch(store: ChunkStore, args: dict) -> tuple[str, Any]:
    f = args.get("file", "").strip()
    s = args.get("symbol", "").strip()
    if not (f and s):
        return "fetch: needs 'file' and 'symbol'", None
    chunk = store.get_chunk(f, s)
    if chunk is None:
        return f"fetch {f}::{s}: not found", None
    body_preview = chunk.text[:1500]
    elided = "" if len(chunk.text) <= 1500 else f"\n... ({len(chunk.text) - 1500} more chars)"
    return (
        f"fetch {f}:{chunk.start_line}-{chunk.end_line} {chunk.symbol_kind} {chunk.symbol_name}:\n"
        f"```\n{body_preview}{elided}\n```",
        chunk,
    )


# ---------- parsing -----------------------------------------------------

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_action(text: str) -> dict | None:
    m = _FENCED_JSON.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: try the whole response as JSON.
    try:
        return json.loads(text.strip())
    except Exception:
        return None


# ---------- main loop ---------------------------------------------------

def iterative_query(
    question: str,
    store: ChunkStore,
    embedder: Embedder,
    *,
    root_model: str | None = None,
    sub_model: str | None = None,
    max_iterations: int = 5,
    retrieval_mode: str = "hybrid",
    rerank: bool = False,
    rerank_model: str | None = None,
    initial_top_k: int = 10,
    model_config: ModelConfig | None = None,
) -> IterativeResult:
    cfg = model_config or default_config()
    root_model = root_model or cfg.model_for(ModelTier.BALANCED)
    root_thinking = cfg.thinking_for(ModelTier.BALANCED)

    reranker = Reranker(model_name=rerank_model) if rerank else None

    # Seed the conversation with an initial search so the model has signal
    # without needing to ask first.
    seed_summary, _ = _do_search(
        store, embedder, reranker,
        {"query": question, "top_k": initial_top_k},
        mode=retrieval_mode,
    )

    history = [
        f"Question: {question}",
        f"Initial retrieval (top_k={initial_top_k}, mode={retrieval_mode}):\n{seed_summary}",
    ]
    iterations: list[IterationLog] = []

    for i in range(1, max_iterations + 1):
        prompt = "\n\n".join(history) + "\n\nNext action:"
        try:
            response = rlm_helper.llm_query(
                prompt,
                system=SYSTEM_PROMPT,
                model=root_model,
                max_tokens=2048,
                thinking_budget=root_thinking,
            )
        except rlm_helper.BudgetExceeded:
            return _wrap_result(
                "(budget exhausted before finalization)",
                iterations, "budget",
            )

        action = _parse_action(response)
        if action is None:
            iterations.append(IterationLog(
                iteration=i, action="parse_error",
                payload={"raw": response[:500]},
                summary="parse_error",
            ))
            return _wrap_result(
                f"(failed to parse action from root model on iteration {i})",
                iterations, "parse_error",
            )

        kind = action.get("action", "")
        try:
            if kind == "final":
                ans = action.get("answer", "")
                iterations.append(IterationLog(i, "final", action, "final"))
                return _wrap_result(ans, iterations, "final")

            elif kind == "search":
                summary, raw = _do_search(store, embedder, reranker, action, mode=retrieval_mode)
            elif kind == "find":
                summary, raw = _do_find(store, action)
            elif kind == "grep":
                summary, raw = _do_grep(store, action)
            elif kind == "callers":
                summary, raw = _do_callers(store, action)
            elif kind == "imports":
                summary, raw = _do_imports(store, action)
            elif kind == "files_importing":
                summary, raw = _do_files_importing(store, action)
            elif kind == "fetch":
                summary, raw = _do_fetch(store, action)
            else:
                summary, raw = f"unknown action: {kind!r}", None
        except Exception as e:
            summary, raw = f"action {kind!r} raised: {e}", None

        iterations.append(IterationLog(i, kind, action, summary, raw))
        history.append(f"Iteration {i}: {summary}")

    return _wrap_result(
        "(reached max_iterations without a final answer)",
        iterations, "iterations",
    )


def _wrap_result(answer: str, logs: list[IterationLog], reason: str) -> IterativeResult:
    return IterativeResult(answer=answer, iterations=logs, stopped_reason=reason)
