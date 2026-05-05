"""VCS abstraction over Git and Perforce.

The tool needs two things from a VCS:
  - diff(rev_spec) -> unified diff text (repo-relative paths) for `rlm-rag pr`.
  - blame(file, start, end) -> per-line author/changelist/timestamp for the
    iterative loop's `blame_of` action.

Auto-detection: a `.git` ancestor → Git. Otherwise we probe `p4 info -s` in
the working dir; if that succeeds with a "Client root:" line, the tree is in
a P4 client. Detection returns None if neither matches — callers decide
whether that's fatal.

P4 paths in diffs come back as depot paths (`//depot/foo/bar.cpp`); P4VCS
translates them to repo-relative paths using `p4 where` so the rest of the
pipeline can join them against indexed `file_path` rows verbatim.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BlameLine:
    line_no: int
    rev: str                       # commit sha (git) or CL number (p4)
    author: str = ""
    timestamp: datetime | None = None
    summary: str = ""              # one-line description if the VCS gave us one


class VCS:
    name: str = ""

    def __init__(self, root: Path) -> None:
        self.root = root

    def diff(self, rev_spec: str) -> str:
        raise NotImplementedError

    def blame(self, file_path: str, start_line: int, end_line: int) -> list[BlameLine]:
        raise NotImplementedError


# ---------- git ---------------------------------------------------------

class GitVCS(VCS):
    name = "git"

    def diff(self, rev_spec: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--unified=3", rev_spec],
            check=True, capture_output=True, text=True,
        )
        return out.stdout

    def blame(self, file_path: str, start_line: int, end_line: int) -> list[BlameLine]:
        out = subprocess.run(
            ["git", "-C", str(self.root), "blame", "--porcelain",
             "-L", f"{start_line},{end_line}", "--", file_path],
            check=True, capture_output=True, text=True,
        )
        return parse_git_porcelain(out.stdout)


def parse_git_porcelain(text: str) -> list[BlameLine]:
    """Parse `git blame --porcelain` output. Each line is preceded by a
    header block keyed on commit sha; we carry those headers forward as
    git only repeats them on the first line per commit.
    """
    lines: list[BlameLine] = []
    meta: dict[str, dict] = {}        # sha → {author, time, summary}
    cur_sha: str | None = None
    cur_line: int | None = None
    in_block = False

    for raw in text.splitlines():
        if not in_block:
            m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)(?: \d+)?$", raw)
            if m:
                cur_sha = m.group(1)
                cur_line = int(m.group(2))
                meta.setdefault(cur_sha, {})
                in_block = True
            continue
        if raw.startswith("\t"):
            # The actual file line; close the block.
            if cur_sha is not None and cur_line is not None:
                info = meta.get(cur_sha, {})
                lines.append(BlameLine(
                    line_no=cur_line,
                    rev=cur_sha,
                    author=info.get("author", ""),
                    timestamp=info.get("timestamp"),
                    summary=info.get("summary", ""),
                ))
            in_block = False
            cur_sha = None
            cur_line = None
            continue
        if cur_sha is None:
            continue
        if raw.startswith("author "):
            meta[cur_sha]["author"] = raw[len("author "):]
        elif raw.startswith("author-mail "):
            # prefer mail when populated and we don't have a name yet
            meta[cur_sha].setdefault("author", raw[len("author-mail "):].strip("<>"))
        elif raw.startswith("author-time "):
            try:
                meta[cur_sha]["timestamp"] = datetime.fromtimestamp(
                    int(raw[len("author-time "):]), tz=timezone.utc,
                )
            except ValueError:
                pass
        elif raw.startswith("summary "):
            meta[cur_sha]["summary"] = raw[len("summary "):]
    return lines


# ---------- perforce ----------------------------------------------------

_P4_DIFF_HEADER = re.compile(r"^==== (//[^#\s]+)#\d+", re.MULTILINE)


class P4VCS(VCS):
    name = "p4"

    def diff(self, rev_spec: str) -> str:
        """Source a diff for a P4 revision spec.

        Spec syntax:
          "12345"        → submitted CL              (`p4 describe -du`)
          "@=12345"      → shelved CL                (`p4 describe -du -S`)
          "12340,12345"  → range between two CLs     (`p4 diff2 @a @b`)
        """
        rev_spec = rev_spec.strip()
        if "," in rev_spec:
            a, b = (s.strip() for s in rev_spec.split(",", 1))
            cmd = ["p4", "diff2", "-du", f"@{a.lstrip('@')}", f"@{b.lstrip('@')}"]
        elif rev_spec.startswith("@="):
            cmd = ["p4", "describe", "-du", "-S", rev_spec[2:]]
        else:
            cmd = ["p4", "describe", "-du", rev_spec.lstrip("@")]
        out = subprocess.run(cmd, cwd=str(self.root), check=True,
                             capture_output=True, text=True)
        return self._depot_to_repo_relative(out.stdout)

    def blame(self, file_path: str, start_line: int, end_line: int) -> list[BlameLine]:
        out = subprocess.run(
            ["p4", "annotate", "-c", file_path],
            cwd=str(self.root), check=True, capture_output=True, text=True,
        )
        lines = parse_p4_annotate(out.stdout, start_line, end_line)
        # Backfill author/timestamp/summary by batch-describing unique CLs.
        cls = sorted({b.rev for b in lines if b.rev})
        if cls:
            meta = self._describe_changes(cls)
            for b in lines:
                info = meta.get(b.rev)
                if info:
                    b.author = info.get("author", b.author)
                    b.timestamp = info.get("timestamp", b.timestamp)
                    b.summary = info.get("summary", b.summary)
        return lines

    # ---- internals ------------------------------------------------------

    def _depot_to_repo_relative(self, diff_text: str) -> str:
        depots = sorted({m.group(1) for m in _P4_DIFF_HEADER.finditer(diff_text)})
        if not depots:
            return diff_text
        mapping = self._where(depots)
        if not mapping:
            return diff_text

        def rewrite(match: re.Match) -> str:
            depot = match.group(1)
            local = mapping.get(depot)
            if not local:
                return match.group(0)
            try:
                rel = Path(local).resolve().relative_to(self.root.resolve())
            except ValueError:
                return match.group(0)
            return match.group(0).replace(depot, str(rel), 1)

        return _P4_DIFF_HEADER.sub(rewrite, diff_text)

    def _where(self, depot_paths: list[str]) -> dict[str, str]:
        """Run `p4 -ztag where` for a batch of depot paths. Returns
        {depot_path: local_path}. Paths not in the client mapping are
        omitted silently.
        """
        if not depot_paths:
            return {}
        try:
            out = subprocess.run(
                ["p4", "-ztag", "where", *depot_paths],
                cwd=str(self.root), check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            return {}
        mapping: dict[str, str] = {}
        cur_depot: str | None = None
        for raw in out.stdout.splitlines():
            if raw.startswith("... depotFile "):
                cur_depot = raw[len("... depotFile "):].strip()
            elif raw.startswith("... path ") and cur_depot:
                mapping[cur_depot] = raw[len("... path "):].strip()
                cur_depot = None
        return mapping

    def _describe_changes(self, cls: list[str]) -> dict[str, dict]:
        try:
            out = subprocess.run(
                ["p4", "-ztag", "describe", "-s", *cls],
                cwd=str(self.root), check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            return {}
        return parse_p4_describe_tagged(out.stdout)


def parse_p4_annotate(text: str, start_line: int, end_line: int) -> list[BlameLine]:
    """Parse `p4 annotate -c <file>`. The first line is a depot-path header;
    subsequent lines look like `<changelist>: <line content>`. We slice to
    [start_line, end_line] (1-based, inclusive).
    """
    out: list[BlameLine] = []
    cur = 0
    for raw in text.splitlines():
        if raw.startswith("//") and "#" in raw:
            continue  # depot path header
        m = re.match(r"^(\d+):\s?(.*)$", raw)
        if not m:
            continue
        cur += 1
        if cur < start_line:
            continue
        if cur > end_line:
            break
        out.append(BlameLine(line_no=cur, rev=m.group(1)))
    return out


def parse_p4_describe_tagged(text: str) -> dict[str, dict]:
    """Parse `p4 -ztag describe -s <cls...>` for change metadata. Returns
    {cl_number: {author, timestamp, summary}}.
    """
    out: dict[str, dict] = {}
    cur: dict | None = None
    cur_cl: str | None = None
    for raw in text.splitlines():
        if raw.startswith("... change "):
            if cur_cl is not None and cur is not None:
                out[cur_cl] = cur
            cur_cl = raw[len("... change "):].strip()
            cur = {}
        elif cur is None:
            continue
        elif raw.startswith("... user "):
            cur["author"] = raw[len("... user "):].strip()
        elif raw.startswith("... time "):
            try:
                cur["timestamp"] = datetime.fromtimestamp(
                    int(raw[len("... time "):]), tz=timezone.utc,
                )
            except ValueError:
                pass
        elif raw.startswith("... desc "):
            desc = raw[len("... desc "):].strip()
            cur["summary"] = desc.splitlines()[0] if desc else ""
    if cur_cl is not None and cur is not None:
        out[cur_cl] = cur
    return out


# ---------- detection ---------------------------------------------------

def detect_vcs(root: Path) -> VCS | None:
    """Walk up from root looking for `.git`. Failing that, probe `p4 info -s`
    in the directory. Returns None if neither detection succeeds.
    """
    cur = root.resolve()
    for d in [cur, *cur.parents]:
        if (d / ".git").exists():
            return GitVCS(d)
    try:
        r = subprocess.run(
            ["p4", "info", "-s"], cwd=str(root),
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "Client root:" in r.stdout:
            return P4VCS(root)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
