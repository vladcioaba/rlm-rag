"""Tests for the VCS abstraction. Subprocess calls are mocked so the suite
stays offline and doesn't require git or p4 to be installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rlm_rag.vcs import (
    BlameLine,
    GitVCS,
    P4VCS,
    detect_vcs,
    parse_git_porcelain,
    parse_p4_annotate,
    parse_p4_describe_tagged,
)


# ---------- git porcelain ----------------------------------------------

GIT_PORCELAIN = (
    "abcdef0123456789abcdef0123456789abcdef01 1 7\n"
    "author Jane Doe\n"
    "author-mail <jane@example.com>\n"
    "author-time 1700000000\n"
    "author-tz +0000\n"
    "summary refactor authenticate\n"
    "previous 0000000000000000000000000000000000000000 src/auth.py\n"
    "filename src/auth.py\n"
    "\tdef authenticate(user, password):\n"
    "1234567890abcdef1234567890abcdef12345678 2 8\n"
    "author John Smith\n"
    "author-mail <john@example.com>\n"
    "author-time 1710000000\n"
    "author-tz +0000\n"
    "summary tighten input check\n"
    "filename src/auth.py\n"
    "\t    if not user or not password:\n"
)


def test_parse_git_porcelain_extracts_per_line_metadata():
    lines = parse_git_porcelain(GIT_PORCELAIN)
    assert len(lines) == 2
    assert lines[0].line_no == 7
    assert lines[0].rev == "abcdef0123456789abcdef0123456789abcdef01"
    assert lines[0].author == "Jane Doe"
    assert lines[0].timestamp == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert "refactor" in lines[0].summary
    assert lines[1].line_no == 8
    assert lines[1].author == "John Smith"


def test_parse_git_porcelain_handles_repeated_sha_block_collapsed():
    """When the same commit owns several consecutive lines, git porcelain
    only writes the metadata block on the first line; later lines just have
    `<sha> <orig_line> <result_line>` followed by a `\\t...` content row.
    The parser carries the metadata forward."""
    text = (
        "deadbeef00000000000000000000000000000000 1 5\n"
        "author Alice\n"
        "author-time 1600000000\n"
        "summary first\n"
        "\tline five\n"
        "deadbeef00000000000000000000000000000000 2 6\n"
        "\tline six\n"
    )
    lines = parse_git_porcelain(text)
    assert len(lines) == 2
    assert all(b.rev.startswith("deadbeef") for b in lines)
    # The second line should have inherited metadata via the meta dict.
    assert lines[1].author == "Alice"
    assert lines[1].timestamp is not None


# ---------- p4 annotate -------------------------------------------------

P4_ANNOTATE = (
    "//depot/src/auth.cpp#5 - edit change 12345 (text)\n"
    "12340: int authenticate(const std::string& user) {\n"
    "12340:     if (user.empty()) return 0;\n"
    "12345:     if (!validate(user)) return 0;\n"
    "12340:     return 1;\n"
    "12340: }\n"
)


def test_parse_p4_annotate_slices_to_range():
    lines = parse_p4_annotate(P4_ANNOTATE, start_line=2, end_line=4)
    assert [b.line_no for b in lines] == [2, 3, 4]
    assert [b.rev for b in lines] == ["12340", "12345", "12340"]


def test_parse_p4_annotate_skips_depot_header():
    lines = parse_p4_annotate(P4_ANNOTATE, start_line=1, end_line=10)
    # Five real content lines; header should not have produced a BlameLine.
    assert len(lines) == 5
    assert lines[0].line_no == 1


# ---------- p4 describe -------------------------------------------------

P4_DESCRIBE_TAGGED = (
    "... change 12340\n"
    "... user alice\n"
    "... time 1700000000\n"
    "... desc Fix auth path\\nMore detail line two\n"
    "\n"
    "... change 12345\n"
    "... user bob\n"
    "... time 1710000000\n"
    "... desc Add validate guard\n"
)


def test_parse_p4_describe_tagged_returns_per_cl_metadata():
    meta = parse_p4_describe_tagged(P4_DESCRIBE_TAGGED)
    assert set(meta.keys()) == {"12340", "12345"}
    assert meta["12340"]["author"] == "alice"
    assert meta["12340"]["timestamp"] == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert meta["12345"]["author"] == "bob"
    # First line of the description only.
    assert meta["12345"]["summary"] == "Add validate guard"


# ---------- p4 diff sourcing -------------------------------------------

class _FakeRun:
    """Records the cmd it was called with and returns canned stdout."""
    def __init__(self, stdout: str = ""):
        self.stdout = stdout
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        # subprocess.run-like result
        class R:
            returncode = 0
        r = R()
        r.stdout = self.stdout
        r.stderr = ""
        return r


def test_p4vcs_diff_routes_submitted_cl_through_describe(tmp_path):
    fake = _FakeRun(stdout="")
    vcs = P4VCS(tmp_path)
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake):
        vcs.diff("12345")
    assert fake.calls[0][:4] == ["p4", "describe", "-du", "12345"]


def test_p4vcs_diff_routes_shelved_cl_with_dash_S(tmp_path):
    fake = _FakeRun(stdout="")
    vcs = P4VCS(tmp_path)
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake):
        vcs.diff("@=12345")
    assert fake.calls[0][:5] == ["p4", "describe", "-du", "-S", "12345"]


def test_p4vcs_diff_routes_range_through_diff2(tmp_path):
    fake = _FakeRun(stdout="")
    vcs = P4VCS(tmp_path)
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake):
        vcs.diff("12300,12345")
    assert fake.calls[0][:2] == ["p4", "diff2"]
    assert "@12300" in fake.calls[0]
    assert "@12345" in fake.calls[0]


def test_p4vcs_rewrites_depot_paths_to_repo_relative(tmp_path):
    """diff() should turn `==== //depot/foo/bar.cpp#3 ===` headers into
    `==== src/bar.cpp#3 ===` using `p4 where` output.
    """
    repo = tmp_path
    diff_text = (
        "==== //depot/foo/bar.cpp#3 (text) ====\n"
        "@@ -1,2 +1,3 @@\n"
        " line\n"
        "+added\n"
    )
    where_out = (
        "... depotFile //depot/foo/bar.cpp\n"
        f"... clientFile //client/foo/bar.cpp\n"
        f"... path {repo}/src/bar.cpp\n"
    )

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stderr = ""
        r = R()
        if cmd[:2] == ["p4", "describe"] or cmd[:2] == ["p4", "diff2"]:
            r.stdout = diff_text
        elif "where" in cmd:
            r.stdout = where_out
        else:
            r.stdout = ""
        return r

    vcs = P4VCS(repo)
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake_run):
        out = vcs.diff("12345")
    assert "//depot/foo/bar.cpp" not in out
    assert "src/bar.cpp" in out


# ---------- detect ------------------------------------------------------

def test_detect_vcs_finds_git_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    vcs = detect_vcs(tmp_path)
    assert isinstance(vcs, GitVCS)
    assert vcs.root == tmp_path.resolve()


def test_detect_vcs_returns_none_when_neither(tmp_path):
    """No .git, and `p4 info -s` either fails or is missing."""
    def fake_run(cmd, **kw):
        raise FileNotFoundError("p4 not installed")
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake_run):
        vcs = detect_vcs(tmp_path)
    assert vcs is None


def test_detect_vcs_finds_p4_when_p4_info_succeeds(tmp_path):
    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = "User name: alice\nClient root: /tmp/wks\n"
            stderr = ""
        return R()
    with patch("rlm_rag.vcs.subprocess.run", side_effect=fake_run):
        vcs = detect_vcs(tmp_path)
    assert isinstance(vcs, P4VCS)
