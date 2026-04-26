"""Tests for the diff parser and the diff→symbol overlap logic.

The LLM impact-analysis call is not exercised here (no API key in CI).
"""

from __future__ import annotations

from rlm_rag.diff_mode import (
    changed_symbols_from_hunks,
    parse_unified_diff,
)
from rlm_rag.indexer import index_directory
from rlm_rag.store import ChunkStore


SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index abc..def 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -7,3 +7,5 @@ def hash_password(pw):
     return hashlib.sha256(pw.encode()).hexdigest()

 def authenticate(user, password):
-    return USERS.get(user) == hash_password(password)
+    if not user or not password:
+        return False
+    return USERS.get(user) == hash_password(password)
"""


def test_parse_unified_diff_basic():
    hunks = parse_unified_diff(SAMPLE_DIFF)
    assert len(hunks) == 1
    h = hunks[0]
    assert h.file_path == "src/auth.py"
    # Three added lines (two new + one replacement of the removed line).
    # All land at or after the @@ header's new-file start (line 7).
    assert len(h.added_lines) == 3
    assert all(ln >= 7 for ln in h.added_lines)


def test_parse_diff_with_multiple_files():
    multi = SAMPLE_DIFF + """
diff --git a/src/models.py b/src/models.py
index 111..222 100644
--- a/src/models.py
+++ b/src/models.py
@@ -1,3 +1,4 @@
+import os
 class User:
     def __init__(self, name):
         self.name = name
"""
    hunks = parse_unified_diff(multi)
    files = {h.file_path for h in hunks}
    assert files == {"src/auth.py", "src/models.py"}


def test_changed_symbols_overlap_with_indexed_symbols(tmp_repo, fake_embedder, tmp_path):
    """Hunk that touches `authenticate` should surface that symbol."""
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)

    hunks = parse_unified_diff(SAMPLE_DIFF)
    syms = changed_symbols_from_hunks(hunks, store)
    names = {s.name for s in syms}
    assert "authenticate" in names


def test_changed_symbols_returns_nothing_for_unrelated_diff(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)
    unrelated = """\
diff --git a/src/never_indexed.py b/src/never_indexed.py
index 000..111 100644
--- a/src/never_indexed.py
+++ b/src/never_indexed.py
@@ -1,1 +1,2 @@
 x = 1
+y = 2
"""
    hunks = parse_unified_diff(unrelated)
    syms = changed_symbols_from_hunks(hunks, store)
    assert syms == []
