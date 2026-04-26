"""Tests for the symbol/import/call graph stored alongside chunks."""

from __future__ import annotations

from rlm_rag.indexer import index_directory
from rlm_rag.store import ChunkStore


def test_symbols_and_methods_indexed(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)

    rows = store.find_symbol("authenticate")
    assert any(r["name"] == "authenticate" for r in rows)

    # Methods are indexed as 'Class.method'.
    rows = store.find_symbol("User.display")
    assert any(r["kind"] == "method" for r in rows)


def test_grep_partial_case_insensitive(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)
    rows = store.grep_symbol("auth")
    names = {r["name"] for r in rows}
    assert "authenticate" in names


def test_callers_of_finds_callee(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)
    # `authenticate` calls `hash_password`.
    rows = store.callers_of("hash_password")
    callers = {r["caller"] for r in rows}
    assert "authenticate" in callers


def test_imports_of_file(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)
    mods = store.imports_of("src/auth.py")
    assert "hashlib" in mods


def test_replace_file_clears_old_graph_rows(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)

    # Rewrite auth.py with a different symbol; the old "authenticate" row must go.
    (tmp_repo / "src" / "auth.py").write_text(
        "def brand_new(): return 1\n"
    )
    index_directory(tmp_repo, store, fake_embedder)
    rows = store.find_symbol("authenticate")
    assert rows == []
    assert store.find_symbol("brand_new")
