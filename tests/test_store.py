"""Unit tests for the SQLite vector store and indexer.

These do not need sentence-transformers — the tmp_repo + fake_embedder
fixtures keep everything in-process and deterministic.
"""

from __future__ import annotations

import numpy as np

from rlm_rag.chunker import Chunk
from rlm_rag.indexer import index_directory
from rlm_rag.store import ChunkStore


def _chunk(name="foo", text="def foo(): pass", path="x.py"):
    return Chunk(
        file_path=path, symbol_name=name, symbol_kind="function",
        start_line=1, end_line=1, text=text,
    )


def test_store_round_trip(tmp_path, fake_embedder):
    store = ChunkStore(tmp_path / "x.db")
    c = _chunk()
    e = fake_embedder.embed(c.text)
    store.replace_file("x.py", "deadbeef", [(c, e)])

    embs, chunks = store.all_embeddings()
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "foo"
    assert embs.shape == (1, fake_embedder.dim)
    # Stored vector should round-trip exactly through float32 bytes.
    np.testing.assert_array_equal(embs[0], e)


def test_search_returns_descending_scores(tmp_path, fake_embedder):
    store = ChunkStore(tmp_path / "y.db")
    chunks = [
        _chunk(name="alpha", text="auth login password user"),
        _chunk(name="beta",  text="render html template"),
        _chunk(name="gamma", text="auth user password session"),
    ]
    embs = fake_embedder.embed_batch([c.text for c in chunks])
    store.replace_file("x.py", "h1", list(zip(chunks, embs)))

    q = fake_embedder.embed("auth user password")
    hits = store.search(q, k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_replace_file_overwrites_prior_chunks(tmp_path, fake_embedder):
    store = ChunkStore(tmp_path / "z.db")
    c1 = _chunk(name="old", text="old text")
    store.replace_file("x.py", "h1", [(c1, fake_embedder.embed(c1.text))])
    assert store.count() == 1

    c2 = _chunk(name="new", text="new text")
    store.replace_file("x.py", "h2", [(c2, fake_embedder.embed(c2.text))])
    assert store.count() == 1  # not 2
    _, chunks = store.all_embeddings()
    assert chunks[0].symbol_name == "new"


def test_remove_file(tmp_path, fake_embedder):
    store = ChunkStore(tmp_path / "z2.db")
    store.replace_file("a.py", "h", [(_chunk(path="a.py"), fake_embedder.embed("a"))])
    store.replace_file("b.py", "h", [(_chunk(path="b.py"), fake_embedder.embed("b"))])
    assert store.count() == 2
    store.remove_file("a.py")
    assert store.count() == 1
    assert store.known_files() == {"b.py"}


# ---------- indexer integration -----------------------------------------

def test_index_directory_picks_up_python_files(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "idx.db")
    res = index_directory(tmp_repo, store, fake_embedder)
    assert "src/auth.py" in res.rebuilt
    assert "src/models.py" in res.rebuilt
    # Common ignores must be pruned.
    assert not any(".git" in p for p in res.rebuilt)
    assert store.count() > 0


def test_index_directory_is_idempotent(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "idx.db")
    index_directory(tmp_repo, store, fake_embedder)
    res2 = index_directory(tmp_repo, store, fake_embedder)
    assert res2.rebuilt == []  # nothing changed
    assert len(res2.unchanged) >= 2


def test_index_directory_rebuilds_changed_files(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "idx.db")
    index_directory(tmp_repo, store, fake_embedder)

    # Modify one file.
    (tmp_repo / "src" / "auth.py").write_text(
        "def authenticate(u, p):\n    return True\n"
    )
    res = index_directory(tmp_repo, store, fake_embedder)
    assert res.rebuilt == ["src/auth.py"]
    assert "src/models.py" in res.unchanged


def test_index_directory_drops_deleted_files(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "idx.db")
    index_directory(tmp_repo, store, fake_embedder)
    (tmp_repo / "src" / "models.py").unlink()
    res = index_directory(tmp_repo, store, fake_embedder)
    assert "src/models.py" in res.removed
    assert "src/models.py" not in store.known_files()
