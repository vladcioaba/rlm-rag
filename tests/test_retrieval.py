"""Tests for BM25, RRF, and the retrieval pipeline."""

from __future__ import annotations

from rlm_rag.chunker import Chunk
from rlm_rag.retrieval import BM25Index, _tokenize, rrf_fuse, retrieve
from rlm_rag.store import ChunkStore


def test_tokenize_splits_camel_and_snake():
    toks = _tokenize("calculate_total_price + UserSession")
    # subtokens preserve both the raw form and the splits
    assert "calculate" in toks
    assert "total" in toks
    assert "price" in toks
    assert "user" in toks
    assert "session" in toks


def test_bm25_ranks_keyword_matches_higher():
    bm = BM25Index.build([
        "this chunk talks about authentication and login",
        "completely unrelated rendering of html templates",
        "user authentication and session management",
    ])
    scores = bm.score("authentication user")
    # Doc 2 (idx=2) mentions both terms; doc 1 (idx=1) mentions neither.
    assert scores[2] > scores[1]
    assert scores[0] > scores[1]


def test_bm25_returns_zeros_when_empty():
    bm = BM25Index.build([])
    assert bm.score("anything").shape == (0,)


def test_rrf_fuses_two_rankings():
    # In ranking A, doc 0 is top; in ranking B, doc 1 is top.
    # RRF should rank them similarly because each appears at rank 0 in one list.
    fused = rrf_fuse([[0, 1, 2], [1, 0, 2]])
    ids = [d for d, _ in fused]
    # doc 0 and 1 both score 1/61 + 1/62, doc 2 scores 1/63 + 1/63 — lower.
    assert ids[-1] == 2


def test_rrf_handles_single_ranking():
    fused = rrf_fuse([[2, 0, 1]])
    ids = [d for d, _ in fused]
    assert ids == [2, 0, 1]


def test_retrieve_modes_dont_crash(tmp_path, fake_embedder):
    """End-to-end retrieve pipeline runs in all three modes against a small store."""
    store = ChunkStore(tmp_path / "r.db")
    chunks = [
        Chunk(file_path=f"f{i}.py", symbol_name=f"foo{i}",
              symbol_kind="function", start_line=1, end_line=2,
              text=text, language="python")
        for i, text in enumerate([
            "auth login user password",
            "render html template page",
            "user session logout",
            "database connection pool",
        ])
    ]
    embs = fake_embedder.embed_batch([c.text for c in chunks])
    store.replace_file("file.py", "h", list(zip(chunks, embs)))

    for mode in ("cosine", "bm25", "hybrid"):
        hits = retrieve(
            "auth user login", store, fake_embedder,
            top_k=2, mode=mode,
        )
        assert 0 < len(hits) <= 2
