"""Retrieval strategies: cosine-only, BM25-only, or hybrid via reciprocal
rank fusion. Optional cross-encoder reranking on top.

BM25 is implemented in-process (no `rank_bm25` dep) over a simple word-token
representation. It's not the fastest possible BM25 but it's correct, ~50
lines, and zero-dep.

The cross-encoder is optional — if `sentence-transformers` isn't installed
or the model can't load, rerank() falls back to the input order.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .chunker import Chunk
from .store import ChunkStore, SearchHit


# ---------- BM25 --------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Split CamelCase / snake_case / dotted into searchable subtokens.

    e.g. "User.greet" -> ["User", "user", "greet"]
         "calculate_total_price" -> ["calculate", "total", "price"]
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        out.append(raw.lower())
        # snake_case
        for part in raw.split("_"):
            if part:
                out.append(part.lower())
        # CamelCase
        for part in re.findall(r"[A-Z]+[a-z0-9]*|[a-z0-9]+", raw):
            out.append(part.lower())
    return out


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed corpus."""

    docs: list[list[str]]
    doc_freqs: list[Counter]
    df: Counter           # document frequency per term
    avg_dl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: Iterable[str]) -> "BM25Index":
        docs = [_tokenize(t) for t in texts]
        df: Counter = Counter()
        doc_freqs: list[Counter] = []
        for d in docs:
            tf = Counter(d)
            doc_freqs.append(tf)
            for term in tf:
                df[term] += 1
        avg_dl = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0
        return cls(docs=docs, doc_freqs=doc_freqs, df=df, avg_dl=avg_dl)

    def score(self, query: str) -> np.ndarray:
        if not self.docs:
            return np.zeros(0, dtype=np.float32)
        q_terms = _tokenize(query)
        N = len(self.docs)
        scores = np.zeros(N, dtype=np.float32)
        for term in q_terms:
            n_t = self.df.get(term, 0)
            if n_t == 0:
                continue
            idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
            for i, tf in enumerate(self.doc_freqs):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# ---------- Reciprocal Rank Fusion --------------------------------------

def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Standard RRF: score(d) = sum over rankings of 1 / (k + rank(d)).
    Returns list of (doc_index, score) sorted by score desc.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------- Reranker (optional) -----------------------------------------

DEFAULT_RERANK_MODEL = os.environ.get(
    "RLM_RAG_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Curated alternatives users can drop in via --rerank-model or RLM_RAG_RERANK_MODEL.
# These are general-purpose; a code-tuned reranker is on the roadmap.
KNOWN_RERANK_MODELS = {
    # Default — general web text, fast (~22M params).
    "ms-marco-mini":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Larger, slower, generally better.
    "ms-marco-base":    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # Multilingual (still general — not code-specific).
    "bge-reranker-base":   "BAAI/bge-reranker-base",
    "bge-reranker-large":  "BAAI/bge-reranker-large",
    # Mixedbread mxbai — strong on technical text, including code-adjacent.
    "mxbai-rerank-base":   "mixedbread-ai/mxbai-rerank-base-v1",
}


def resolve_rerank_model(name_or_alias: str | None) -> str:
    """Accept a HuggingFace path OR a short alias from KNOWN_RERANK_MODELS."""
    if not name_or_alias:
        return DEFAULT_RERANK_MODEL
    return KNOWN_RERANK_MODELS.get(name_or_alias, name_or_alias)


class Reranker:
    """Lazy-loads a cross-encoder. If the model can't load, rerank() is a no-op."""

    def __init__(self, model_name: str | None = None):
        self.model_name = resolve_rerank_model(model_name)
        self._model = None
        self._loaded = False
        self._failed = False

    def _load(self):
        if self._loaded:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._failed = True
            self._model = None
        self._loaded = True
        return self._model

    def available(self) -> bool:
        self._load()
        return self._model is not None

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits or not self.available():
            return hits
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [
            SearchHit(chunk=hits[i].chunk, score=float(scores[i]))
            for i in order
        ]


# ---------- High-level retrieve -----------------------------------------

def retrieve(
    query: str,
    store: ChunkStore,
    embedder,
    top_k: int = 20,
    mode: str = "hybrid",          # "cosine" | "bm25" | "hybrid"
    rerank: bool = False,
    reranker: Reranker | None = None,
    rerank_model: str | None = None,  # alias or HF path; ignored if reranker is given
    over_fetch: int = 4,            # fetch top_k * over_fetch then rerank/fuse
) -> list[SearchHit]:
    """Pluggable retrieval pipeline. `over_fetch` widens the candidate pool
    before rerank/fusion to give those steps room to reorder.
    """
    embs, chunks = store.all_embeddings()
    if len(chunks) == 0:
        return []

    pool_k = min(top_k * over_fetch, len(chunks))

    cosine_idx: list[int] = []
    bm25_idx: list[int] = []

    if mode in ("cosine", "hybrid"):
        q_emb = embedder.embed(query)
        sims = embs @ q_emb.astype(np.float32)
        cosine_idx = list(np.argsort(-sims)[:pool_k])

    if mode in ("bm25", "hybrid"):
        bm = BM25Index.build([c.text for c in chunks])
        scores = bm.score(query)
        bm25_idx = list(np.argsort(-scores)[:pool_k])

    if mode == "cosine":
        ranking = cosine_idx
    elif mode == "bm25":
        ranking = bm25_idx
    else:  # hybrid
        fused = rrf_fuse([cosine_idx, bm25_idx])
        ranking = [idx for idx, _ in fused[:pool_k]]

    # Build SearchHits with placeholder scores; rerank may overwrite them.
    hits: list[SearchHit] = []
    seen: set[int] = set()
    for r, idx in enumerate(ranking):
        if idx in seen:
            continue
        seen.add(idx)
        hits.append(SearchHit(chunk=chunks[idx], score=1.0 / (r + 1)))

    if rerank:
        if reranker is None:
            reranker = Reranker(model_name=rerank_model)
        hits = reranker.rerank(query, hits)

    return hits[:top_k]
