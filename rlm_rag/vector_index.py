"""Vector-search backend abstraction.

The default `NumpyBackend` does in-memory cosine search and is fine up to
~tens of thousands of chunks. The `FaissBackend` activates if `faiss` is
installed and the chunk count exceeds a threshold; otherwise we transparently
fall back to numpy.

The store currently keeps embeddings in SQLite as float32 BLOBs and rebuilds
the in-memory matrix per query. For faiss we build an index lazily and cache
it; cache invalidates on write.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorBackend(Protocol):
    def search(self, embs: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Returns list of (row_index, score) sorted by score desc."""
        ...

    @property
    def name(self) -> str:
        ...


class NumpyBackend:
    name = "numpy"

    def search(self, embs: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(embs) == 0:
            return []
        sims = embs @ query.astype(np.float32)
        k = min(k, len(embs))
        top = np.argpartition(-sims, k - 1)[:k]
        ranked = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in ranked]


class FaissBackend:
    """Inner-product search via faiss. Vectors must be unit-normalized for
    cosine equivalence (the embedder already does this).
    """

    name = "faiss"

    def __init__(self):
        import faiss  # noqa: PLC0415
        self.faiss = faiss
        self._index = None
        self._index_for_shape: tuple[int, int] | None = None

    def _build(self, embs: np.ndarray):
        idx = self.faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype(np.float32))
        self._index = idx
        self._index_for_shape = embs.shape

    def search(self, embs: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(embs) == 0:
            return []
        if self._index is None or self._index_for_shape != embs.shape:
            self._build(embs)
        k = min(k, len(embs))
        D, I = self._index.search(query.reshape(1, -1).astype(np.float32), k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0])]


def auto_backend(chunk_count: int, threshold: int = 50_000) -> VectorBackend:
    """Use faiss above `threshold` chunks if available, otherwise numpy."""
    if chunk_count >= threshold:
        try:
            return FaissBackend()
        except Exception:
            pass
    return NumpyBackend()
