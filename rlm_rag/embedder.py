"""Local embeddings via sentence-transformers.

Lazy-loads the model on first use so test imports don't pay the cost. The
default model is small (~80MB) and runs on CPU. If you want a different
model, pass it explicitly or set RLM_RAG_EMBED_MODEL.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


DEFAULT_MODEL = os.environ.get("RLM_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Embedder:
    """Wraps a sentence-transformers model with normalize_embeddings=True so
    cosine similarity reduces to a dot product.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            # Imported lazily so `import rlm_rag.embedder` is cheap and so the
            # rest of the package can be imported in environments where
            # sentence-transformers isn't installed (e.g. chunker-only tests).
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """One text → one (dim,) float32 vector."""
        v = self._load().encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)

    def embed_batch(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """N texts → (N, dim) float32 array."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._load().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return np.asarray(v, dtype=np.float32)
