"""Shared fixtures for the rlm-rag test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_repo(tmp_path):
    """A small Python repo for indexing tests."""
    root = tmp_path / "fake_repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text(
        '"""Authentication module."""\n'
        "import hashlib\n\n"
        "USERS = {}\n\n"
        "def hash_password(pw):\n"
        '    """Return a sha256 hex of the password."""\n'
        "    return hashlib.sha256(pw.encode()).hexdigest()\n\n"
        "def authenticate(user, password):\n"
        '    """Check user/password against USERS."""\n'
        "    return USERS.get(user) == hash_password(password)\n"
    )
    (root / "src" / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n\n"
        "    def display(self):\n"
        "        return f'User: {self.name}'\n\n"
        "class Session:\n"
        "    pass\n"
    )
    # An ignored dir
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("garbage\n")
    return root


@pytest.fixture
def fake_embedder():
    """Cheap deterministic embedder. Hashes text to a fixed-dim vector so the
    store/indexer can be tested without sentence-transformers installed.
    """
    import hashlib
    import numpy as np

    DIM = 16

    class FakeEmbedder:
        @property
        def dim(self):
            return DIM

        def _vec(self, text: str) -> np.ndarray:
            h = hashlib.sha256(text.encode()).digest()
            # Take first DIM*4 bytes as float32, normalize.
            v = np.frombuffer(h[: DIM * 4].ljust(DIM * 4, b"\0"),
                              dtype=np.float32).copy()
            n = float(np.linalg.norm(v))
            if n > 0:
                v /= n
            return v.astype(np.float32)

        def embed(self, text: str) -> np.ndarray:
            return self._vec(text)

        def embed_batch(self, texts, batch_size=32):
            import numpy as np
            texts = list(texts)
            if not texts:
                return np.zeros((0, self.dim), dtype=np.float32)
            return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    return FakeEmbedder()
