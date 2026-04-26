"""Index a directory of source files into a ChunkStore.

Walks the tree, dispatches per file extension to the right chunker, hashes
each file, re-chunks and re-embeds only changed files.

For large repos, chunking is parallelized across files via a ThreadPoolExecutor
(chunkers release the GIL on I/O via `path.read_text()`, and the Python
chunker via stdlib parsing is fast enough that thread parallelism is fine).
Embedding is batched into one big call to amortize sentence-transformers
warmup and exploit per-batch GPU/CPU parallelism inside the model.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import Chunk, chunk_file, language_of
from .embedder import Embedder
from .store import ChunkStore


DEFAULT_IGNORES = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt",
    ".venv", "venv", "env",
    "dist", "build", "target", ".gradle",
    "cmake-build-debug", "cmake-build-release",
    "obj", "bin", "out",
    ".rlm-rag",  # don't index our own state
}

# Directory-name *prefixes* that mean "build artifact, don't index".
# Catches build.axmol, build.tests, build.dev, build-android, etc.
DEFAULT_IGNORE_PREFIXES = ("build.", "build-", "cmake-build-")


@dataclass
class IndexResult:
    rebuilt: list[str]
    unchanged: list[str]
    removed: list[str]
    skipped: list[str]
    total_chunks: int


def _is_ignored(part: str) -> bool:
    if part in DEFAULT_IGNORES:
        return True
    return any(part.startswith(p) for p in DEFAULT_IGNORE_PREFIXES)


def _walk_source(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(_is_ignored(part) for part in path.parts):
            continue
        if language_of(path) is None:
            continue
        out.append(path)
    out.sort()
    return out


def index_directory(
    root: Path,
    store: ChunkStore,
    embedder: Embedder | None,
    progress: Callable[[str, str], None] | None = None,
    workers: int = 8,
    embed_batch_size: int = 256,
    graph_only: bool = False,
) -> IndexResult:
    """Index every supported source file under `root` into `store`.

    Stages:
      1. Walk + hash + read every file in parallel; partition into
         {unchanged, needs_rebuild, skipped}.
      2. Chunk every needs_rebuild file in parallel (CPU-bound but
         GIL-friendly).
      3. Embed all chunks across all files in batched calls (one
         sentence-transformers call per `embed_batch_size`).
      4. Sequentially write each file's (chunks, embeddings) into SQLite.

    Steps 1-3 dominate runtime on large repos; (4) is fast.
    """
    root = root.resolve()
    files = _walk_source(root)

    seen: set[str] = set()
    rebuilt: list[str] = []
    unchanged: list[str] = []
    skipped: list[str] = []

    # ---- Stage 1: read + hash + decide what to rebuild ----------------

    def _classify(path: Path) -> tuple[Path, str, str | None, str]:
        rel = str(path.relative_to(root))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return path, rel, None, "skipped"
        sha = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        return path, rel, text, sha

    classified: list[tuple[Path, str, str | None, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        classified = list(pool.map(_classify, files))

    pending: list[tuple[Path, str, str, str]] = []  # (path, rel, text, sha)
    for path, rel, text, sha in classified:
        seen.add(rel)
        if text is None:
            skipped.append(rel)
            continue
        if store.file_sha1(rel) == sha:
            unchanged.append(rel)
            if progress:
                progress(rel, "unchanged")
            continue
        pending.append((path, rel, text, sha))

    # ---- Stage 2: chunk pending files in parallel ---------------------

    def _chunk_one(args):
        path, rel, text, sha = args
        chunks = chunk_file(path)
        for c in chunks:
            c.file_path = rel
        return rel, sha, chunks

    chunked: list[tuple[str, str, list[Chunk]]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunked = list(pool.map(_chunk_one, pending))

    # ---- Stage 3: batch-embed ALL pending chunks at once --------------

    flat_chunks: list[Chunk] = []
    file_chunk_ranges: list[tuple[str, str, int, int]] = []  # (rel, sha, start, end)
    for rel, sha, chunks in chunked:
        if not chunks:
            store.remove_file(rel)
            skipped.append(rel)
            continue
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        end = len(flat_chunks)
        file_chunk_ranges.append((rel, sha, start, end))

    if flat_chunks and not graph_only:
        # Single batch call — sentence-transformers internally chunks by
        # embed_batch_size, which is the per-call batch size we pass.
        all_embs = embedder.embed_batch(
            [c.text for c in flat_chunks],
            batch_size=embed_batch_size,
        )
    elif graph_only:
        # Use zero-vectors so the SQL schema (NOT NULL embedding) stays
        # valid. They make retrieval useless, which is the correct trade —
        # caller asked for graph-only.
        import numpy as np
        dim = 1
        all_embs = np.zeros((len(flat_chunks), dim), dtype=np.float32)
    else:
        all_embs = []

    # ---- Stage 4: write per file (sequential — SQLite is single-writer) ---

    for rel, sha, start, end in file_chunk_ranges:
        chunks_slice = flat_chunks[start:end]
        embs_slice = all_embs[start:end]
        store.replace_file(rel, sha, list(zip(chunks_slice, embs_slice)))
        rebuilt.append(rel)
        if progress:
            progress(rel, "rebuilt")

    removed: list[str] = []
    for rel in store.known_files() - seen:
        store.remove_file(rel)
        removed.append(rel)
        if progress:
            progress(rel, "removed")

    return IndexResult(
        rebuilt=rebuilt,
        unchanged=unchanged,
        removed=removed,
        skipped=skipped,
        total_chunks=store.count(),
    )
