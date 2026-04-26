"""SQLite vector store + symbol/import/call graph.

Tables:
  chunks   one row per per-symbol chunk; embeddings as float32 BLOB.
  symbols  one row per defined symbol (functions, classes, methods, types).
  imports  one row per (file, imported_module) edge.
  calls    one row per (file, callee_name) edge — best-effort, name-based.

Search is in-memory cosine over the chunks table. The graph tables are
queried by the iterative query layer.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunker import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float          # cosine in [-1, 1]


class ChunkStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY,
            file_path    TEXT NOT NULL,
            symbol_name  TEXT NOT NULL,
            symbol_kind  TEXT NOT NULL,
            language     TEXT NOT NULL DEFAULT 'python',
            start_line   INTEGER NOT NULL,
            end_line     INTEGER NOT NULL,
            text         TEXT NOT NULL,
            embedding    BLOB NOT NULL,
            file_sha1    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);

        CREATE TABLE IF NOT EXISTS symbols (
            id           INTEGER PRIMARY KEY,
            file_path    TEXT NOT NULL,
            name         TEXT NOT NULL,
            kind         TEXT NOT NULL,
            language     TEXT NOT NULL DEFAULT 'python',
            start_line   INTEGER NOT NULL,
            end_line     INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);

        CREATE TABLE IF NOT EXISTS imports (
            id           INTEGER PRIMARY KEY,
            file_path    TEXT NOT NULL,
            module       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_path);
        CREATE INDEX IF NOT EXISTS idx_imports_mod ON imports(module);

        CREATE TABLE IF NOT EXISTS calls (
            id           INTEGER PRIMARY KEY,
            file_path    TEXT NOT NULL,
            caller       TEXT NOT NULL,    -- enclosing symbol name (e.g. "User.greet")
            callee       TEXT NOT NULL     -- leaf name being called (e.g. "authenticate")
        );
        CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee);
        CREATE INDEX IF NOT EXISTS idx_calls_file   ON calls(file_path);
        """)
        self.db.commit()

    # ---------- writes -------------------------------------------------

    def replace_file(
        self,
        file_path: str,
        file_sha1: str,
        chunks_with_embeddings: list[tuple[Chunk, np.ndarray]],
    ) -> None:
        """Atomically replace all chunks/symbols/imports/calls for a file."""
        cur = self.db.cursor()
        cur.execute("DELETE FROM chunks  WHERE file_path = ?", (file_path,))
        cur.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        cur.execute("DELETE FROM imports WHERE file_path = ?", (file_path,))
        cur.execute("DELETE FROM calls   WHERE file_path = ?", (file_path,))

        cur.executemany(
            """INSERT INTO chunks
               (file_path, symbol_name, symbol_kind, language, start_line,
                end_line, text, embedding, file_sha1)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.file_path, c.symbol_name, c.symbol_kind, c.language,
                    c.start_line, c.end_line, c.text,
                    e.astype(np.float32).tobytes(), file_sha1,
                )
                for c, e in chunks_with_embeddings
            ],
        )

        # symbols: skip "<module>" / "<file>" pseudo-chunks.
        cur.executemany(
            "INSERT INTO symbols (file_path, name, kind, language, start_line, end_line) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (c.file_path, c.symbol_name, c.symbol_kind, c.language,
                 c.start_line, c.end_line)
                for c, _ in chunks_with_embeddings
                if not c.symbol_name.startswith("<")
            ],
        )

        # imports: dedup at file level (every chunk in a file carries the same imports list).
        seen_imports: set[str] = set()
        if chunks_with_embeddings:
            for m in chunks_with_embeddings[0][0].imports:
                if m not in seen_imports:
                    seen_imports.add(m)
        cur.executemany(
            "INSERT INTO imports (file_path, module) VALUES (?, ?)",
            [(file_path, m) for m in sorted(seen_imports)],
        )

        # calls: per-chunk, dedup on (caller, callee).
        call_rows: set[tuple[str, str, str]] = set()
        for c, _ in chunks_with_embeddings:
            if c.symbol_name.startswith("<"):
                continue
            for callee in c.calls:
                call_rows.add((file_path, c.symbol_name, callee))
        cur.executemany(
            "INSERT INTO calls (file_path, caller, callee) VALUES (?, ?, ?)",
            list(call_rows),
        )

        self.db.commit()

    def remove_file(self, file_path: str) -> None:
        for tbl in ("chunks", "symbols", "imports", "calls"):
            self.db.execute(f"DELETE FROM {tbl} WHERE file_path = ?", (file_path,))
        self.db.commit()

    # ---------- chunk reads --------------------------------------------

    def file_sha1(self, file_path: str) -> str | None:
        row = self.db.execute(
            "SELECT file_sha1 FROM chunks WHERE file_path = ? LIMIT 1",
            (file_path,),
        ).fetchone()
        return row[0] if row else None

    def known_files(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT DISTINCT file_path FROM chunks")}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[Chunk]:
        return [
            self._row_to_chunk(r)
            for r in self.db.execute(
                "SELECT file_path, symbol_name, symbol_kind, language, start_line, end_line, text FROM chunks"
            )
        ]

    def all_embeddings(self) -> tuple[np.ndarray, list[Chunk]]:
        rows = self.db.execute(
            """SELECT file_path, symbol_name, symbol_kind, language,
                      start_line, end_line, text, embedding
               FROM chunks"""
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        embs = np.stack([np.frombuffer(r[7], dtype=np.float32) for r in rows])
        chunks = [self._row_to_chunk(r[:7]) for r in rows]
        return embs, chunks

    def search(self, query_emb: np.ndarray, k: int = 20) -> list[SearchHit]:
        embs, chunks = self.all_embeddings()
        if len(chunks) == 0:
            return []
        sims = embs @ query_emb.astype(np.float32)
        k = min(k, len(chunks))
        top = np.argpartition(-sims, k - 1)[:k]
        ranked = top[np.argsort(-sims[top])]
        return [SearchHit(chunk=chunks[i], score=float(sims[i])) for i in ranked]

    # ---------- graph reads --------------------------------------------

    def find_symbol(self, name: str) -> list[dict]:
        """Exact-match lookup. For methods, accepts 'Class.method'."""
        rows = self.db.execute(
            "SELECT file_path, name, kind, language, start_line, end_line FROM symbols WHERE name = ?",
            (name,),
        ).fetchall()
        return [
            {"file_path": r[0], "name": r[1], "kind": r[2], "language": r[3],
             "start_line": r[4], "end_line": r[5]}
            for r in rows
        ]

    def grep_symbol(self, pattern: str, limit: int = 50) -> list[dict]:
        """LIKE-match lookup. Useful for partial / case-insensitive search."""
        rows = self.db.execute(
            """SELECT file_path, name, kind, language, start_line, end_line
               FROM symbols WHERE name LIKE ? COLLATE NOCASE LIMIT ?""",
            (f"%{pattern}%", limit),
        ).fetchall()
        return [
            {"file_path": r[0], "name": r[1], "kind": r[2], "language": r[3],
             "start_line": r[4], "end_line": r[5]}
            for r in rows
        ]

    def callers_of(self, symbol_name: str) -> list[dict]:
        """Return {file, caller} where some chunk's body called `symbol_name`."""
        rows = self.db.execute(
            "SELECT DISTINCT file_path, caller FROM calls WHERE callee = ? LIMIT 200",
            (symbol_name,),
        ).fetchall()
        return [{"file_path": r[0], "caller": r[1]} for r in rows]

    def imports_of(self, file_path: str) -> list[str]:
        rows = self.db.execute(
            "SELECT module FROM imports WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [r[0] for r in rows]

    def files_importing(self, module: str) -> list[str]:
        rows = self.db.execute(
            "SELECT file_path FROM imports WHERE module = ? OR module LIKE ?",
            (module, f"{module}.%"),
        ).fetchall()
        return [r[0] for r in rows]

    def get_chunk(self, file_path: str, symbol_name: str) -> Chunk | None:
        row = self.db.execute(
            """SELECT file_path, symbol_name, symbol_kind, language,
                      start_line, end_line, text
               FROM chunks WHERE file_path = ? AND symbol_name = ? LIMIT 1""",
            (file_path, symbol_name),
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    # ---------- internal -----------------------------------------------

    def _row_to_chunk(self, r: Iterable) -> Chunk:
        r = tuple(r)
        return Chunk(
            file_path=r[0],
            symbol_name=r[1],
            symbol_kind=r[2],
            language=r[3],
            start_line=r[4],
            end_line=r[5],
            text=r[6],
        )

    def close(self) -> None:
        self.db.close()
