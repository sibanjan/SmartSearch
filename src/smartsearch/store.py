"""SQLite-backed index: file metadata, text chunks, FTS5 keyword index, vectors.

Vectors are stored as float32 blobs and loaded into an in-memory numpy matrix
for brute-force cosine search. That comfortably handles a few hundred thousand
chunks on a laptop; swap in FAISS / LanceDB / sqlite-vec here if you outgrow it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    ext TEXT,
    size INTEGER,
    mtime REAL,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    vec BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, name, chunk_id UNINDEXED, tokenize = 'porter unicode61'
);
"""


@dataclass
class FileRecord:
    id: int
    path: str
    name: str
    ext: str
    size: int
    mtime: float


class Store:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()
        self._vec_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ---- meta ------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # ---- files -----------------------------------------------------------
    def get_file(self, path: str) -> FileRecord | None:
        r = self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return (
            FileRecord(r["id"], r["path"], r["name"], r["ext"], r["size"], r["mtime"])
            if r
            else None
        )

    def all_paths(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT path FROM files")]

    def upsert_file(
        self, path: str, size: int, mtime: float, chunks: list[str], vecs: np.ndarray
    ) -> None:
        p = Path(path)
        with self.lock, self.conn:
            self._delete_file_rows(path)
            cur = self.conn.execute(
                "INSERT INTO files(path,name,ext,size,mtime,indexed_at) VALUES(?,?,?,?,?,?)",
                (path, p.name, p.suffix.lower(), size, mtime, time.time()),
            )
            file_id = cur.lastrowid
            # Always index the file name itself so name-only matches are possible
            # even for files with no extractable text.
            if not chunks:
                chunks = [p.name]
                vecs = None
            for i, text in enumerate(chunks):
                blob = vecs[i].astype(np.float32).tobytes() if vecs is not None else None
                c = self.conn.execute(
                    "INSERT INTO chunks(file_id,ord,text,vec) VALUES(?,?,?,?)",
                    (file_id, i, text, blob),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(text,name,chunk_id) VALUES(?,?,?)",
                    (text, p.stem.replace("_", " ").replace("-", " "), c.lastrowid),
                )
        self._vec_cache = None

    def _delete_file_rows(self, path: str) -> None:
        row = self.conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if not row:
            return
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id=?)",
            (row["id"],),
        )
        self.conn.execute("DELETE FROM files WHERE id=?", (row["id"],))

    def delete_file(self, path: str) -> None:
        with self.lock, self.conn:
            self._delete_file_rows(path)
        self._vec_cache = None

    def clear(self) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM files")
        self._vec_cache = None

    # ---- retrieval -------------------------------------------------------
    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (chunk_ids, matrix) for all chunks with embeddings (cached)."""
        if self._vec_cache is None:
            with self.lock:
                rows = self.conn.execute(
                    "SELECT id, vec FROM chunks WHERE vec IS NOT NULL"
                ).fetchall()
            if rows:
                ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
                mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
            else:
                ids, mat = np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
            self._vec_cache = (ids, mat)
        return self._vec_cache

    def fts_search(self, match_expr: str, limit: int) -> list[tuple[int, float]]:
        with self.lock:
            try:
                rows = self.conn.execute(
                    "SELECT chunk_id, bm25(chunks_fts, 1.0, 2.0) AS score FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (match_expr, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(int(r["chunk_id"]), float(r["score"])) for r in rows]

    def filename_search(self, terms: list[str], limit: int) -> list[int]:
        """Chunk ids (first chunk of each file) whose file name contains all terms."""
        if not terms:
            return []
        where = " AND ".join(["lower(f.name) LIKE ?"] * len(terms))
        params = [f"%{t.lower()}%" for t in terms] + [limit]
        with self.lock:
            rows = self.conn.execute(
                f"SELECT c.id FROM files f JOIN chunks c ON c.file_id=f.id AND c.ord=0 "
                f"WHERE {where} ORDER BY f.mtime DESC LIMIT ?",
                params,
            ).fetchall()
        return [r[0] for r in rows]

    def chunk_details(self, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not chunk_ids:
            return {}
        q = ",".join("?" * len(chunk_ids))
        with self.lock:
            rows = self.conn.execute(
                f"SELECT c.id AS chunk_id, c.ord, c.text, f.id AS file_id, f.path, f.name, "
                f"f.ext, f.size, f.mtime FROM chunks c JOIN files f ON f.id=c.file_id "
                f"WHERE c.id IN ({q})",
                chunk_ids,
            ).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def stats(self) -> dict:
        with self.lock:
            files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            by_ext = self.conn.execute(
                "SELECT ext, COUNT(*) n FROM files GROUP BY ext ORDER BY n DESC"
            ).fetchall()
        return {
            "files": files,
            "chunks": chunks,
            "by_extension": {r["ext"] or "(none)": r["n"] for r in by_ext},
            "embedder": self.get_meta("embedder"),
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1e6, 2) if self.path.exists() else 0,
        }

    def close(self) -> None:
        self.conn.close()
