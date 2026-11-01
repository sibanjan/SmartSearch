"""Walk the configured folders and keep the index in sync (incremental)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .chunker import chunk_text
from .config import Config
from .embedder import Embedder
from .extractors import extract_text
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class IndexReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[str] = field(default_factory=list)


class Indexer:
    def __init__(self, cfg: Config, store: Store, embedder: Embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self._check_embedder()

    def _check_embedder(self) -> None:
        stored = self.store.get_meta("embedder")
        if stored and stored != self.embedder.name:
            log.warning(
                "Embedder changed (%s -> %s); clearing index so it can be rebuilt.",
                stored,
                self.embedder.name,
            )
            self.store.clear()
        self.store.set_meta("embedder", self.embedder.name)

    # ---- filtering -------------------------------------------------------
    def is_candidate(self, path: Path) -> bool:
        if path.suffix.lower() not in self.cfg.extension_set:
            return False
        # Only judge the part of the path *below* its root, so a root that
        # itself lives inside a hidden/excluded folder still works.
        root = next((r for r in self.cfg.root_paths if _is_relative_to(path, r)), None)
        if root is None:
            return False
        rel_parts = path.relative_to(root).parts if path != root else (path.name,)
        if set(rel_parts[:-1]) & set(self.cfg.exclude_dirs):
            return False
        if self.cfg.skip_hidden and any(p.startswith(".") for p in rel_parts):
            return False
        return True

    def iter_files(self) -> Iterator[Path]:
        excluded = set(self.cfg.exclude_dirs)
        for root in self.cfg.root_paths:
            if root.is_file():
                if self.is_candidate(root):
                    yield root
                continue
            if not root.is_dir():
                log.warning("Root does not exist: %s", root)
                continue
            for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in excluded and not (self.cfg.skip_hidden and d.startswith("."))
                ]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if self.is_candidate(p):
                        yield p

    # ---- indexing --------------------------------------------------------
    def index_file(self, path: Path, force: bool = False) -> str:
        """Index one file. Returns 'added' | 'updated' | 'unchanged' | 'skipped'."""
        try:
            st = path.stat()
        except OSError:
            return "skipped"
        if st.st_size > self.cfg.max_file_size_mb * 1024 * 1024:
            return "skipped"
        key = str(path)
        existing = self.store.get_file(key)
        if existing and not force and existing.mtime == st.st_mtime and existing.size == st.st_size:
            return "unchanged"

        text = extract_text(path)
        header = f"{path.name}\n"  # gives each chunk a hint of where it came from
        chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        vecs = self.embedder.encode([header + c for c in chunks]) if chunks else None
        self.store.upsert_file(key, st.st_size, st.st_mtime, chunks, vecs)
        return "updated" if existing else "added"

    def remove_file(self, path: Path) -> None:
        self.store.delete_file(str(path))

    def run(
        self,
        force: bool = False,
        progress: Callable[[Path, int], None] | None = None,
    ) -> IndexReport:
        report = IndexReport()
        seen: set[str] = set()
        for i, path in enumerate(self.iter_files()):
            report.scanned += 1
            seen.add(str(path))
            if progress:
                progress(path, i)
            try:
                status = self.index_file(path, force=force)
            except Exception as exc:  # keep going on bad files
                log.warning("Failed to index %s: %s", path, exc)
                report.failed.append(str(path))
                continue
            if status == "added":
                report.added += 1
            elif status == "updated":
                report.updated += 1
            elif status == "unchanged":
                report.unchanged += 1

        # Drop files that disappeared or are now out of scope
        for p in self.store.all_paths():
            if p not in seen:
                self.store.delete_file(p)
                report.removed += 1
        return report


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False
