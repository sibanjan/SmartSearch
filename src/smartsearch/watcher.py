"""Keep the index live: re-index files as they are created, changed, moved or deleted."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import Indexer

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Debounces bursts of events (editors save files several times)."""

    def __init__(self, indexer: Indexer, delay: float = 2.0):
        self.indexer = indexer
        self.delay = delay
        self.pending: dict[str, str] = {}  # path -> "upsert" | "delete"
        self.lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _queue(self, path: str, action: str) -> None:
        with self.lock:
            self.pending[path] = action

    def on_created(self, e: FileSystemEvent) -> None:
        if not e.is_directory:
            self._queue(e.src_path, "upsert")

    on_modified = on_created

    def on_deleted(self, e: FileSystemEvent) -> None:
        if not e.is_directory:
            self._queue(e.src_path, "delete")

    def on_moved(self, e: FileSystemEvent) -> None:
        if not e.is_directory:
            self._queue(e.src_path, "delete")
            self._queue(e.dest_path, "upsert")

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.delay)
            with self.lock:
                batch, self.pending = self.pending, {}
            for path, action in batch.items():
                p = Path(path)
                try:
                    if action == "delete" or not p.exists():
                        self.indexer.remove_file(p)
                        log.info("removed  %s", p)
                    elif self.indexer.is_candidate(p):
                        status = self.indexer.index_file(p)
                        if status in ("added", "updated"):
                            log.info("%-8s %s", status, p)
                except Exception as exc:
                    log.warning("watch: failed on %s: %s", p, exc)

    def stop(self) -> None:
        self._stop.set()


def start_watching(indexer: Indexer) -> tuple[Observer, _Handler]:
    handler = _Handler(indexer)
    observer = Observer()
    for root in indexer.cfg.root_paths:
        if root.is_dir():
            observer.schedule(handler, str(root), recursive=True)
            log.info("Watching %s", root)
    observer.daemon = True
    observer.start()
    return observer, handler
