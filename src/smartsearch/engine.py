"""One object that owns config, store, embedder, indexer and searcher.

Both the CLI and the web app use this, so new features usually start here.
"""

from __future__ import annotations

import os
import platform
import subprocess
from functools import cached_property
from pathlib import Path

from . import llm
from .config import Config, load_config
from .embedder import Embedder, make_embedder
from .indexer import Indexer
from .search import Searcher, SearchFilters, SearchResult
from .store import Store


class Engine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.store = Store(self.cfg.db_path)

    @cached_property
    def embedder(self) -> Embedder:
        return make_embedder(self.cfg.embedder, self.cfg.model_name, self.cfg.batch_size)

    @cached_property
    def indexer(self) -> Indexer:
        return Indexer(self.cfg, self.store, self.embedder)

    @cached_property
    def searcher(self) -> Searcher:
        return Searcher(self.cfg, self.store, self.embedder)

    def search(
        self,
        query: str,
        limit: int | None = None,
        filters: SearchFilters | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        return self.searcher.search(query, limit, filters, mode)

    def ask(
        self, question: str, k: int = 6, filters: SearchFilters | None = None
    ) -> tuple[str, list[SearchResult]]:
        results = self.search(question, limit=k, filters=filters)
        answer = llm.ask(question, results, self.cfg.ollama_url, self.cfg.ollama_model)
        return answer, results

    def is_indexed(self, path: str) -> bool:
        return self.store.get_file(path) is not None

    @staticmethod
    def open_path(path: str, reveal: bool = False) -> None:
        """Open a file with the OS default app (or reveal it in the file manager)."""
        p = Path(path)
        target = str(p.parent if reveal and platform.system() != "Darwin" else p)
        system = platform.system()
        if system == "Windows":
            if reveal:
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(target)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", target] if reveal else ["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
