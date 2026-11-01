"""Hybrid search: semantic vectors + BM25 keywords + file-name matching,
combined with Reciprocal Rank Fusion (RRF), then grouped per file."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .embedder import Embedder
from .store import Store

RRF_K = 60
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class SearchFilters:
    extensions: list[str] = field(default_factory=list)  # e.g. [".pdf", ".docx"]
    path_contains: str | None = None
    modified_after: float | None = None  # unix timestamp


@dataclass
class SearchResult:
    path: str
    name: str
    ext: str
    score: float
    snippet: str
    chunk_text: str
    modified: float
    size: int
    matched_by: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 expression (prefix-matching, OR'ed)."""
    terms = [t for t in _WORD.findall(q.lower()) if len(t) > 1]
    return " OR ".join(f'"{t}"*' for t in terms)


def _snippet(text: str, query: str, width: int = 280) -> str:
    terms = [t for t in _WORD.findall(query.lower()) if len(t) > 2]
    low = text.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    snip = text[start : start + width].replace("\n", " ").strip()
    return ("…" if start > 0 else "") + snip + ("…" if start + width < len(text) else "")


class Searcher:
    def __init__(self, cfg: Config, store: Store, embedder: Embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

    def _semantic(self, query: str, k: int) -> list[int]:
        ids, mat = self.store.vectors()
        if len(ids) == 0:
            return []
        qv = self.embedder.encode([query], query=True)[0]
        if mat.shape[1] != qv.shape[0]:
            return []  # index built with a different embedder
        sims = mat @ qv
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [int(ids[i]) for i in top if sims[i] > 0.05]

    def search(
        self,
        query: str,
        limit: int | None = None,
        filters: SearchFilters | None = None,
        mode: str = "hybrid",  # hybrid | semantic | keyword
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        limit = limit or self.cfg.default_limit
        filters = filters or SearchFilters()
        pool = max(limit * 10, 100)

        ranked: dict[str, tuple[list[int], float]] = {}
        if mode in ("hybrid", "semantic"):
            ranked["semantic"] = (self._semantic(query, pool), self.cfg.semantic_weight)
        if mode in ("hybrid", "keyword"):
            expr = _fts_query(query)
            kw = [cid for cid, _ in self.store.fts_search(expr, pool)] if expr else []
            ranked["keyword"] = (kw, self.cfg.keyword_weight)
            terms = [t for t in _WORD.findall(query.lower()) if len(t) > 1]
            ranked["filename"] = (self.store.filename_search(terms, pool), self.cfg.filename_weight)

        # Reciprocal Rank Fusion
        scores: dict[int, float] = {}
        sources: dict[int, set[str]] = {}
        for src, (ids, weight) in ranked.items():
            for rank, cid in enumerate(ids):
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank + 1)
                sources.setdefault(cid, set()).add(src)

        details = self.store.chunk_details(list(scores))
        exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in filters.extensions}

        # Group by file, keep the best chunk; small bonus for multiple matching chunks
        best: dict[int, tuple[float, int]] = {}
        hits: dict[int, int] = {}
        file_sources: dict[int, set[str]] = {}
        for cid, s in scores.items():
            row = details.get(cid)
            if row is None:
                continue
            if exts and row["ext"] not in exts:
                continue
            if filters.path_contains and filters.path_contains.lower() not in row["path"].lower():
                continue
            if filters.modified_after and row["mtime"] < filters.modified_after:
                continue
            fid = row["file_id"]
            hits[fid] = hits.get(fid, 0) + 1
            file_sources.setdefault(fid, set()).update(sources.get(cid, set()))
            if fid not in best or s > best[fid][0]:
                best[fid] = (s, cid)

        results = []
        for fid, (s, cid) in best.items():
            row = details[cid]
            matched = file_sources.get(fid, set())
            final = s * (1 + 0.1 * min(hits[fid] - 1, 5))
            results.append(
                SearchResult(
                    path=row["path"],
                    name=row["name"],
                    ext=row["ext"],
                    score=round(final * 1000, 3),
                    snippet=_snippet(row["text"], query),
                    chunk_text=row["text"],
                    modified=row["mtime"],
                    size=row["size"],
                    matched_by=sorted(matched),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]


def parse_since(value: str | None) -> float | None:
    """'7d', '12h', '2w', '30m' or 'YYYY-MM-DD' -> unix timestamp."""
    if not value:
        return None
    m = re.fullmatch(r"(\d+)\s*([mhdw])", value.strip().lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        return time.time() - n * secs
    from datetime import datetime

    return datetime.fromisoformat(value).timestamp()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def short_path(p: str) -> str:
    home = str(Path.home())
    return "~" + p[len(home) :] if p.startswith(home) else p
