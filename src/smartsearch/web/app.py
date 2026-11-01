"""Local web UI + JSON API (FastAPI). Binds to 127.0.0.1 by default."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..engine import Engine
from ..search import SearchFilters, parse_since

STATIC = Path(__file__).parent / "static"


class OpenRequest(BaseModel):
    path: str
    reveal: bool = False


def create_app(engine: Engine, watch: bool = True, index_on_start: bool = True) -> FastAPI:
    state = {"indexing": False, "last_report": None}

    def _index_then_watch():
        state["indexing"] = True
        try:
            state["last_report"] = engine.indexer.run().__dict__
        finally:
            state["indexing"] = False
        if watch:
            from ..watcher import start_watching

            start_watching(engine.indexer)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _ = engine.embedder  # load model up front
        if index_on_start:
            threading.Thread(target=_index_then_watch, daemon=True).start()
        yield

    app = FastAPI(title="SmartSearch", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    def index_page():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/search")
    def api_search(
        q: str,
        limit: int = Query(20, le=100),
        ext: list[str] = Query(default=[]),
        path: str | None = None,
        since: str | None = None,
        mode: str = "hybrid",
    ):
        filters = SearchFilters(ext, path, parse_since(since))
        results = engine.search(q, limit, filters, mode)
        return {"query": q, "results": [r.to_dict() for r in results]}

    @app.get("/api/ask")
    def api_ask(q: str, k: int = 6):
        try:
            answer, results = engine.ask(q, k)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"answer": answer, "sources": [r.to_dict() for r in results]}

    @app.post("/api/open")
    def api_open(req: OpenRequest):
        # Only allow opening files that are actually in the index.
        if not engine.is_indexed(req.path):
            raise HTTPException(404, "File is not in the index")
        engine.open_path(req.path, reveal=req.reveal)
        return {"ok": True}

    @app.get("/api/status")
    def api_status():
        s = engine.store.stats()
        return {
            **s,
            "indexing": state["indexing"],
            "last_report": state["last_report"],
            "roots": engine.cfg.roots,
        }

    @app.post("/api/reindex")
    def api_reindex():
        if state["indexing"]:
            return {"ok": False, "message": "Already indexing"}

        def run():
            state["indexing"] = True
            try:
                state["last_report"] = engine.indexer.run().__dict__
            finally:
                state["indexing"] = False

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    return app
