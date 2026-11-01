"""Command-line interface:  smartsearch --help"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import DEFAULT_HOME, Config, load_config
from .engine import Engine
from .search import SearchFilters, human_size, parse_since, short_path

app = typer.Typer(help="SmartSearch — private AI search over your files.", no_args_is_help=True)
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config.yaml")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    for noisy in ("httpx", "sentence_transformers", "urllib3", "watchdog", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


def _engine(config: Optional[Path]) -> Engine:
    return Engine(load_config(config))


def _filters(ext: list[str], path: Optional[str], since: Optional[str]) -> SearchFilters:
    return SearchFilters(
        extensions=ext or [], path_contains=path, modified_after=parse_since(since)
    )


@app.command()
def init(
    folders: list[Path] = typer.Argument(None, help="Folders to index (default: ~/Documents)"),
    output: Path = typer.Option(DEFAULT_HOME / "config.yaml", "--output", "-o"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
) -> None:
    """Create a config file listing the folders to index."""
    if output.exists() and not force:
        console.print(f"[yellow]Config already exists:[/] {output}  (use --force to overwrite)")
        raise typer.Exit(1)
    cfg = Config()
    if folders:
        cfg.roots = [str(f.expanduser().resolve()) for f in folders]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cfg.to_yaml(), encoding="utf-8")
    console.print(f"[green]✓[/] Wrote {output}\n  Indexing: " + ", ".join(cfg.roots))
    console.print("Next: [bold]smartsearch index[/]")


@app.command()
def index(
    config: Optional[Path] = ConfigOpt,
    rebuild: bool = typer.Option(False, "--rebuild", help="Re-index every file from scratch"),
) -> None:
    """Scan folders and update the index (only changed files are re-processed)."""
    eng = _engine(config)
    console.print(f"Embedder: [cyan]{eng.embedder.name}[/]   Roots: " + ", ".join(eng.cfg.roots))
    if rebuild:
        eng.store.clear()
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("Scanning…")
        rep = eng.indexer.run(
            progress=lambda p, i: prog.update(
                task, description=f"[{i + 1}] {short_path(str(p))[-80:]}"
            )
        )
    console.print(
        f"[green]✓ Done in {time.time() - start:.1f}s[/] — scanned {rep.scanned}, "
        f"added {rep.added}, updated {rep.updated}, unchanged {rep.unchanged}, "
        f"removed {rep.removed}, failed {len(rep.failed)}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="What you're looking for, in plain language"),
    limit: int = typer.Option(10, "--limit", "-n"),
    ext: list[str] = typer.Option(
        [], "--ext", "-e", help="Filter by extension, e.g. -e pdf -e docx"
    ),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Path must contain this text"),
    since: Optional[str] = typer.Option(None, "--since", "-s", help="e.g. 7d, 2w, 2026-01-01"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid | semantic | keyword"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    open_first: bool = typer.Option(False, "--open", "-o", help="Open the top result"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Search your files."""
    eng = _engine(config)
    t0 = time.time()
    results = eng.search(query, limit, _filters(ext, path, since), mode)
    if as_json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        console.print("[yellow]No results.[/] Have you run [bold]smartsearch index[/]?")
        return
    table = Table(
        show_lines=True, title=f"{len(results)} results in {(time.time() - t0) * 1000:.0f} ms"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", overflow="fold", max_width=50)
    table.add_column("Match", overflow="fold")
    table.add_column("Modified", style="dim", no_wrap=True)
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"[bold]{escape(r.name)}[/]\n[dim]{escape(short_path(str(Path(r.path).parent)))}[/]\n"
            f"[magenta]{', '.join(r.matched_by)}[/]",
            escape(r.snippet),
            datetime.fromtimestamp(r.modified).strftime("%Y-%m-%d") + f"\n{human_size(r.size)}",
        )
    console.print(table)
    if open_first:
        eng.open_path(results[0].path)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question about the contents of your files"),
    k: int = typer.Option(6, "--k", help="How many files to give the LLM as context"),
    ext: list[str] = typer.Option([], "--ext", "-e"),
    path: Optional[str] = typer.Option(None, "--path", "-p"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Answer a question from your files with a local LLM (requires Ollama)."""
    eng = _engine(config)
    with console.status(f"Thinking with {eng.cfg.ollama_model}…"):
        try:
            answer, results = eng.ask(question, k, _filters(ext, path, None))
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from None
    console.print(Panel(escape(answer), title="Answer", border_style="green"))
    for i, r in enumerate(results, 1):
        console.print(f"  [dim][{i}][/] {short_path(r.path)}")


@app.command()
def watch(config: Optional[Path] = ConfigOpt) -> None:
    """Index once, then keep the index updated as files change (Ctrl+C to stop)."""
    from .watcher import start_watching

    eng = _engine(config)
    rep = eng.indexer.run()
    console.print(f"[green]✓ Index ready[/] ({rep.scanned} files). Watching for changes…")
    observer, handler = start_watching(eng.indexer)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.stop()
        observer.stop()
        observer.join()


@app.command()
def serve(
    config: Optional[Path] = ConfigOpt,
    host: Optional[str] = typer.Option(None, help="Default 127.0.0.1 (local only)"),
    port: Optional[int] = typer.Option(None),
    no_watch: bool = typer.Option(False, "--no-watch", help="Don't auto-update the index"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Start the local web UI."""
    import threading
    import webbrowser

    import uvicorn

    from .web.app import create_app

    eng = _engine(config)
    host = host or eng.cfg.host
    port = port or eng.cfg.port
    web_app = create_app(eng, watch=not no_watch)
    url = f"http://{host}:{port}"
    console.print(f"[green]SmartSearch UI →[/] [link={url}]{url}[/link]")
    if not no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@app.command()
def stats(config: Optional[Path] = ConfigOpt) -> None:
    """Show what's in the index."""
    eng = _engine(config)
    s = eng.store.stats()
    console.print(
        f"Files: [bold]{s['files']}[/]   Chunks: {s['chunks']}   "
        f"Embedder: {s['embedder']}   DB: {s['db_path']} ({s['db_size_mb']} MB)"
    )
    t = Table("Extension", "Files")
    for ext_, n in s["by_extension"].items():
        t.add_row(ext_, str(n))
    console.print(t)


@app.command()
def doctor(config: Optional[Path] = ConfigOpt) -> None:
    """Check configuration and optional components."""
    from . import llm

    cfg = load_config(config)
    ok = lambda b: "[green]✓[/]" if b else "[red]✗[/]"  # noqa: E731
    for r in cfg.root_paths:
        console.print(f"{ok(r.exists())} root {r}")
    try:
        import sentence_transformers  # noqa: F401

        st = True
    except ImportError:
        st = False
    console.print(
        f"{ok(st)} sentence-transformers installed "
        + ("" if st else "[dim](pip install -e '.\\[ai]' for semantic search)[/]")
    )
    ol = llm.ollama_available(cfg.ollama_url)
    console.print(
        f"{ok(ol)} Ollama at {cfg.ollama_url} "
        + ("" if ol else "[dim](optional, needed for `ask`)[/]")
    )
    console.print(f"  index: {cfg.db_path}")


if __name__ == "__main__":
    app()
