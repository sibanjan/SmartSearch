# SmartSearch — private AI search for your laptop

Search the files on your own computer in plain language ("the contract with the
landscaping company", "python script that trains a gradient boosting model").
Everything — indexing, AI embeddings, the optional LLM — runs **locally**. Nothing
is uploaded anywhere.

**How it finds things.** Every query runs three searches and fuses them with
Reciprocal Rank Fusion:

| Signal | What it catches | Tech |
|---|---|---|
| Semantic | Meaning, synonyms, paraphrases | sentence-transformers embeddings + cosine similarity |
| Keyword | Exact words, codes, names | SQLite FTS5 (BM25, stemming, prefix match) |
| File name | "that budget spreadsheet" | name substring match |

**Supports:** PDF, Word, PowerPoint, Excel, CSV, HTML, Markdown, text, Jupyter
notebooks, JSON/YAML and most source-code files. Incremental re-indexing (only
changed files), live folder watching, filters (type, path, date), a CLI and a
local web UI, plus optional "Ask AI" answers with citations via Ollama.

---

## 1. Install (Windows, macOS, Linux — Python 3.10+)

```bash
cd smart-search
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e ".[ai,dev]"
```

`[ai]` installs sentence-transformers (PyTorch). The embedding model
(~90 MB) downloads once on first run and is cached. Skip `[ai]` if you want a
tiny install — the app falls back to a lexical embedder and still works.

## 2. Choose folders and index

```bash
smartsearch init ~/Documents ~/Desktop     # writes ~/.smartsearch/config.yaml
smartsearch index                           # first run takes a while; later runs are incremental
smartsearch doctor                          # checks folders, AI model, Ollama
```

Edit `~/.smartsearch/config.yaml` (see `config.example.yaml` for every option),
or put a `config.yaml` in the project folder while developing.

## 3. Search

```bash
smartsearch search "quarterly sales deck for EMEA"
smartsearch search "Insurance Document" -e pdf --since 30d
smartsearch search "oauth token refresh" -e py -p projects --open
smartsearch search "tax" --json            # for scripting
```

Web UI (opens in your browser, auto-indexes and watches for changes):

```bash
smartsearch serve
```

Click a result to open it in its default app, or "show in folder".

Keep the index fresh without the UI: `smartsearch watch`.

## 4. Ask your files (optional local LLM)

1. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`
2. `smartsearch ask "when does my car insurance renew?"`

The top matching excerpts are sent to the local model, which answers with
`[n]` citations to your files. Use the **Ask AI** button in the web UI too.

---

## Working in VS Code

Open the `smart-search` folder (**File → Open Folder**), accept the recommended
extensions, and select the `.venv` interpreter (**Python: Select Interpreter**).

- **Run & Debug** panel: *Web UI*, *Index*, *Search (prompt)*, *Ask AI*, *Pytest*
- **Testing** panel: pytest is pre-configured (`tests/`)
- **Terminal → Run Task**: tests, lint, index, start UI
- Ruff formats on save

## Project layout

```
src/smartsearch/
  config.py      settings + config file loading
  extractors.py  file type -> text (add new formats here)
  chunker.py     split text into overlapping chunks
  embedder.py    sentence-transformers / hash fallback embeddings
  store.py       SQLite: files, chunks, FTS5 index, vectors
  indexer.py     folder walk, incremental indexing, deletions
  search.py      hybrid search + RRF fusion + filters
  watcher.py     live re-indexing with watchdog
  llm.py         Ollama Q&A
  engine.py      wires everything together (used by CLI + web)
  cli.py         Typer CLI
  web/app.py     FastAPI JSON API + static UI
  web/static/    index.html (single-file UI)
tests/           pytest suite (runs offline with the hash embedder)
```

## Ideas for next enhancements

- **OCR** for scanned PDFs/images: `pytesseract` in `extractors.py`
- **Image search** by content with a CLIP model (`sentence-transformers/clip-ViT-B-32`)
- **Faster vectors at scale**: swap the numpy matrix in `store.py` for FAISS, `sqlite-vec` or LanceDB
- **Re-ranking** the top 50 with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Run at login**: Task Scheduler (Windows), launchd (macOS), systemd user service (Linux) running `smartsearch watch`
- **Global hotkey / tray app** with `pystray` + a small search popup
- Email (`.eml`, `.mbox`) and e-book (`.epub`) extractors
- Duplicate / near-duplicate detection using the stored embeddings

## Notes

- The index lives in `~/.smartsearch/index.db` (override with `index_dir` or `SMARTSEARCH_HOME`).
- Changing `model_name` or `embedder` automatically clears the index so it can be rebuilt with `smartsearch index`.
- The web server binds to `127.0.0.1` only, and `/api/open` refuses paths that aren't in the index.
- Vector search is brute-force numpy — fine for ~100k–300k chunks on a typical laptop.
