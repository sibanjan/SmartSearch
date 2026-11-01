"""Text extraction for different file types.

Add a new format by writing a function `(Path) -> str` and registering it
in EXTRACTORS. Anything not registered is read as plain text.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:  # looks binary
        return ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"[page {i + 1}]\n{page.extract_text() or ''}")
        except Exception as exc:  # malformed page
            log.debug("pdf page %s failed in %s: %s", i, path, exc)
    return "\n\n".join(pages)


def _docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides, 1):
        texts = [
            shape.text_frame.text
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip()
        ]
        if texts:
            out.append(f"[slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(out)


def _xlsx(path: Path, max_rows: int = 2000) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"[sheet {ws.title}]")
        for r, row in enumerate(ws.iter_rows(values_only=True)):
            if r >= max_rows:
                break
            vals = [str(v) for v in row if v is not None]
            if vals:
                out.append(" | ".join(vals))
    wb.close()
    return "\n".join(out)


def _csv(path: Path, max_rows: int = 5000) -> str:
    text = _read_text(path)
    rows = []
    for i, row in enumerate(csv.reader(text.splitlines())):
        if i >= max_rows:
            break
        rows.append(" | ".join(row))
    return "\n".join(rows)


def _html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def _ipynb(path: Path) -> str:
    nb = json.loads(_read_text(path))
    cells = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        cells.append("".join(src) if isinstance(src, list) else src)
    return "\n\n".join(cells)


EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".pdf": _pdf,
    ".docx": _docx,
    ".pptx": _pptx,
    ".xlsx": _xlsx,
    ".csv": _csv,
    ".html": _html,
    ".htm": _html,
    ".ipynb": _ipynb,
}


def extract_text(path: Path) -> str:
    """Return the text content of a file, or '' if it can't be read."""
    fn = EXTRACTORS.get(path.suffix.lower(), _read_text)
    try:
        return fn(path) or ""
    except Exception as exc:
        log.warning("Could not extract %s: %s", path, exc)
        return ""
