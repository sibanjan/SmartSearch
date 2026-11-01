"""Split long text into overlapping chunks, preferring paragraph/line boundaries."""

from __future__ import annotations

import re


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # Break into units: paragraphs, then hard-split anything still too long.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > size:
            cut = para.rfind("\n", 0, size)
            if cut < size // 2:
                cut = para.rfind(" ", 0, size)
            if cut < size // 2:
                cut = size
            units.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            units.append(para)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = (tail + "\n\n" + unit).strip() if tail else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks
