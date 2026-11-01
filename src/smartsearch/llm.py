"""Optional "ask your files" answers using a local LLM through Ollama.

Nothing leaves your laptop: Ollama runs the model locally.
Install from https://ollama.com, then e.g. `ollama pull llama3.2`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .search import SearchResult

PROMPT = """You are a helpful assistant answering questions using ONLY the excerpts
from the user's own files below. Cite sources by their [number]. If the excerpts
don't contain the answer, say so plainly.

{context}

Question: {question}
Answer:"""


def build_context(results: list[SearchResult], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r.path}\n{r.chunk_text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def ollama_available(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def ask(
    question: str, results: list[SearchResult], url: str, model: str, timeout: int = 180
) -> str:
    if not results:
        return "No relevant files found in the index."
    payload = {
        "model": model,
        "prompt": PROMPT.format(context=build_context(results), question=question),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {url} ({exc}). Is it running? "
            f"Try: ollama serve  and  ollama pull {model}"
        ) from exc
