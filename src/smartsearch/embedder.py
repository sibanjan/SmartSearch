"""Embedding backends.

- SentenceTransformerEmbedder: neural semantic embeddings, runs fully on your
  laptop (CPU or GPU). Model is downloaded once from Hugging Face, then cached.
- HashEmbedder: dependency-free fallback using feature hashing of words and
  word-bigrams. Lexical rather than semantic, but keeps the app usable offline
  and makes tests fast.

All embedders return L2-normalised float32 vectors, so dot product == cosine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hash-{dim}"

    _token = re.compile(r"[a-z0-9]+")

    def _features(self, text: str) -> list[str]:
        words = self._token.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        # sublinear term frequency
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        # Prefer the local Hugging Face cache so later runs don't hit the network;
        # only download when the model isn't cached yet.
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
            log.info("Loaded embedding model %s from local cache", model_name)
        except Exception:
            log.info("Downloading embedding model %s (first run only)...", model_name)
            self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.name = model_name
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        # Some models (e5, bge) expect instruction prefixes
        lname = model_name.lower()
        self._q_prefix = "query: " if "e5" in lname else ""
        self._d_prefix = "passage: " if "e5" in lname else ""

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        prefix = self._q_prefix if query else self._d_prefix
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(kind: str, model_name: str, batch_size: int = 32) -> Embedder:
    kind = kind.lower()
    if kind == "hash":
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name, batch_size)
    except Exception as exc:
        if kind == "sentence-transformers":
            raise
        log.warning(
            "Semantic model unavailable (%s). Falling back to lexical hash embedder. "
            "Install with: pip install -e '.[ai]'",
            exc,
        )
        return HashEmbedder()
