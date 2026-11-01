"""Configuration loading.

Lookup order for the config file:
  1. Path in the SMARTSEARCH_CONFIG environment variable
  2. ./config.yaml (current working directory)
  3. ~/.smartsearch/config.yaml
If none exists, defaults are used.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_HOME = Path(os.environ.get("SMARTSEARCH_HOME", Path.home() / ".smartsearch"))

DEFAULT_EXTENSIONS = [
    # documents
    ".txt",
    ".md",
    ".rst",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".log",
    ".tex",
    ".ipynb",
    # code
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".ps1",
    ".sql",
    ".r",
    ".scala",
    ".kt",
    ".swift",
]

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "site-packages",
    ".cache",
    "Library",
    "AppData",
    "$RECYCLE.BIN",
    ".Trash",
    ".smartsearch",
]


@dataclass
class Config:
    roots: list[str] = field(default_factory=lambda: [str(Path.home() / "Documents")])
    extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    skip_hidden: bool = True
    max_file_size_mb: float = 25.0

    # chunking
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150

    # embeddings: "sentence-transformers" | "hash" | "auto"
    embedder: str = "auto"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    # storage
    index_dir: str = str(DEFAULT_HOME)

    # search
    default_limit: int = 10
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    filename_weight: float = 0.6

    # local LLM (optional, for `ask`) via Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # web UI
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def db_path(self) -> Path:
        return Path(self.index_dir).expanduser() / "index.db"

    @property
    def root_paths(self) -> list[Path]:
        return [Path(r).expanduser().resolve() for r in self.roots]

    @property
    def extension_set(self) -> set[str]:
        return {e.lower() if e.startswith(".") else "." + e.lower() for e in self.extensions}

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)


def config_path_candidates() -> list[Path]:
    paths = []
    if env := os.environ.get("SMARTSEARCH_CONFIG"):
        paths.append(Path(env).expanduser())
    paths.append(Path.cwd() / "config.yaml")
    paths.append(DEFAULT_HOME / "config.yaml")
    return paths


def load_config(path: str | Path | None = None) -> Config:
    candidates = [Path(path).expanduser()] if path else config_path_candidates()
    for p in candidates:
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
            return Config(**known)
    return Config()
