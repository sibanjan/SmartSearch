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

DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"]

# category name (also the folder name when organising) -> text prompt CLIP matches against
DEFAULT_IMAGE_CATEGORIES = {
    "people": "a photo of a person or a group of people",
    "pets": "a photo of a pet cat or dog",
    "animals": "a photo of a wild animal or bird",
    "nature": "a landscape photo of nature, mountains, a beach, the sky or a forest",
    "city": "a photo of buildings, streets or a city",
    "food": "a photo of food or a meal",
    "vehicles": "a photo of a car, motorbike, train or airplane",
    "events": "a photo of a celebration, party, wedding or festival",
    "documents": "a scanned document, form or page of printed text",
    "receipts": "a photo of a receipt, invoice or bill",
    "screenshots": "a screenshot of a computer or phone screen",
    "charts": "a chart, graph, plot or data visualization",
    "diagrams": "a diagram, flowchart or architecture drawing",
    "slides": "a presentation slide",
    "logos-icons": "a logo, icon or simple graphic",
    "art": "a painting, drawing or illustration",
}


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

    # embeddings: "auto" (ONNX, then sentence-transformers, then hash) | "onnx" | "sentence-transformers" | "hash"
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

    # images: `smartsearch images ...` (needs the [ai] extra; CLIP runs locally)
    image_roots: list[str] = field(default_factory=list)  # empty = same as roots
    image_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))
    image_min_size_kb: float = 20.0  # skip tiny icons/thumbnails
    image_model: str = "clip-ViT-B-32"
    image_categories: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_IMAGE_CATEGORIES))
    image_min_confidence: float = 0.35  # top category probability needed to assign it
    image_label_threshold: float = 0.25  # also link an image to any category scoring this high
    image_clusters: int = 0  # clusters for leftover images; 0 = automatic
    image_min_cluster_size: int = 3

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
