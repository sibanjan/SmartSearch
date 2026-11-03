"""Find images and link them to categories by content. Files are never moved.

How it works:
  1. scan()      embeds every image with a local CLIP model (stored in the same SQLite DB).
  2. classify()  scores each image against the text prompts in `image_categories`
                 (zero-shot) and links it to every category it matches well enough, in
                 the `image_labels` table. Images that fit none of them are clustered by
                 visual similarity and each cluster gets a proposed name ("other-<concept>").
                 Re-run it freely after editing categories: no re-embedding needed.

Browse the links in SQL through the `image_category_view` view, e.g.
  SELECT path, score FROM image_category_view WHERE category='receipts' ORDER BY score DESC;
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from .config import Config
from .store import Store

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER,
    mtime REAL,
    width INTEGER,
    height INTEGER,
    vec BLOB NOT NULL,
    indexed_at REAL
);
-- image <-> category links; an image can be in several categories
CREATE TABLE IF NOT EXISTS image_labels (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    score REAL NOT NULL,           -- match probability (1.0 for cluster membership)
    is_primary INTEGER NOT NULL,   -- 1 for the image's best category
    source TEXT NOT NULL,          -- 'prompt' (configured category) or 'cluster'
    PRIMARY KEY (image_id, category)
);
CREATE INDEX IF NOT EXISTS idx_image_labels_category ON image_labels(category, score);
CREATE VIEW IF NOT EXISTS image_category_view AS
    SELECT l.category, l.score, l.is_primary, l.source, i.path, i.width, i.height,
           i.size, i.mtime
    FROM image_labels l JOIN images i ON i.id = l.image_id;
"""

UNCATEGORIZED = "uncategorized"

# Concepts used to name clusters of images that matched no configured category.
CLUSTER_VOCAB = [
    "text", "handwriting", "a whiteboard", "math equations", "source code", "a table of data",
    "a user interface", "a website", "a map", "a certificate", "an id card", "a book cover",
    "faces", "children", "a family", "a crowd", "sports", "music", "a classroom", "an office",
    "a room interior", "furniture", "a kitchen", "flowers", "plants", "trees", "water",
    "the sky", "a sunset", "snow", "mountains", "a beach", "a road", "a building", "a temple",
    "night", "a cartoon", "anime", "a pattern", "abstract art", "a texture", "clothes",
    "shoes", "jewelry", "toys", "a phone", "a computer", "electronics", "tools", "a medical scan",
    "an x-ray", "a video game", "a poster", "a flyer", "a book page", "a photo collage",
    "a black background", "a white background", "a blurry photo", "a product",
]


class ImageEncoder(Protocol):
    name: str

    def encode_images(self, images: list) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class ClipEncoder:
    """CLIP through sentence-transformers: images and text share one vector space."""

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            log.info("Downloading image model %s (first run only, ~600 MB)...", model_name)
            self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.batch_size = batch_size

    def _encode(self, items: list) -> np.ndarray:
        vecs = self.model.encode(
            items,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_images(self, images: list) -> np.ndarray:
        return self._encode(images)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


@dataclass
class ScanReport:
    scanned: int = 0
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[str] = field(default_factory=list)


def _load_image(path: Path, max_side: int = 448):
    from PIL import Image

    with Image.open(path) as img:
        img.draft("RGB", (max_side, max_side))  # fast JPEG downscale on decode
        w, h = img.size
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        return img, w, h


def _kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Small k-means++ on L2-normalised vectors (cosine). Returns a label per row."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = 1 - np.max(x @ np.array(centers).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        if d.sum() == 0:
            break
        centers.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(centers)
    labels = np.zeros(len(x), dtype=int)
    for it in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if it > 0 and np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(c)):
            members = x[labels == j]
            if len(members):
                m = members.mean(axis=0)
                c[j] = m / (np.linalg.norm(m) or 1.0)
    return labels


class ImageCatalog:
    def __init__(
        self, cfg: Config, store: Store, encoder_factory: Callable[[], ImageEncoder] | None = None
    ):
        self.cfg = cfg
        self.store = store
        self.conn = store.conn
        self._encoder_factory = encoder_factory or (lambda: ClipEncoder(cfg.image_model))
        self._encoder: ImageEncoder | None = None
        with store.lock:
            self.conn.executescript(SCHEMA)

    @property
    def encoder(self) -> ImageEncoder:
        if self._encoder is None:
            self._encoder = self._encoder_factory()
            stored = self.store.get_meta("image_model")
            if stored and stored != self._encoder.name:
                log.warning("Image model changed (%s -> %s); clearing image index.",
                            stored, self._encoder.name)
                with self.store.lock, self.conn:
                    self.conn.execute("DELETE FROM images")
            self.store.set_meta("image_model", self._encoder.name)
        return self._encoder

    # ---- scanning --------------------------------------------------------
    @property
    def roots(self) -> list[Path]:
        return [Path(r).expanduser().resolve() for r in (self.cfg.image_roots or self.cfg.roots)]

    def iter_images(self, roots: list[Path] | None = None) -> Iterator[Path]:
        exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in self.cfg.image_extensions}
        excluded = set(self.cfg.exclude_dirs)
        min_bytes = self.cfg.image_min_size_kb * 1024
        for root in roots or self.roots:
            if not root.is_dir():
                log.warning("Image root does not exist: %s", root)
                continue
            for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
                dirnames[:] = [
                    d for d in dirnames
                    if d not in excluded and not (self.cfg.skip_hidden and d.startswith("."))
                ]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if p.suffix.lower() not in exts or (self.cfg.skip_hidden and fn.startswith(".")):
                        continue
                    try:
                        if p.stat().st_size >= min_bytes:
                            yield p
                    except OSError:
                        continue

    def scan(
        self,
        roots: list[Path] | None = None,
        progress: Callable[[Path, int], None] | None = None,
        batch: int = 32,
    ) -> ScanReport:
        """Embed new/changed images, drop vanished ones, then re-classify everything."""
        report = ScanReport()
        known = {
            r["path"]: (r["size"], r["mtime"])
            for r in self.conn.execute("SELECT path, size, mtime FROM images")
        }
        seen: set[str] = set()
        pending: list[tuple[Path, os.stat_result]] = []
        for i, p in enumerate(self.iter_images(roots)):
            report.scanned += 1
            key = str(p)
            seen.add(key)
            if progress:
                progress(p, i)
            st = p.stat()
            if known.get(key) == (st.st_size, st.st_mtime):
                report.unchanged += 1
                continue
            pending.append((p, st))
            if len(pending) >= batch:
                self._embed_batch(pending, report)
                pending = []
        if pending:
            self._embed_batch(pending, report)

        scan_roots = roots or self.roots
        with self.store.lock, self.conn:
            for path in known:
                # Only forget images under the roots we just scanned.
                if path not in seen and any(_under(Path(path), r) for r in scan_roots):
                    self.conn.execute("DELETE FROM images WHERE path=?", (path,))
                    report.removed += 1
        self.classify()
        return report

    def _embed_batch(self, items: list[tuple[Path, os.stat_result]], report: ScanReport) -> None:
        loaded = []
        for p, st in items:
            try:
                img, w, h = _load_image(p)
                loaded.append((p, st, img, w, h))
            except Exception as exc:
                log.debug("Could not read image %s: %s", p, exc)
                report.failed.append(str(p))
        if not loaded:
            return
        vecs = self.encoder.encode_images([x[2] for x in loaded])
        now = time.time()
        with self.store.lock, self.conn:
            for (p, st, _img, w, h), v in zip(loaded, vecs):
                self.conn.execute(
                    "INSERT INTO images(path,size,mtime,width,height,vec,indexed_at) "
                    "VALUES(?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET size=excluded.size, "
                    "mtime=excluded.mtime, width=excluded.width, height=excluded.height, "
                    "vec=excluded.vec, indexed_at=excluded.indexed_at",
                    (str(p), st.st_size, st.st_mtime, w, h, v.astype(np.float32).tobytes(), now),
                )
                report.added += 1

    # ---- classification --------------------------------------------------
    def classify(self) -> dict[str, int]:
        """(Re)link every image to its categories from stored embeddings."""
        rows = self.conn.execute("SELECT id, vec FROM images").fetchall()
        with self.store.lock, self.conn:
            self.conn.execute("DELETE FROM image_labels")
        if not rows:
            return {}
        ids = [r["id"] for r in rows]
        x = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])

        names = list(self.cfg.image_categories)
        links: list[tuple[int, str, float, int, str]] = []
        primary: list[str | None] = [None] * len(ids)
        if names:
            t = self.encoder.encode_texts([self.cfg.image_categories[n] for n in names])
            logits = 100.0 * (x @ t.T)  # CLIP's logit scale
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            for i, row in enumerate(probs):
                top = int(np.argmax(row))
                if row[top] < self.cfg.image_min_confidence:
                    continue  # no confident match: left for clustering, no weak links
                primary[i] = names[top]
                for j in np.flatnonzero(row >= self.cfg.image_label_threshold):
                    is_primary = int(primary[i] == names[j])
                    links.append((ids[i], names[j], round(float(row[j]), 3), is_primary, "prompt"))

        for i, name in self._cluster_leftovers(x, primary).items():
            links.append((ids[i], name, 1.0, 1, "cluster"))

        with self.store.lock, self.conn:
            self.conn.executemany(
                "INSERT INTO image_labels(image_id,category,score,is_primary,source) "
                "VALUES(?,?,?,?,?)",
                links,
            )
        return self.summary()

    def _cluster_leftovers(self, x: np.ndarray, primary: list[str | None]) -> dict[int, str]:
        """Group images with no primary category; returns {row index: cluster name}."""
        idx = [i for i, c in enumerate(primary) if c is None]
        min_size = self.cfg.image_min_cluster_size
        if len(idx) < 2 * min_size:
            return {}
        k = self.cfg.image_clusters or int(np.clip(round(np.sqrt(len(idx) / 4)), 1, 12))
        k = min(k, len(idx) // min_size)
        assign = _kmeans(x[idx], k)
        vocab = self.encoder.encode_texts([f"a photo of {w}" for w in CLUSTER_VOCAB])
        used: dict[str, int] = {}
        out: dict[int, str] = {}
        for j in range(k):
            members = [idx[n] for n in np.flatnonzero(assign == j)]
            if len(members) < min_size:
                continue  # too small to be a meaningful group; stays uncategorized
            centroid = x[members].mean(axis=0)
            word = CLUSTER_VOCAB[int(np.argmax(vocab @ centroid))]
            base = "other-" + re.sub(r"^(a|an|the) ", "", word).replace(" ", "-")
            used[base] = used.get(base, 0) + 1
            name = base if used[base] == 1 else f"{base}-{used[base]}"
            out.update({m: name for m in members})
        return out

    # ---- queries ---------------------------------------------------------
    def summary(self) -> dict[str, int]:
        """Images per category (an image counts in every category it is linked to)."""
        rows = self.conn.execute(
            "SELECT category, COUNT(*) n FROM image_labels GROUP BY category ORDER BY n DESC"
        ).fetchall()
        counts = {r["category"]: r["n"] for r in rows}
        unlinked = self.conn.execute(
            "SELECT COUNT(*) FROM images WHERE id NOT IN (SELECT image_id FROM image_labels)"
        ).fetchone()[0]
        if unlinked:
            counts[UNCATEGORIZED] = unlinked
        return counts

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def list(
        self,
        category: str | None = None,
        limit: int = 200,
        offset: int = 0,
        primary_only: bool = False,
    ) -> list[dict]:
        """Images (at their original paths) in a category, best matches first."""
        if category == UNCATEGORIZED:
            sql = (
                "SELECT i.* FROM images i WHERE i.id NOT IN (SELECT image_id FROM image_labels) "
                "ORDER BY i.mtime DESC LIMIT ? OFFSET ?"
            )
            params: list = [limit, offset]
        elif category:
            sql = (
                "SELECT i.* FROM image_labels l JOIN images i ON i.id = l.image_id "
                "WHERE l.category = ?"
                + (" AND l.is_primary = 1" if primary_only else "")
                + " ORDER BY l.score DESC, i.mtime DESC LIMIT ? OFFSET ?"
            )
            params = [category, limit, offset]
        else:
            sql = "SELECT i.* FROM images i ORDER BY i.mtime DESC LIMIT ? OFFSET ?"
            params = [limit, offset]
        rows = self.conn.execute(sql, params).fetchall()
        labels = self._labels_for([r["id"] for r in rows])
        out = []
        for r in rows:
            lbls = labels.get(r["id"], [])
            out.append(
                {
                    "path": r["path"],
                    "name": Path(r["path"]).name,
                    "folder": str(Path(r["path"]).parent),
                    "size": r["size"],
                    "modified": r["mtime"],
                    "width": r["width"],
                    "height": r["height"],
                    "category": next(
                        (lb["category"] for lb in lbls if lb["primary"]), UNCATEGORIZED
                    ),
                    "labels": lbls,
                }
            )
        return out

    def _labels_for(self, image_ids: list[int]) -> dict[int, list[dict]]:
        if not image_ids:
            return {}
        q = ",".join("?" * len(image_ids))
        out: dict[int, list[dict]] = {}
        for r in self.conn.execute(
            f"SELECT image_id, category, score, is_primary, source FROM image_labels "
            f"WHERE image_id IN ({q}) ORDER BY is_primary DESC, score DESC",
            image_ids,
        ):
            out.setdefault(r["image_id"], []).append(
                {
                    "category": r["category"],
                    "score": r["score"],
                    "primary": bool(r["is_primary"]),
                    "source": r["source"],
                }
            )
        return out

    def is_indexed(self, path: str) -> bool:
        return self.conn.execute("SELECT 1 FROM images WHERE path=?", (path,)).fetchone() is not None


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
