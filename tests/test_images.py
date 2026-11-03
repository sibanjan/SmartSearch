import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from smartsearch.config import Config
from smartsearch.engine import Engine
from smartsearch.images import ImageCatalog
from smartsearch.web.app import create_app

COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220), "green": (20, 200, 20)}


class FakeEncoder:
    """Images -> their mean colour; texts naming a colour -> that colour, else a random vector."""

    name = "fake-clip"

    def encode_images(self, images):
        return _norm(np.array([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images]))

    def encode_texts(self, texts):
        out = []
        for t in texts:
            hit = next((rgb for c, rgb in COLORS.items() if c in t), None)
            if hit is None:
                seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
                hit = np.random.default_rng(seed).normal(size=3)
            out.append(hit)
        return _norm(np.array(out, dtype=np.float32))


def _norm(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def pics(tmp_path):
    root = tmp_path / "pics"
    (root / "trip").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for color, n in (("red", 3), ("blue", 2), ("green", 6)):
        for i in range(n):
            arr = np.clip(np.array(COLORS[color]) + rng.integers(-15, 15, (64, 64, 3)), 0, 255)
            Image.fromarray(arr.astype(np.uint8)).save(root / "trip" / f"{color}_{i}.png")
    Image.new("RGB", (8, 8), "red").save(root / "tiny.png")  # below the size limit
    return root


@pytest.fixture
def catalog(tmp_path, pics):
    cfg = Config(
        roots=[str(pics)],
        index_dir=str(tmp_path / "index"),
        image_min_size_kb=1,
        image_categories={"reds": "something red", "blues": "something blue"},
        image_min_confidence=0.6,
    )
    eng = Engine(cfg)
    eng.images = ImageCatalog(cfg, eng.store, FakeEncoder)
    eng.images.scan()
    return eng


def test_scan_categorises_and_clusters_leftovers(catalog):
    counts = catalog.images.summary()
    assert counts["reds"] == 3 and counts["blues"] == 2
    # green matched no category, so it became an auto-named cluster
    others = {k: v for k, v in counts.items() if k.startswith("other-")}
    assert sum(others.values()) == 6
    assert not any("tiny" in im["path"] for im in catalog.images.list())


def test_rescan_is_incremental(catalog):
    rep = catalog.images.scan()
    assert rep.added == 0 and rep.unchanged == 11


def test_links_point_to_original_files_and_nothing_moves(catalog, pics):
    before = sorted(p.name for p in pics.rglob("*"))
    reds = catalog.images.list("reds")
    assert len(reds) == 3
    assert all(im["folder"] == str(pics / "trip") for im in reds)
    assert all(im["category"] == "reds" and im["labels"][0]["primary"] for im in reds)
    catalog.images.classify()
    assert sorted(p.name for p in pics.rglob("*")) == before  # no files moved or copied


def test_image_can_be_in_several_categories(tmp_path, pics):
    # a purple leaning red: confidently red (64%), but blue enough (36%) for a second link
    Image.new("RGB", (64, 64), (220, 20, 218)).save(pics / "trip" / "purple.png")
    cfg = Config(
        roots=[str(pics)],
        index_dir=str(tmp_path / "index2"),
        image_min_size_kb=0,
        image_categories={"reds": "something red", "blues": "something blue"},
        image_min_confidence=0.6,
    )
    cat = ImageCatalog(cfg, Engine(cfg).store, FakeEncoder)
    cat.scan()
    in_red = {im["name"] for im in cat.list("reds")}
    in_blue = {im["name"] for im in cat.list("blues")}
    assert "purple.png" in in_red and "purple.png" in in_blue
    assert "purple.png" in {im["name"] for im in cat.list("reds", primary_only=True)}
    assert "purple.png" not in {im["name"] for im in cat.list("blues", primary_only=True)}


def test_category_view_in_database(catalog):
    rows = catalog.store.conn.execute(
        "SELECT path, score FROM image_category_view WHERE category='blues' ORDER BY score DESC"
    ).fetchall()
    assert len(rows) == 2 and all(r["path"].endswith(".png") for r in rows)


def test_removed_image_drops_its_links(catalog, pics):
    (pics / "trip" / "red_0.png").unlink()
    catalog.images.scan()
    assert catalog.images.summary()["reds"] == 2
    orphans = catalog.store.conn.execute(
        "SELECT COUNT(*) FROM image_labels WHERE image_id NOT IN (SELECT id FROM images)"
    ).fetchone()[0]
    assert orphans == 0


def test_image_api(catalog):
    client = TestClient(create_app(catalog, watch=False, index_on_start=False))
    cats = client.get("/api/images/categories").json()
    assert cats["categories"]["reds"] == 3 and cats["total"] == 11
    imgs = client.get("/api/images", params={"category": "blues"}).json()["images"]
    assert len(imgs) == 2
    r = client.get("/api/images/thumb", params={"path": imgs[0]["path"]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/images/thumb", params={"path": "/etc/passwd"}).status_code == 404
