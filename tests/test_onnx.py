"""ONNX Runtime backend: chosen first, falls back cleanly, and gives sensible vectors."""

import numpy as np
import pytest
from PIL import Image

from smartsearch import embedder as emb_mod
from smartsearch import onnx_backend
from smartsearch.embedder import HashEmbedder, make_embedder

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _cached(repo, files):
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(repo, f), str) for f in files)


def test_model_files_lists_only_what_onnx_loads():
    files = dict(onnx_backend.model_files("auto", MINILM, "clip-ViT-B-32"))
    assert files[MINILM] == onnx_backend.TEXT_FILES and "onnx/model.onnx" in files[MINILM]
    assert files["Qdrant/clip-ViT-B-32-vision"] == ["model.onnx", "preprocessor_config.json"]
    assert files["Qdrant/clip-ViT-B-32-text"] == ["model.onnx", "tokenizer.json"]
    assert onnx_backend.model_files("hash", MINILM, "some-other-clip") == []


def test_auto_falls_back_to_hash_when_no_backend_loads(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("blocked by policy")

    monkeypatch.setattr(onnx_backend, "OnnxTextEmbedder", broken)
    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", broken)
    assert isinstance(make_embedder("auto", MINILM), HashEmbedder)
    with pytest.raises(RuntimeError):
        make_embedder("onnx", MINILM)


@pytest.mark.skipif(not (onnx_backend.available() and _cached(MINILM, onnx_backend.TEXT_FILES)),
                    reason="ONNX text model not downloaded")
def test_text_model_ranks_meaning():
    e = make_embedder("auto", MINILM)
    assert type(e).__name__ == "OnnxTextEmbedder" and e.dim == 384 and e.name == MINILM
    q, near, far = e.encode(["How hot should oolong tea be?", "Steep oolong at 90C for 3 minutes.",
                             "The plumber's invoice total was 412.50."])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-4)
    assert q @ near > q @ far + 0.2
    assert e.encode([]).shape == (0, 384)


@pytest.mark.skipif(not (onnx_backend.available()
                         and _cached("Qdrant/clip-ViT-B-32-vision", onnx_backend.CLIP_VISION_FILES)
                         and _cached("Qdrant/clip-ViT-B-32-text", onnx_backend.CLIP_TEXT_FILES)),
                    reason="ONNX CLIP model not downloaded")
def test_clip_matches_images_to_descriptions():
    from smartsearch.images import make_image_encoder

    enc = make_image_encoder("clip-ViT-B-32")
    assert type(enc).__name__ == "OnnxClipEncoder"
    red, blue = Image.new("RGB", (320, 200), (220, 20, 20)), Image.new("RGB", (120, 300), (20, 20, 220))
    im = enc.encode_images([red, blue])
    tx = enc.encode_texts(["a plain red image", "a plain blue image"])
    sims = tx @ im.T
    assert sims[0, 0] > sims[0, 1] and sims[1, 1] > sims[1, 0]
