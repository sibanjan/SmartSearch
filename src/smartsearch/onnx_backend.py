"""Run the embedding models with ONNX Runtime instead of PyTorch.

Same models, same vectors (cosine similarity with the PyTorch output above 0.999), so an
index built either way stays valid. Why it is the default:

- ONNX Runtime's binaries are signed by Microsoft, so Windows Smart App Control loads them;
  it can block PyTorch's unsigned DLLs, which silently left SmartSearch on keyword search.
- It is about 600 MB smaller to install than PyTorch, and starts faster.

Text models: any sentence-transformers repository that ships onnx/model.onnx (all-MiniLM-L6-v2
does). Image model: CLIP ViT-B/32, from Qdrant's ONNX export of the same OpenAI weights.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TEXT_FILES = ["onnx/model.onnx", "tokenizer.json", "modules.json", "1_Pooling/config.json",
              "sentence_bert_config.json"]
CLIP_ONNX = {  # image_model name -> (vision repo, text repo)
    "clip-ViT-B-32": ("Qdrant/clip-ViT-B-32-vision", "Qdrant/clip-ViT-B-32-text"),
    "sentence-transformers/clip-ViT-B-32": ("Qdrant/clip-ViT-B-32-vision", "Qdrant/clip-ViT-B-32-text"),
}
CLIP_VISION_FILES = ["model.onnx", "preprocessor_config.json"]
CLIP_TEXT_FILES = ["model.onnx", "tokenizer.json"]


def repo_id(name: str) -> str:
    """sentence-transformers' rule: a bare model name belongs to its organisation."""
    return name if "/" in name else f"sentence-transformers/{name}"


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except Exception:
        return False
    return True


def model_files(embedder: str, model_name: str, image_model: str) -> list[tuple[str, list[str]]]:
    """(repository, files) the ONNX backend needs for this configuration; a first-run
    downloader fetches exactly these and nothing else."""
    out: list[tuple[str, list[str]]] = []
    if embedder.lower() in ("auto", "onnx") and not Path(model_name).expanduser().exists():
        out.append((repo_id(model_name), TEXT_FILES))
    if image_model in CLIP_ONNX:
        vision, text = CLIP_ONNX[image_model]
        out += [(vision, CLIP_VISION_FILES), (text, CLIP_TEXT_FILES)]
    return out


def _fetch(repo: str, filename: str) -> str:
    """A model file from the local cache, downloading it only if it is not there yet."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        log.info("Downloading %s/%s (first run only)...", repo, filename)
        return hf_hub_download(repo, filename)


def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)  # leave the laptop usable
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class OnnxTextEmbedder:
    """A sentence-transformers text model (Transformer -> Pooling -> Normalize) on ONNX."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from tokenizers import Tokenizer

        repo = model_name if Path(model_name).expanduser().exists() else repo_id(model_name)
        get = (lambda f: str(Path(repo).expanduser() / f)) if Path(repo).expanduser().exists() \
            else (lambda f: _fetch(repo, f))
        self.session = _session(get("onnx/model.onnx"))
        self.inputs = {i.name for i in self.session.get_inputs()}
        pooling = json.loads(Path(get("1_Pooling/config.json")).read_text(encoding="utf-8"))
        modules = json.loads(Path(get("modules.json")).read_text(encoding="utf-8"))
        max_len = json.loads(Path(get("sentence_bert_config.json")).read_text(encoding="utf-8")).get(
            "max_seq_length", 256)
        self.cls = bool(pooling.get("pooling_mode_cls_token"))
        self.normalize = any(m.get("type", "").endswith("Normalize") for m in modules)
        self.tok = Tokenizer.from_file(get("tokenizer.json"))
        self.tok.enable_truncation(max_length=int(max_len))
        pad = self.tok.token_to_id("[PAD]") or self.tok.token_to_id("<pad>") or 0
        self.tok.enable_padding(pad_id=pad, pad_token=self.tok.id_to_token(pad) or "[PAD]")
        self.batch_size = batch_size
        self.name = model_name  # same name as the PyTorch backend: the vectors are the same
        self.dim = int(pooling.get("word_embedding_dimension") or 0)
        lname = model_name.lower()
        self._q_prefix = "query: " if "e5" in lname else ""
        self._d_prefix = "passage: " if "e5" in lname else ""
        log.info("Loaded embedding model %s (ONNX Runtime)", model_name)

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        prefix = self._q_prefix if query else self._d_prefix
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch([prefix + t for t in texts[i:i + self.batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.session.run(None, feed)[0]
            if self.cls:
                vec = hidden[:, 0]
            else:  # mean of the real (unpadded) tokens
                m = mask[..., None].astype(np.float32)
                vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(vec)
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.vstack(out).astype(np.float32)
        return _normalize(vecs) if self.normalize else vecs


class OnnxClipEncoder:
    """CLIP on ONNX: images and text in one vector space, like the PyTorch ClipEncoder."""

    def __init__(self, model_name: str, batch_size: int = 16):
        from tokenizers import Tokenizer

        vision, text = CLIP_ONNX[model_name]
        self.vision = _session(_fetch(vision, "model.onnx"))
        self.text = _session(_fetch(text, "model.onnx"))
        pre = json.loads(Path(_fetch(vision, "preprocessor_config.json")).read_text(encoding="utf-8"))
        self.size = int(pre.get("size", {}).get("shortest_edge", 224))
        self.crop = int(pre.get("crop_size", {}).get("height", 224))
        self.mean = np.array(pre.get("image_mean"), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(pre.get("image_std"), dtype=np.float32).reshape(3, 1, 1)
        self.tok = Tokenizer.from_file(_fetch(text, "tokenizer.json"))
        self.tok.enable_truncation(max_length=77)
        pad = self.tok.token_to_id("<|endoftext|>") or 0
        self.tok.enable_padding(pad_id=pad, pad_token="<|endoftext|>")
        self.text_inputs = {i.name for i in self.text.get_inputs()}
        self.name = model_name
        self.batch_size = batch_size
        log.info("Loaded image model %s (ONNX Runtime)", model_name)

    def _pixels(self, img) -> np.ndarray:
        """CLIP's preprocessing: shortest edge to 224 (bicubic), centre crop, scale, normalise."""
        from PIL import Image

        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.crop, int(w * s)), max(self.crop, int(h * s))), Image.BICUBIC)
        w, h = img.size
        left, upper = (w - self.crop) // 2, (h - self.crop) // 2
        img = img.crop((left, upper, left + self.crop, upper + self.crop))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def encode_images(self, images: list) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([self._pixels(im) for im in images[i:i + self.batch_size]])
            out.append(self.vision.run(["image_embeds"], {"pixel_values": batch.astype(np.float32)})[0])
        return _normalize(np.vstack(out)) if out else np.zeros((0, 512), dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch(texts[i:i + self.batch_size])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
            if "attention_mask" in self.text_inputs:
                feed["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
            out.append(self.text.run(["text_embeds"], feed)[0])
        return _normalize(np.vstack(out)) if out else np.zeros((0, 512), dtype=np.float32)
