"""
evaluator/clip_evaluator.py
---------------------------
CLIP-based Evaluator — scores candidate images and selects the best one.

Responsibilities
----------------
- Load a pretrained OpenCLIP model once (cached singleton)
- Compute text-image similarity (scene alignment)
- Compute image-image similarity (identity and temporal consistency)
- Score all candidate images and return the best index + full score breakdown

Public interface
----------------
    image_image_similarity(img1, img2) -> float
    text_image_similarity(text, img)   -> float
    score_candidates(images, scene_text, reference_image, previous_image, weights) -> dict

No pipeline logic. No memory logic. No generator logic.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_CACHE_PATH = REPO_ROOT / "cache" / "embedding_cache.json"

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model       = None
_preprocess  = None
_tokenizer   = None
_device: Optional[str] = None

_embedding_json_cache: Optional[dict[str, Any]] = None


def _load_embedding_cache() -> dict[str, Any]:
    global _embedding_json_cache
    if _embedding_json_cache is not None:
        return _embedding_json_cache
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if EMBEDDING_CACHE_PATH.is_file():
            with EMBEDDING_CACHE_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
                _embedding_json_cache = data if isinstance(data, dict) else {}
        else:
            _embedding_json_cache = {}
    except Exception as exc:
        logger.warning("embedding cache load failed: %s", exc)
        _embedding_json_cache = {}
    return _embedding_json_cache


def _save_embedding_cache(cache: dict[str, Any]) -> None:
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EMBEDDING_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("embedding cache save failed: %s", exc)


def _embedding_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_model(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k"):
    """
    Load (or return cached) an OpenCLIP model, preprocessor, and tokenizer.

    The model is cached at module level so it is loaded only once per session.

    Parameters
    ----------
    model_name : OpenCLIP architecture name. Default: "ViT-L-14".
    pretrained : Pretrained weights tag.     Default: "laion2b_s32b_b82k".

    Returns
    -------
    (model, preprocess, tokenizer, device)
    """
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        logger.debug("Reusing cached CLIP model (%s / %s)", model_name, pretrained)
        return _model, _preprocess, _tokenizer, _device

    logger.info("Loading OpenCLIP model: %s / %s", model_name, pretrained)

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip is required. Install it with:\n"
            "  pip install open-clip-torch"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("CLIP using device: %s", device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    _model      = model
    _preprocess = preprocess
    _tokenizer  = tokenizer
    _device     = device

    logger.info("OpenCLIP model ready")
    return _model, _preprocess, _tokenizer, _device


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_image(img: Image.Image) -> torch.Tensor:
    """
    Encode a PIL image into a normalised CLIP embedding.

    Parameters
    ----------
    img : PIL Image (RGB or convertible).

    Returns
    -------
    Normalised 1-D float tensor on CPU.
    """
    model, preprocess, _, device = _load_model()

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        embedding = model.encode_image(tensor)          # (1, D)

    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0).cpu()                   # (D,)


def _encode_text_uncached(text: str) -> torch.Tensor:
    """
    Encode a text string into a normalised CLIP embedding (no disk cache).
    """
    model, _, tokenizer, device = _load_model()

    tokens = tokenizer([text]).to(device)               # (1, context_len)

    with torch.no_grad():
        embedding = model.encode_text(tokens)           # (1, D)

    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0).cpu()                   # (D,)


def get_text_embedding(text: str) -> torch.Tensor:
    """
    Text embedding with on-disk JSON cache (keyed by SHA-256 of text).
    """
    key = _embedding_cache_key(text)
    cache = _load_embedding_cache()
    if key in cache:
        print("CACHE HIT: embedding")
        vec = cache[key]
        return torch.tensor(vec, dtype=torch.float32)

    emb = _encode_text_uncached(text)
    cache[key] = emb.tolist()
    _save_embedding_cache(cache)
    return emb


def _encode_text(text: str) -> torch.Tensor:
    """
    Encode a text string into a normalised CLIP embedding (cached across runs).

    Parameters
    ----------
    text : Input string (will be truncated to model context length if needed).

    Returns
    -------
    Normalised 1-D float tensor on CPU.
    """
    return get_text_embedding(text)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two normalised 1-D tensors.

    Both tensors are assumed to already be L2-normalised, so the dot product
    equals cosine similarity directly.

    Returns
    -------
    Float in [-1.0, 1.0].
    """
    return float(torch.dot(a, b).item())


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def image_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute CLIP cosine similarity between two PIL images.

    Parameters
    ----------
    img1, img2 : PIL Images to compare.

    Returns
    -------
    Cosine similarity as a float in [-1.0, 1.0].
    """
    emb1 = _encode_image(img1)
    emb2 = _encode_image(img2)
    score = _cosine_similarity(emb1, emb2)
    logger.debug("image_image_similarity = %.4f", score)
    return score


def text_image_similarity(text: str, img: Image.Image) -> float:
    """
    Compute CLIP cosine similarity between a text string and a PIL image.

    Parameters
    ----------
    text : Description or query string.
    img  : PIL Image to compare against.

    Returns
    -------
    Cosine similarity as a float in [-1.0, 1.0].
    """
    emb_text  = _encode_text(text)
    emb_image = _encode_image(img)
    score = _cosine_similarity(emb_text, emb_image)
    logger.debug("text_image_similarity = %.4f", score)
    return score


# ---------------------------------------------------------------------------
# Candidate scorer
# ---------------------------------------------------------------------------

def score_candidates(
    images:          list[Image.Image],
    scene_text:      str,
    reference_image: Optional[Image.Image],
    previous_image:  Optional[Image.Image],
    weights:         dict,
) -> dict:
    """
    Score all candidate images and return the best one with full score breakdown.

    Scoring per candidate
    ---------------------
    scene_alignment      = text_image_similarity(scene_text, image)
    identity_consistency = image_image_similarity(image, reference_image)
                           if reference_image is not None else 0.0
    temporal_consistency = image_image_similarity(image, previous_image)
                           if previous_image is not None else 0.0
    final_score          = (w_align    * scene_alignment)
                         + (w_identity * identity_consistency)
                         + (w_temporal * temporal_consistency)

    Parameters
    ----------
    images          : List of N candidate PIL Images.
    scene_text      : Scene description string (primary field from parser).
    reference_image : Reference image for identity consistency, or None.
    previous_image  : Previous scene's selected image for temporal consistency, or None.
    weights         : Dict with keys 'scene_alignment', 'identity_consistency',
                      'temporal_consistency' — values must sum to 1.0.

    Returns
    -------
    {
        "best_index" : int,
        "best_image" : PIL.Image,
        "scores"     : [
            {
                "scene_alignment"      : float,
                "identity_consistency" : float,
                "temporal_consistency" : float,
                "final_score"          : float,
            },
            ...   # one entry per candidate, in input order
        ]
    }

    Raises
    ------
    ValueError  if images is empty or weights keys are missing.
    """
    if not images:
        raise ValueError("images list must not be empty.")

    # -- Extract weights -----------------------------------------------------
    try:
        w_align    = float(weights["w_align"])
        w_identity = float(weights["w_identity"])
        w_temporal = float(weights["w_temporal"])
    except KeyError as exc:
        raise ValueError(
            f"weights dict is missing key: {exc}. "
            "Required keys: 'W_align', 'W_identity', 'W_temporal'."
        ) from exc

    logger.info(
        "score_candidates | n=%d  w_align=%.2f  w_identity=%.2f  w_temporal=%.2f",
        len(images), w_align, w_identity, w_temporal,
    )
    logger.info(
        "reference_image=%s  previous_image=%s",
        "provided" if reference_image is not None else "None",
        "provided" if previous_image  is not None else "None",
    )

    # -- Pre-encode all images once -----------------------
    ref_emb  = _encode_image(reference_image) if reference_image is not None else None
    prev_emb = _encode_image(previous_image)  if previous_image  is not None else None
    image_embeddings = [_encode_image(img) for img in images]

    # -- Score each candidate ------------------------------------------------
    all_scores: list[dict] = []
    text_emb = _encode_text(scene_text)

    for idx, img in enumerate(image_embeddings):
        logger.info("Scoring candidate %d/%d", idx + 1, len(images))

        # Scene alignment (text ↔ image)
        scene_align   = float(_cosine_similarity(text_emb, img))

        # Identity consistency (image ↔ reference)
        identity_cons = (
            float(_cosine_similarity(img, ref_emb))
            if ref_emb is not None
            else 0.0
        )

        # Temporal consistency (image ↔ previous selected image)
        temporal_cons = (
            float(_cosine_similarity(img, prev_emb))
            if prev_emb is not None
            else 0.0
        )

        # Weighted final score
        final = (
            w_align    * scene_align
            + w_identity * identity_cons
            + w_temporal * temporal_cons
        )

        logger.info(
            "  candidate %d | align=%.4f  identity=%.4f  temporal=%.4f  final=%.4f",
            idx, scene_align, identity_cons, temporal_cons, final,
        )

        all_scores.append({
            "scene_alignment":      scene_align,
            "identity_consistency": identity_cons,
            "temporal_consistency": temporal_cons,
            "final_score":          final,
        })

    # -- Select best candidate -----------------------------------------------
    best_index = max(range(len(all_scores)), key=lambda i: all_scores[i]["final_score"])
    best_image = images[best_index]

    logger.info(
        "Best candidate: index=%d  final_score=%.4f",
        best_index, all_scores[best_index]["final_score"],
    )

    return {
        "best_index": best_index,
        "best_image": best_image,
        "scores":     all_scores,
    }
