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
    """
    Lazily load the on-disk JSON embedding cache into a module-global dict.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, Any]
        Mapping from cache key to serialised embedding list; empty dict if the
        file is missing or invalid.

    Notes
    -----
    Uses ``EMBEDDING_CACHE_PATH``; ensures parent directory exists. On failure
    logs a warning and stores ``{}`` in ``_embedding_json_cache``.

    Edge cases
    ----------
    If the JSON root is not a dict, replaces with ``{}``. Subsequent calls
    return the cached in-memory object without re-reading disk.
    """
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
    """
    Persist ``cache`` to ``EMBEDDING_CACHE_PATH`` as indented JSON.

    Parameters
    ----------
    cache : dict[str, Any]
        Serializable mapping to write.

    Returns
    -------
    None

    Notes
    -----
    Creates parent directories as needed. Swallows exceptions after logging.

    Edge cases
    ----------
    Write failures only emit a warning; caller state remains unchanged on disk.
    """
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EMBEDDING_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("embedding cache save failed: %s", exc)


def _embedding_cache_key(text: str) -> str:
    """
    Compute a SHA-256 hex digest of UTF-8 encoded ``text``.

    Parameters
    ----------
    text : str
        Input string used as cache key material.

    Returns
    -------
    str
        64-character hexadecimal digest.

    Notes
    -----
    Uses ``hashlib.sha256`` over ``text.encode("utf-8")``.

    Edge cases
    ----------
    Distinct Unicode strings that normalise differently remain distinct keys.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_model(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k"):
    """
    Initialise OpenCLIP model, image preprocess, tokenizer, and device once.

    Parameters
    ----------
    model_name : str, optional
        OpenCLIP architecture string (default ``ViT-L-14``).
    pretrained : str, optional
        Pretrained checkpoint tag (default ``laion2b_s32b_b82k``).

    Returns
    -------
    tuple
        ``(model, preprocess, tokenizer, device)`` with ``model`` in eval mode.

    Notes
    -----
    Caches in module globals when first called. Chooses ``cuda`` if available
    else ``cpu``. Builds transforms via ``open_clip.create_model_and_transforms``.

    Raises
    ------
    ImportError
        If ``open_clip`` is not installed.

    Edge cases
    ----------
    Ignores ``model_name``/``pretrained`` mismatch once cache is warm; first call
    determines cached instance.
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
    Encode a PIL image to an L2-normalised CLIP image embedding.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image; converted to RGB if needed.

    Returns
    -------
    torch.Tensor
        1-D float tensor on CPU, L2-normalised along the feature dimension.

    Notes
    -----
    Runs ``model.encode_image`` under ``torch.no_grad`` on the active CLIP device.

    Edge cases
    ----------
    Non-RGB modes are converted before preprocessing.
    """
    model, preprocess, _, device = _load_model()

    if img.mode != "RGB":
        img = img.convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(tensor)

    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0).cpu()


def _encode_text_uncached(text: str) -> torch.Tensor:
    """
    Compute a CLIP text embedding without reading or writing the JSON cache.

    Parameters
    ----------
    text : str
        Input string tokenised as a single batch element.

    Returns
    -------
    torch.Tensor
        1-D normalised CPU tensor.

    Notes
    -----
    Uses the module singleton CLIP model and tokenizer.

    Edge cases
    ----------
    Long texts are handled per tokenizer truncation rules inside OpenCLIP.
    """
    model, _, tokenizer, device = _load_model()

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(tokens)

    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0).cpu()


def get_text_embedding(text: str) -> torch.Tensor:
    """
    Return a CLIP text embedding, using a JSON file cache when possible.

    Parameters
    ----------
    text : str
        Query string; cache key is ``_embedding_cache_key(text)``.

    Returns
    -------
    torch.Tensor
        Float32 tensor reconstructed from cache or from ``_encode_text_uncached``.

    Notes
    -----
    On cache miss, encodes, stores ``emb.tolist()`` under the key, saves JSON,
    and returns the fresh tensor.

    Edge cases
    ----------
    Prints ``CACHE HIT: embedding`` on hits. Cache corruption is not repaired
    beyond JSON load failure handling in ``_load_embedding_cache``.
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
    Alias for ``get_text_embedding`` used by scoring helpers.

    Parameters
    ----------
    text : str
        Text to embed.

    Returns
    -------
    torch.Tensor
        Same as ``get_text_embedding``.

    Notes
    -----
    Delegates entirely to ``get_text_embedding``.

    Edge cases
    ----------
    Identical to ``get_text_embedding``.
    """
    return get_text_embedding(text)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Dot-product similarity for L2-normalised 1-D tensors.

    Parameters
    ----------
    a, b : torch.Tensor
        1-D tensors expected to be unit norm.

    Returns
    -------
    float
        Scalar ``torch.dot(a, b)`` as Python float.

    Notes
    -----
    Does not renormalise; assumes upstream CLIP encoders normalised embeddings.

    Edge cases
    ----------
    If inputs are not normalised, the value is not guaranteed to lie in
    ``[-1, 1]``.
    """
    return float(torch.dot(a, b).item())


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def image_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    CLIP cosine similarity between two images.

    Parameters
    ----------
    img1, img2 : PIL.Image.Image
        Inputs passed through ``_encode_image``.

    Returns
    -------
    float
        Similarity score from ``_cosine_similarity``.

    Notes
    -----
    Logs debug with the numeric result.

    Edge cases
    ----------
    See ``_cosine_similarity`` for assumptions on normalisation.
    """
    emb1 = _encode_image(img1)
    emb2 = _encode_image(img2)
    score = _cosine_similarity(emb1, emb2)
    logger.debug("image_image_similarity = %.4f", score)
    return score


def text_image_similarity(text: str, img: Image.Image) -> float:
    """
    CLIP cosine similarity between a caption and an image.

    Parameters
    ----------
    text : str
        Scene or query string (cached embedding path).
    img : PIL.Image.Image
        Candidate image embedding from ``_encode_image``.

    Returns
    -------
    float
        Similarity from ``_cosine_similarity``.

    Notes
    -----
    Logs debug with the score.

    Edge cases
    ----------
    Uses disk-backed text cache via ``_encode_text``.
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
    Score candidate images with CLIP and select the highest weighted total.

    Parameters
    ----------
    images : list[PIL.Image.Image]
        Non-empty list of candidates in evaluation order.
    scene_text : str
        Text used to build a single shared text embedding for alignment.
    reference_image : PIL.Image.Image or None
        If provided, identity term uses CLIP similarity to this image; else 0.0.
    previous_image : PIL.Image.Image or None
        If provided, temporal term uses similarity to this image; else 0.0.
    weights : dict
        Must contain float-compatible ``w_align``, ``w_identity``, ``w_temporal``.

    Returns
    -------
    dict
        Keys ``best_index`` (argmax of ``final_score``), ``best_image`` (input
        image at that index), and ``scores`` (list of per-candidate component
        dicts).

    Notes
    -----
    Precomputes embeddings for reference, previous, all candidates, and
    ``scene_text`` once. For each candidate computes ``scene_alignment`` as
    text-image cosine, ``identity_consistency`` and ``temporal_consistency`` via
    image-image cosine when embeddings exist, else zero. ``final_score`` is the
    weighted sum. Selects ``best_index`` with ``max`` on ``final_score``.

    Raises
    ------
    ValueError
        If ``images`` is empty or required weight keys are missing.

    Edge cases
    ----------
    The error message for missing keys references capitalised key names while
    the code reads lowercase ``w_*`` keys. Ties in ``final_score`` resolve to the
    first maximal index per Python ``max`` ordering.
    """
    if not images:
        raise ValueError("images list must not be empty.")

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

    ref_emb  = _encode_image(reference_image) if reference_image is not None else None
    prev_emb = _encode_image(previous_image)  if previous_image  is not None else None
    image_embeddings = [_encode_image(img) for img in images]

    all_scores: list[dict] = []
    text_emb = _encode_text(scene_text)

    for idx, img in enumerate(image_embeddings):
        logger.info("Scoring candidate %d/%d", idx + 1, len(images))

        scene_align   = float(_cosine_similarity(text_emb, img))

        identity_cons = (
            float(_cosine_similarity(img, ref_emb))
            if ref_emb is not None
            else 0.0
        )

        temporal_cons = (
            float(_cosine_similarity(img, prev_emb))
            if prev_emb is not None
            else 0.0
        )

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
