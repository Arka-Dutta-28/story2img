"""
generator/image_generator.py
-----------------------------
Image generation module using Stable Diffusion XL (SDXL).

Responsibilities
----------------
- Load the SDXL pipeline once and reuse it across calls
- Generate exactly N candidate images for a given prompt
- Apply deterministic seeding per candidate (incremental or random)
- Read all generation parameters from the config dict
- Return a list of PIL Images with their metadata

Public interface
----------------
    generate_images(prompt, n, config) -> List[GeneratedImage]

No ControlNet. No reference images. No pipeline integration. No memory.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratedImage:
    """
    One generated candidate image and its metadata.

    Attributes
    ----------
    image        : The PIL Image object.
    seed         : The seed used to generate this image.
    candidate_idx: 0-based index within the N candidates for this prompt.
    prompt       : The prompt used (for logging/traceability).
    steps        : Inference steps used.
    guidance_scale: CFG guidance scale used.
    width        : Image width in pixels.
    height       : Image height in pixels.
    """
    image:          Image.Image
    seed:           int
    candidate_idx:  int
    prompt:         str
    steps:          int
    guidance_scale: float
    width:          int
    height:         int


# ---------------------------------------------------------------------------
# Seed strategies
# ---------------------------------------------------------------------------

def _resolve_seeds(n: int, config: dict) -> list[int]:
    """
    Produce ``n`` integer seeds according to ``config["seed_strategy"]``.

    Parameters
    ----------
    n : int
        Number of seeds required (length of returned list).
    config : dict
        Generation configuration; reads ``seed_strategy`` (default
        ``"incremental"``) and ``base_seed`` (default ``42``).

    Returns
    -------
    list[int]
        ``n`` seeds: arithmetic progression for ``incremental``, or independent
        draws for ``random``.

    Notes
    -----
    ``incremental`` uses ``[base_seed + i for i in range(n)]``. ``random`` uses
    ``random.randint(0, 2**32 - 1)`` per slot.

    Raises
    ------
    ValueError
        If ``seed_strategy`` is neither ``incremental`` nor ``random``.

    Edge cases
    ----------
    Unknown strategy strings raise with a message listing supported values.
    """
    strategy  = config.get("seed_strategy", "incremental")
    base_seed = int(config.get("base_seed", 42))

    if strategy == "incremental":
        return [base_seed + i for i in range(n)]

    if strategy == "random":
        return [random.randint(0, 2**32 - 1) for _ in range(n)]

    raise ValueError(
        f"Unknown seed_strategy: {strategy!r}. "
        "Supported values: 'incremental', 'random'."
    )


# ---------------------------------------------------------------------------
# Pipeline loader (module-level singleton)
# ---------------------------------------------------------------------------

_pipeline = None          # cached pipeline instance
_loaded_model_id: Optional[str] = None


def _load_pipeline(model_id: str):
    """
    Return a module-singleton ``StableDiffusionXLPipeline`` for ``model_id``.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier for SDXL weights.

    Returns
    -------
    StableDiffusionXLPipeline
        Loaded or cached pipeline on ``cuda``, ``mps``, or ``cpu``.

    Notes
    -----
    Reuses ``_pipeline`` when ``_loaded_model_id`` matches. Selects dtype
    ``float16`` on GPU backends else ``float32`` on CPU. Attempts
    ``enable_xformers_memory_efficient_attention`` when not on CPU.

    Raises
    ------
    ImportError
        If ``torch`` or ``diffusers`` import fails.

    Edge cases
    ----------
    If xformers enable fails, logs at DEBUG and continues without it.
    """
    global _pipeline, _loaded_model_id

    if _pipeline is not None and _loaded_model_id == model_id:
        logger.debug("Reusing cached SDXL pipeline for %s", model_id)
        return _pipeline

    logger.info("Loading SDXL pipeline: %s", model_id)

    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise ImportError(
            "Required packages not installed. Run:\n"
            "  pip install diffusers transformers accelerate torch"
        ) from exc

    if torch.cuda.is_available():
        device    = "cuda"
        dtype     = torch.float16
    elif torch.backends.mps.is_available():
        device    = "mps"
        dtype     = torch.float16
    else:
        device    = "cpu"
        dtype     = torch.float32
        logger.warning(
            "No GPU detected — running SDXL on CPU. "
            "Generation will be very slow. "
            "Consider using 512x512 resolution and fewer steps."
        )


    logger.info("Using device=%s  dtype=%s", device, dtype)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)

    if device != "cpu":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception:
            logger.debug("xformers not available; skipping")

    _pipeline        = pipe
    _loaded_model_id = model_id

    logger.info("SDXL pipeline ready")
    return _pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_images(
    prompt: str,
    n: int,
    config: dict,
) -> list[GeneratedImage]:
    """
    Generate ``n`` SDXL images for ``prompt`` using settings from ``config``.

    Parameters
    ----------
    prompt : str
        Text prompt; must be non-empty after stripping.
    n : int
        Candidate count; must be at least 1.
    config : dict
        Generation block: ``model_id`` (optional default), ``steps``,
        ``guidance_scale``, ``height``, ``width``, and seed fields consumed by
        ``_resolve_seeds``.

    Returns
    -------
    list[GeneratedImage]
        One record per seed, preserving order, each wrapping ``output.images[0]``.

    Notes
    -----
    Loads pipeline via ``_load_pipeline``, builds a ``torch.Generator`` per seed
    on ``pipe.device.type``, calls ``pipe`` with ``num_images_per_prompt=1`` per
    iteration.

    Raises
    ------
    ValueError
        For empty prompt or invalid ``n``.
    ImportError
        If ``torch`` is missing after pipeline load.
    RuntimeError
        If any ``pipe(...)`` call raises (re-raised with candidate index).

    Edge cases
    ----------
    Each failed pipeline call logs and raises; prior successful candidates are
    discarded (function does not return partial results on error).
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    model_id       = config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
    steps          = int(config.get("steps", 30))
    guidance_scale = float(config.get("guidance_scale", 7.5))
    height         = int(config.get("height", 768))
    width          = int(config.get("width", 768))

    logger.info(
        "generate_images | n=%d  steps=%d  guidance=%.1f  %dx%d  model=%s",
        n, steps, guidance_scale, width, height, model_id,
    )

    seeds = _resolve_seeds(n, config)
    logger.info("Seeds for this call: %s", seeds)

    pipe = _load_pipeline(model_id)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required. Run: pip install torch") from exc

    results: list[GeneratedImage] = []
    logger.debug(f"[GENERATOR] Prompt: {prompt}")

    for idx, seed in enumerate(seeds):
        logger.info(
            "Generating candidate %d/%d  seed=%d", idx + 1, n, seed
        )

        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

        try:
            output = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                num_images_per_prompt=1,
            )
        except Exception as exc:
            logger.error(
                "Pipeline call failed for candidate %d (seed=%d): %s",
                idx, seed, exc,
            )
            raise RuntimeError(
                f"SDXL generation failed for candidate {idx} (seed={seed}): {exc}"
            ) from exc

        pil_image = output.images[0]

        results.append(GeneratedImage(
            image          = pil_image,
            seed           = seed,
            candidate_idx  = idx,
            prompt         = prompt,
            steps          = steps,
            guidance_scale = guidance_scale,
            width          = width,
            height         = height,
        ))

        logger.info("Candidate %d/%d done  seed=%d", idx + 1, n, seed)

    logger.info("generate_images complete | returned %d images", len(results))
    return results