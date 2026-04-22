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
    Compute a list of N seeds based on the configured seed strategy.

    Strategies
    ----------
    incremental : base_seed, base_seed+1, base_seed+2, ...
                  Fully deterministic across runs.
    random      : N independently sampled random integers in [0, 2^32).
                  Different each run, but logged for reproducibility.

    Parameters
    ----------
    n      : Number of seeds to produce.
    config : The `generation` block from config.yaml.

    Returns
    -------
    List of n integer seeds.
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
    Load (or return cached) the SDXL diffusers pipeline.

    The pipeline is cached at module level so repeated calls to
    generate_images() within a session do not reload weights.

    Parameters
    ----------
    model_id : HuggingFace model ID, e.g. "stabilityai/stable-diffusion-xl-base-1.0".

    Returns
    -------
    StableDiffusionXLPipeline
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

    # Detect device
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

    # Memory optimisations (no-op on CPU)
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
    Generate N candidate images for the given prompt using SDXL.

    Parameters
    ----------
    prompt : Text prompt describing the scene to generate.
    n      : Number of candidate images to generate.
    config : The `generation` block from config.yaml. Expected keys:
               model_id        (str)   — HuggingFace model ID
               steps           (int)   — number of inference steps
               guidance_scale  (float) — CFG guidance scale
               height          (int)   — image height in pixels
               width           (int)   — image width in pixels
               seed_strategy   (str)   — "incremental" | "random"
               base_seed       (int)   — base seed for incremental strategy

    Returns
    -------
    List of GeneratedImage, one per candidate, in generation order.

    Raises
    ------
    ValueError  if n < 1 or required config keys are missing.
    ImportError if diffusers / torch are not installed.
    RuntimeError if the pipeline call fails.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    # -- Extract config values -----------------------------------------------
    model_id       = config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
    steps          = int(config.get("steps", 30))
    guidance_scale = float(config.get("guidance_scale", 7.5))
    height         = int(config.get("height", 768))
    width          = int(config.get("width", 768))

    logger.info(
        "generate_images | n=%d  steps=%d  guidance=%.1f  %dx%d  model=%s",
        n, steps, guidance_scale, width, height, model_id,
    )

    # -- Resolve seeds --------------------------------------------------------
    seeds = _resolve_seeds(n, config)
    logger.info("Seeds for this call: %s", seeds)

    # -- Load pipeline --------------------------------------------------------
    pipe = _load_pipeline(model_id)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required. Run: pip install torch") from exc

    # -- Generate one candidate at a time (simple, traceable) ----------------
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