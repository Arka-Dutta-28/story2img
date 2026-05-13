"""
generator/image_generator.py
-----------------------------
Orchestration layer for image generation backends.

Responsibilities
----------------
- Select the configured generation backend based on config
- Manage backend lifecycle and pipeline reuse
- Resolve seeds and assemble metadata for generated images
- Provide unified interface across all backend types
- Keep SDXL behavior compatible while supporting new backends

Architecture role
-----------------
This module serves as the main entry point for image generation in the
story2img pipeline. It abstracts away backend-specific details while
providing consistent metadata handling and seed management across all
supported backends (SDXL, FLUX, future backends).

Public interface
----------------
    generate_images(prompt, n, config) -> List[GeneratedImage]

Backend orchestration
---------------------
- Backend selection via config["generation"]["backend"]
- Automatic backend caching and lifecycle management
- Seed resolution with incremental/random strategies
- Metadata assembly with backend-specific fields
- Error handling and logging for generation failures

Supported backends
------------------
- sdxl: SDXL with ControlNet support
- flux: FLUX.1-dev with precision options
- Extensible: New backends added via backend_factory registry

Seed management
---------------
- incremental: Seeds start from base_seed and increment
- random: Random seeds within uint32 range
- Reproducible results with same seed across backends
"""

import logging
import random
import time
from typing import Any

from generator.backends.base import GeneratedImage
from generator.backends.backend_factory import get_backend

logger = logging.getLogger(__name__)


def _extract_generation_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extract generation config from full config dict.

    Parameters
    ----------
    config : dict[str, Any]
        Input config that may contain 'generation' section.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        (generation_config, full_config) where generation_config
        is the 'generation' section or the whole config if missing.

    Notes
    -----
    Handles backward compatibility for configs without 'generation' section.
    Used internally to separate generation-specific settings from global config.
    """
    if "generation" in config:
        return config["generation"], config
    return config, {"generation": config}


def _resolve_seeds(n: int, config: dict[str, Any]) -> list[int]:
    """
    Generate list of seeds based on strategy and base seed.

    Parameters
    ----------
    n : int
        Number of seeds to generate.
    config : dict[str, Any]
        Config containing seed_strategy and base_seed.

    Returns
    -------
    list[int]
        List of n seed values.

    Notes
    -----
    Supports 'incremental' and 'random' strategies. Incremental starts
    from base_seed and increments by 1. Random generates values in
    uint32 range for broad compatibility.

    Raises
    ------
    ValueError
        If seed_strategy is not 'incremental' or 'random'.
    """
    strategy = config.get("seed_strategy", "incremental")
    base_seed = int(config.get("base_seed", 42))

    if strategy == "incremental":
        return [base_seed + i for i in range(n)]

    if strategy == "random":
        return [random.randint(0, 2**32 - 1) for _ in range(n)]

    raise ValueError(
        f"Unknown seed_strategy: {strategy!r}. Supported values: 'incremental', 'random'."
    )


def generate_images(
    prompt: str,
    n: int,
    config: dict[str, Any],
) -> list[GeneratedImage]:
    """
    Generate multiple images from prompt using configured backend.

    This is the main entry point for image generation. It selects the
    appropriate backend, manages pipeline lifecycle, resolves seeds,
    and assembles comprehensive metadata for each generated image.

    Parameters
    ----------
    prompt : str
        Text prompt describing the desired image.
    n : int
        Number of images to generate (batch size).
    config : dict[str, Any]
        Configuration dictionary containing backend selection and parameters.

    Returns
    -------
    list[GeneratedImage]
        List of GeneratedImage objects with PIL images and metadata.

    Notes
    -----
    Backend selection via config["generation"]["backend"] (default: "sdxl").
    Supports SDXL and FLUX backends with automatic pipeline management.
    Seeds resolved based on config seed_strategy. Each image gets unique
    metadata including backend name, precision, and generation time.

    Raises
    ------
    ValueError
        If prompt is empty or n < 1.
    RuntimeError
        If backend generation fails.

    Edge cases
    ----------
    Single image generation (n=1) works normally.
    Backend switching handled automatically with proper cleanup.
    Config without 'generation' section uses whole config as generation config.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    generation_config, full_config = _extract_generation_config(config)
    backend = get_backend(full_config)

    backend_name = backend.backend_name
    backend_block = full_config.get("backends", {}).get(backend_name, {})
    model_id = generation_config.get("model_id") or backend_block.get("model_id")
    precision = generation_config.get("precision") or backend_block.get("precision", "fp16")
    quantized = bool(generation_config.get("quantized", backend_block.get("quantized", False)))

    logger.info(
        "generate_images | backend=%s model=%s precision=%s quantized=%s",
        backend_name,
        model_id,
        precision,
        quantized,
    )

    steps = int(generation_config.get("steps", 30))
    guidance_scale = float(generation_config.get("guidance_scale", 7.5))
    height = int(generation_config.get("height", 768))
    width = int(generation_config.get("width", 768))

    logger.info(
        "generate_images | n=%d steps=%d guidance=%.1f %dx%d",
        n,
        steps,
        guidance_scale,
        width,
        height,
    )

    seeds = _resolve_seeds(n, generation_config)
    logger.info("Seeds for this call: %s", seeds)

    backend.load_pipeline()
    logger.info("Selected backend=%s", backend_name)

    results: list[GeneratedImage] = []
    logger.debug("[GENERATOR] Prompt: %s", prompt)

    for idx, seed in enumerate(seeds):
        logger.info("Generating candidate %d/%d seed=%d", idx + 1, n, seed)
        start = time.perf_counter()

        try:
            image = backend.generate(prompt=prompt, seed=seed, config=full_config)
        except Exception as exc:
            logger.error(
                "Generation failed for candidate %d (seed=%d): %s",
                idx,
                seed,
                exc,
            )
            raise

        elapsed = time.perf_counter() - start
        results.append(
            GeneratedImage(
                image=image,
                seed=seed,
                candidate_idx=idx,
                prompt=prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                backend_name=backend_name,
                precision=precision,
                quantized=quantized,
                generation_time=elapsed,
            )
        )
        logger.info("Candidate %d/%d done seed=%d time=%.2fs", idx + 1, n, seed, elapsed)

    logger.info("generate_images complete | returned %d images", len(results))
    return results