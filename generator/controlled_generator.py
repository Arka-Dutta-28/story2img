"""
generator/controlled_generator.py
-----------------------------------
Controlled generation orchestration for backends that support ControlNet.

This module retains the existing layout-derived control image path and delegates
model-specific ControlNet pipeline logic to the selected backend. It provides
a unified interface for layout-controlled generation across all supported backends.

Responsibilities
----------------
- Convert story layouts to ControlNet conditioning images
- Orchestrate ControlNet generation through backend abstraction
- Maintain backward compatibility with existing layout format
- Handle ControlNet availability checking and error handling
- Provide unified interface for layout-controlled image generation

Architecture role
-----------------
This module bridges the story layout system with backend-specific ControlNet
implementations. It handles layout parsing and control image generation while
delegating the actual conditioned generation to the selected backend.

Layout processing
-----------------
- Converts character positions/depths to grayscale control images
- Supports foreground/midground/background depth layers
- Handles multiple characters with positional encoding
- Generates conditioning images compatible with depth ControlNet

Backend integration
-------------------
- Checks backend ControlNet support before generation
- Delegates to backend-specific ControlNet pipelines
- Passes control images and conditioning parameters to backend
- Maintains consistent API across ControlNet-capable backends

Supported backends
------------------
- sdxl: Full ControlNet support with depth conditioning
- flux: ControlNet not yet supported (may change in future)

Unsupported features
--------------------
- Multiple ControlNet models simultaneously
- Non-depth ControlNet types (canny, pose, etc.)
- Dynamic layout formats beyond character positioning
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from generator.backends.backend_factory import get_backend
from generator.backends.base import GeneratedImage

logger = logging.getLogger(__name__)


def layout_to_control_image(layout: dict[str, Any], width: int, height: int) -> Image.Image:
    """
    Convert story layout to ControlNet depth conditioning image.

    Parameters
    ----------
    layout : dict[str, Any]
        Layout specification with character positions and depths.
    width : int
        Output image width in pixels.
    height : int
        Output image height in pixels.

    Returns
    -------
    Image.Image
        Grayscale PIL image for ControlNet depth conditioning.

    Notes
    -----
    Converts character positions to depth values:
    - foreground: 0.7 (lighter)
    - midground: 0.5 (medium)
    - background: 0.3 (darker)

    Position mapping: left=0.25, center=0.5, right=0.75 of width.
    Characters rendered as filled circles with depth-based grayscale.

    Edge cases
    ----------
    Invalid layout returns blank image.
    Missing characters/depths handled gracefully.
    Non-dict layout returns blank image.
    """
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    positions = {
        "left": 0.25,
        "center": 0.5,
        "right": 0.75,
    }

    depths = {
        "foreground": 0.7,
        "midground": 0.5,
        "background": 0.3,
    }

    if not isinstance(layout, dict):
        return img.convert("RGB")

    chars = layout.get("characters", [])
    if not isinstance(chars, list):
        return img.convert("RGB")

    for idx, c in enumerate(chars):
        if not isinstance(c, dict):
            continue
        pos = c.get("position")
        dep = c.get("depth")

        if pos not in positions or dep not in depths:
            continue

        x = int(positions[pos] * width)
        y = int(depths[dep] * height)
        radius = 80 if dep == "foreground" else 60 if dep == "midground" else 40
        value = int(50 + (idx * 150 / max(1, len(chars) - 1))) if len(chars) > 1 else 200

        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=value)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        label = str(c.get("name", ""))[:2].upper()
        draw.text((x - 10, y - 10), label, fill=value, font=font)

    return img.convert("RGB")


def _resolve_seeds(n: int, config: dict[str, Any]) -> list[int]:
    """
    Generate list of seeds for controlled generation.

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
    Same seed resolution logic as standard generation.
    Supports 'incremental' and 'random' strategies.
    Ensures reproducible controlled generation.

    Raises
    ------
    ValueError
        If seed_strategy is not supported.
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
        (generation_config, full_config) tuple.

    Notes
    -----
    Same logic as standard generation for consistency.
    Handles backward compatibility for older configs.
    """
    if "generation" in config:
        return config["generation"], config
    return config, {"generation": config}


def generate_images_controlled(
    prompt: str,
    layout: dict[str, Any],
    n: int,
    config: dict[str, Any],
) -> list[GeneratedImage]:
    """
    Generate images with ControlNet conditioning from story layout.

    This function converts story layouts to control images and generates
    conditioned images using backends that support ControlNet.

    Parameters
    ----------
    prompt : str
        Text prompt for the scene.
    layout : dict[str, Any]
        Story layout specification with character positions/depths.
    n : int
        Number of images to generate.
    config : dict[str, Any]
        Configuration with backend selection and ControlNet parameters.

    Returns
    -------
    list[GeneratedImage]
        List of generated images with metadata.

    Notes
    -----
    Converts layout to depth control image using layout_to_control_image.
    Checks backend ControlNet support before generation.
    Uses depth ControlNet by default (configurable via controlnet.model_id).
    Conditioning scale defaults to 0.5 but configurable.

    Raises
    ------
    ValueError
        If prompt is empty or n < 1.
    RuntimeError
        If selected backend doesn't support ControlNet.

    Edge cases
    ----------
    Invalid layouts generate without conditioning (standard generation).
    Backend switching handled with proper ControlNet pipeline management.
    ControlNet parameters override defaults from config.
    """
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt must be a non-empty string.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    generation_config, full_config = _extract_generation_config(config)
    backend = get_backend(full_config)

    if not backend.supports_controlnet():
        raise RuntimeError(
            f"Backend '{backend.backend_name}' does not support ControlNet generation."
        )

    steps = int(generation_config.get("steps", 30))
    guidance_scale = float(generation_config.get("guidance_scale", 7.5))
    height = int(generation_config.get("height", 768))
    width = int(generation_config.get("width", 768))
    conditioning_scale = float((config.get("controlnet") or {}).get("conditioning_scale", 0.5))

    logger.info(
        "generate_images_controlled | backend=%s steps=%d guidance=%.1f %dx%d",
        backend.backend_name,
        steps,
        guidance_scale,
        width,
        height,
    )

    control_img = layout_to_control_image(layout, width, height)

    if (config.get("controlnet") or {}).get("save_debug_control", True):
        debug_dir = "debug_outputs"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"control_{random.randint(0, 100000)}.png")
        control_img.save(debug_path)
        logger.info("Saved debug control image: %s", debug_path)

    seeds = _resolve_seeds(n, generation_config)
    logger.info("Seeds (controlled): %s", seeds)

    results: list[GeneratedImage] = []
    logger.debug("[CONTROLLED GENERATOR] Prompt: %s", prompt)

    for idx, seed in enumerate(seeds):
        logger.info("Controlled candidate %d/%d seed=%d", idx + 1, n, seed)
        start = time.perf_counter()

        try:
            image = backend.generate_controlled(
                prompt=prompt,
                control_image=control_img,
                seed=seed,
                config=full_config,
            )
        except Exception as exc:
            logger.error(
                "ControlNet generation failed for candidate %d (seed=%d): %s",
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
                backend_name=backend.backend_name,
                precision=str(generation_config.get("precision", "fp16")),
                quantized=bool(generation_config.get("quantized", False)),
                generation_time=elapsed,
            )
        )
        logger.info("Controlled candidate %d/%d done seed=%d time=%.2fs", idx + 1, n, seed, elapsed)

    logger.info("generate_images_controlled complete | returned %d images", len(results))
    return results
