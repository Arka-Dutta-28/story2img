"""
generator/controlled_generator.py
-----------------------------------
Optional SDXL + ControlNet path driven by constraints[\"layout\"].

Does not replace ``image_generator.generate_images``; use when config enables ControlNet.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from generator.image_generator import GeneratedImage, _resolve_seeds

logger = logging.getLogger(__name__)

_cn_pipeline = None
_cn_key: Optional[tuple[str, str]] = None


def load_controlnet_pipeline(base_model_id: str, controlnet_model_id: str):
    """
    Return a cached ``StableDiffusionXLControlNetPipeline`` pair for given ids.

    Parameters
    ----------
    base_model_id : str
        Hugging Face SDXL base checkpoint id.
    controlnet_model_id : str
        Hugging Face ControlNet weights id.

    Returns
    -------
    StableDiffusionXLControlNetPipeline
        Pipeline moved to ``cuda``, ``mps``, or ``cpu`` with matching dtype.

    Notes
    -----
    Caches in module globals ``_cn_pipeline`` and ``_cn_key``. Builds
    ``ControlNetModel`` and SDXL ControlNet pipeline, enables xformers when not
    on CPU (ignores failure).

    Raises
    ------
    ImportError
        If ``torch`` or ``diffusers`` imports fail.

    Edge cases
    ----------
    CPU path logs a performance warning and uses ``float32``.
    """
    global _cn_pipeline, _cn_key

    key = (base_model_id, controlnet_model_id)
    if _cn_pipeline is not None and _cn_key == key:
        logger.debug("Reusing cached ControlNet pipeline for %s + %s", base_model_id, controlnet_model_id)
        return _cn_pipeline

    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "diffusers / torch required for ControlNet. "
            "Run: pip install diffusers transformers accelerate torch"
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        logger.warning("ControlNet on CPU: generation will be slow.")

    logger.info(
        "Loading ControlNet | base=%s  controlnet=%s  device=%s  dtype=%s",
        base_model_id, controlnet_model_id, device, dtype,
    )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)

    if device != "cpu":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.debug("xformers not available for ControlNet pipeline")

    _cn_pipeline = pipe
    _cn_key = key
    logger.info("ControlNet pipeline ready")
    return _cn_pipeline


def layout_to_control_image(layout: dict[str, Any], width: int, height: int) -> Image.Image:
    """
    Rasterise character layout into an RGB control image for ControlNet.

    Parameters
    ----------
    layout : dict[str, Any]
        Expected to contain ``characters`` list of dicts with ``name``, ``position``
        in ``{left,center,right}``, and ``depth`` in
        ``{foreground,midground,background}``.
    width : int
        Output image width in pixels.
    height : int
        Output image height in pixels.

    Returns
    -------
    PIL.Image.Image
        ``RGB`` image derived from grayscale ellipses and two-letter labels.

    Notes
    -----
    Draws filled ellipses per valid character entry; maps position to horizontal
    fraction and depth to vertical fraction; ellipse radius depends on depth.
    Appends uppercase first-two letters of ``name`` as text.

    Edge cases
    ----------
    If ``layout`` is not a dict or ``characters`` is not a list, returns a black
    ``RGB`` image. Skips character dicts with unknown position/depth. Uses a
    heuristic grayscale value spread across characters; single-character layouts
    use value 200.
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

        if c["depth"] == "foreground":
            radius = 80
        elif c["depth"] == "midground":
            radius = 60
        else:
            radius = 40

        value = int(50 + (idx * 150 / max(1, len(chars) - 1))) if len(chars) > 1 else 200

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=value,
        )

        try:
            font = ImageFont.load_default()
        except:
            font = None

        label = c["name"][:2].upper()

        draw.text(
            (x - 10, y - 10),
            label,
            fill=value,
            font=font
        )

    return img.convert("RGB")


def generate_images_controlled(
    prompt: str,
    layout: dict[str, Any],
    n: int,
    config: dict[str, Any],
) -> list[GeneratedImage]:
    """
    Generate ``n`` ControlNet-conditioned SDXL images for a fixed control map.

    Parameters
    ----------
    prompt : str
        Text prompt; must be non-empty when stringified and stripped.
    layout : dict[str, Any]
        Passed to ``layout_to_control_image`` to build the conditioning image.
    n : int
        Number of candidates; must be at least 1.
    config : dict[str, Any]
        Base generation parameters plus ``controlnet`` block (``model_id``,
        ``conditioning_scale``, ``save_debug_control``).

    Returns
    -------
    list[GeneratedImage]
        Same structure as ``generate_images`` outputs.

    Notes
    -----
    Builds ``control_img``, optionally saves under ``debug_outputs`` when
    ``save_debug_control`` is true (default), loads pipeline, resolves seeds via
    ``_resolve_seeds``, and calls ``pipe`` with ``image=control_img`` and
    ``controlnet_conditioning_scale``.

    Raises
    ------
    ValueError
        For invalid ``prompt`` or ``n``.
    ImportError
        If ``torch`` is missing.
    RuntimeError
        If a pipeline call fails.

    Edge cases
    ----------
    Debug control filename includes a random integer in ``[0, 100000]``.
    """
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt must be a non-empty string.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    cn_block = config.get("controlnet") or {}
    model_id = config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
    cn_model_id = cn_block.get("model_id", "diffusers/controlnet-depth-sdxl-1.0")

    steps = int(config.get("steps", 30))
    guidance_scale = float(config.get("guidance_scale", 7.5))
    height = int(config.get("height", 768))
    width = int(config.get("width", 768))
    conditioning_scale = float(cn_block.get("conditioning_scale", 0.5))

    logger.info(
        "generate_images_controlled | n=%d  steps=%d  guidance=%.1f  %dx%d  base=%s  cn=%s",
        n, steps, guidance_scale, width, height, model_id, cn_model_id,
    )

    control_img = layout_to_control_image(layout, width, height)
    print("CONTROL IMAGE GENERATED")
    print("LAYOUT USED:", layout)

    if cn_block.get("save_debug_control", True):
        debug_dir = "debug_outputs"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"control_{random.randint(0, 100000)}.png")
        control_img.save(debug_path)
        print(f"CONTROL IMAGE SAVED: {debug_path}")

    pipe = load_controlnet_pipeline(model_id, cn_model_id)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required.") from exc

    seeds = _resolve_seeds(n, config)
    logger.info("Seeds (controlled): %s", seeds)

    results: list[GeneratedImage] = []

    for idx, seed in enumerate(seeds):
        logger.info("Controlled candidate %d/%d  seed=%d", idx + 1, n, seed)
        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

        try:
            output = pipe(
                prompt=prompt,
                image=control_img,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                controlnet_conditioning_scale=conditioning_scale,
                num_images_per_prompt=1,
            )
        except Exception as exc:
            logger.error("ControlNet pipeline failed for candidate %d: %s", idx, exc)
            raise RuntimeError(
                f"ControlNet generation failed for candidate {idx} (seed={seed}): {exc}"
            ) from exc

        pil_image = output.images[0]
        results.append(
            GeneratedImage(
                image=pil_image,
                seed=seed,
                candidate_idx=idx,
                prompt=prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
        )

    logger.info("generate_images_controlled complete | %d images", len(results))
    return results
