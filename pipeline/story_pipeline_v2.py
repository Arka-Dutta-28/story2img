"""
pipeline/story_pipeline_v2.py
------------------------------
Pipeline Orchestrator — Phase 7 (WITH MEMORY).

Responsibilities
----------------
- Accept parsed scenes, character list, and config
- For each scene: retrieve memory, build prompt, generate candidates,
  evaluate them, select the best, and update memory
- Track previous selected image for temporal consistency
- Return ordered list of selected images with per-scene logs

Public interface
----------------
    run_pipeline_v2(scenes, characters, config) -> dict

Constraints
-----------
- Does NOT modify story_pipeline.py (Phase 5)
- Does NOT modify evaluator or generator
- Uses MemoryManager, build_constraints → build_prompt_from_constraints (fallback: build_prompt), generate_images or generate_images_controlled, score_candidates
- Uses scene["description"] (NOT the built prompt) for evaluation
"""

import logging
from typing import Any, Optional
import json
from pathlib import Path
from PIL import Image

from generator.image_generator import generate_images
from generator.controlled_generator import (
    generate_images_controlled,
    layout_to_control_image,
)
from evaluator.clip_evaluator import score_candidates
from memory.memory_manager import MemoryManager
from constraints.constraint_builder import (
    build_constraints,
    build_prompt,
    build_prompt_from_constraints,
    compress_prompt,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Prompt debugger
# ---------------------------------------------------------------------------

def _analyze_prompt(prompt: str, scene_id: int) -> None:
    """
    Emit stdout diagnostics for prompt length and a short preview.

    Parameters
    ----------
    prompt : str
        Final prompt string for the current scene (not mutated).
    scene_id : int
        Scene identifier printed in the analysis header.

    Returns
    -------
    None
        Always returns ``None``; side effect is printed lines only.

    Notes
    -----
    Prints character length, ``length // 4`` as an estimated token count, optional
    warnings when length is below 80 or above 500, and a 200-character preview.

    Edge cases
    ----------
    Empty ``prompt`` still prints length 0 and preview ``''``. Does not raise.
    """
    length     = len(prompt)
    est_tokens = length // 4

    print(f"\n[SCENE {scene_id}] PROMPT ANALYSIS")
    print(f"  Character length : {length}")
    print(f"  Estimated tokens : {est_tokens}")

    if length < 80:
        print(f"  WARNING: Prompt too short ({length} chars)")
    if length > 500:
        print(f"  WARNING: Prompt may be truncated by model ({length} chars)")

    print(f"  Preview          : {prompt[:200]!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline_v2(
    scenes:     list[dict[str, Any]],
    characters: list[dict[str, str]],
    config:     dict[str, Any],
) -> dict[str, Any]:
    """
    Run memory-aware generation and CLIP selection for each scene in order.

    Parameters
    ----------
    scenes : list of dict[str, Any]
        Parsed scenes. Each dict must include ``"description"`` (used as
        ``scene_text`` for evaluation). ``scene_id`` defaults to loop index plus
        one. ``characters_present`` defaults to ``[]``.
    characters : list of dict[str, str]
        Parsed characters; entries with ``name`` populate a lookup of default
        descriptions when memory has none.
    config : dict[str, Any]
        Must include ``"generation"`` and ``config["evaluator"]["weights"]``.
        Optional: ``constraint_builder``, ``layout``, ``controlnet``,
        ``prompt_cache``, ``debug_output_dir`` (see module docstring).

    Returns
    -------
    dict[str, Any]
        ``"images"``: ordered best ``PIL.Image.Image`` per scene.
        ``"logs"``: per-scene dicts with ``scene_id``, ``prompt``,
        ``prompt_length``, ``best_index``, and evaluator ``scores``.

    Notes
    -----
    Builds prompts via ``build_constraints`` and either ``compress_prompt`` /
    ``build_prompt_from_constraints`` or, on failure when ControlNet is off,
    ``build_prompt``. Optional prompt cache reads/writes ``prompt.txt`` under
    ``debug_output_dir``. Generates with ``generate_images_controlled`` when
    ``config["controlnet"]["enabled"]`` is truthy, else ``generate_images``.
    Evaluates with ``score_candidates`` using ``scene["description"]`` (not the
    built prompt), a reference image from the first value in
    ``memory.get_reference_images`` if any, and ``previous_selected_image``.
    Updates ``MemoryManager`` after each scene. Reads optional
    ``config["debug_output_dir"]`` for filesystem debug output when provided by
    the caller.

    Raises
    ------
    ValueError
        For empty ``scenes``, missing config keys, ControlNet/layout/prompt
        validation failures, and other explicit ``raise`` paths in the body.

    Edge cases
    ----------
    When memory returns no descriptions, falls back to ``all_char_descriptions``
    for the current scene's names. ``reference_image`` is ``None`` if
    ``ref_images`` is empty; otherwise it is ``next(iter(ref_images.values()))``
    (dictionary iteration order). Prompt cache with ControlNet still requires a
    successful ``build_constraints`` call.
    """
    if not scenes:
        raise ValueError("scenes list must not be empty.")

    try:
        generation_config = config["generation"]
    except KeyError:
        raise ValueError("config is missing required key: 'generation'.")

    try:
        weights = config["evaluator"]["weights"]
    except KeyError:
        raise ValueError("config is missing required key: 'evaluator.weights'.")

    cb_config   = config.get("constraint_builder", {})
    style_prefix = cb_config.get("style_prefix", "")
    style_suffix = cb_config.get("style_suffix", "")

    n_candidates = int(generation_config.get("num_candidates", 3))

    all_char_descriptions: dict[str, str] = {
        c["name"]: c.get("description", "")
        for c in characters
        if c.get("name")
    }

    logger.info(
        "run_pipeline_v2 | scenes=%d  n_candidates=%d  characters=%d",
        len(scenes), n_candidates, len(all_char_descriptions),
    )

    memory = MemoryManager()

    selected_images: list[Image.Image] = []
    logs: list[dict[str, Any]] = []

    previous_selected_image: Optional[Image.Image] = None

    for scene_idx, scene in enumerate(scenes):
        scene_id = scene.get("scene_id", scene_idx + 1)
        scene_text = scene["description"]
        chars_present = scene.get("characters_present", [])

        char_descriptions = memory.get_character_descriptions(chars_present)
        ref_images = memory.get_reference_images(chars_present)

        if not char_descriptions:
            char_descriptions = {
                name: all_char_descriptions.get(name, "")
                for name in chars_present
            }

        cn_cfg = config.get("controlnet") or {}
        use_controlnet = bool(cn_cfg.get("enabled"))

        scene_dir = config.get("debug_output_dir", None)
        scene_path: Optional[Path] = None
        if scene_dir:
            scene_path = Path(scene_dir) / f"scene_{scene_id:02d}"
            scene_path.mkdir(parents=True, exist_ok=True)

        prompt_cache_enabled = bool((config.get("prompt_cache") or {}).get("enabled", False))
        prompt_file: Optional[Path] = None
        if scene_path is not None:
            prompt_file = scene_path / "prompt.txt"

        constraints: Optional[dict[str, Any]] = None
        cached_prompt_text: Optional[str] = None
        if prompt_cache_enabled and prompt_file is not None and prompt_file.is_file():
            try:
                cached_prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            except OSError:
                cached_prompt_text = None

        if (
            prompt_cache_enabled
            and prompt_file is not None
            and cached_prompt_text
        ):
            print(f"[SCENE {scene_id}] PROMPT CACHE HIT")
            prompt = cached_prompt_text
            if use_controlnet:
                try:
                    constraints = build_constraints(scene, char_descriptions, config)
                    if not str(prompt).strip():
                        raise ValueError("empty cached prompt")
                except Exception as exc:
                    raise ValueError(
                        "ControlNet enabled but constraint rebuild failed (prompt cache hit)."
                    ) from exc
            else:
                constraints = None
        else:
            try:
                constraints = build_constraints(scene, char_descriptions, config)
                if use_controlnet:
                    prompt = compress_prompt(constraints)
                    if len(prompt) > 200:
                        prompt = prompt[:200]
                    print("PROMPT COMPRESSED:", prompt)
                else:
                    prompt = build_prompt_from_constraints(
                        constraints,
                        include_layout=True,
                    )
                if not str(prompt).strip():
                    raise ValueError("empty prompt from constraints")
            except Exception as exc:
                if use_controlnet:
                    raise ValueError(
                        "ControlNet enabled but constraint path failed; layout is required."
                    ) from exc
                constraints = None
                prompt = build_prompt(
                    scene=scene,
                    character_descriptions=char_descriptions,
                    style_prefix=style_prefix,
                    style_suffix=style_suffix,
                )

            if prompt_cache_enabled and prompt_file is not None:
                try:
                    prompt_file.write_text(prompt, encoding="utf-8")
                except OSError as wexc:
                    logger.warning("prompt cache save failed: %s", wexc)
                print(f"[SCENE {scene_id}] PROMPT GENERATED")

        print("CONSTRAINTS:", constraints)
        print("PROMPT:", prompt)

        _analyze_prompt(prompt, scene_id)

        if use_controlnet:
            if not constraints or "layout" not in constraints:
                raise ValueError(
                    "ControlNet enabled but layout missing in constraints"
                )

            layout_for_control: dict[str, Any] = constraints["layout"]
            if not isinstance(layout_for_control, dict):
                raise ValueError("Invalid layout: not a dict")
            if "characters" not in layout_for_control:
                raise ValueError("Invalid layout: missing 'characters' key")
            if not isinstance(layout_for_control["characters"], list):
                raise ValueError("Invalid layout: 'characters' must be list")
            if not layout_for_control.get("characters"):
                raise ValueError("Invalid layout: no characters present")

            print(f"[SCENE {scene_id}] USING CONTROLNET")
            print(f"[SCENE {scene_id}] LAYOUT CHAR COUNT:", len(layout_for_control["characters"]))

            if scene_path is not None:
                debug_control_img = layout_to_control_image(
                    layout_for_control,
                    int(generation_config.get("width", 768)),
                    int(generation_config.get("height", 768)),
                )
                debug_control_img.save(scene_path / "control.png")

            merged_gen: dict[str, Any] = {**generation_config, "controlnet": cn_cfg}
            generated = generate_images_controlled(
                prompt=prompt,
                layout=layout_for_control,
                n=n_candidates,
                config=merged_gen,
            )
        else:
            generated = generate_images(
                prompt=prompt,
                n=n_candidates,
                config=generation_config,
            )
        candidate_images = [g.image for g in generated]

        if scene_dir and scene_path is not None:
            candidates_path = scene_path / "candidates"
            candidates_path.mkdir(parents=True, exist_ok=True)

            with open(scene_path / "prompt.txt", "w") as f:
                f.write(prompt)

            for i, img in enumerate(candidate_images):
                img.save(candidates_path / f"{i}.png")

        reference_image = None
        if ref_images:
            reference_image = next(iter(ref_images.values()))

        eval_result = score_candidates(
            images=candidate_images,
            scene_text=scene_text,
            reference_image=reference_image,
            previous_image=previous_selected_image,
            weights=weights,
        )

        best_index = eval_result["best_index"]
        best_image = eval_result["best_image"]

        if scene_path is not None:
            best_image.save(scene_path / "selected.png")

            with open(scene_path / "scores.json", "w") as f:
                json.dump(eval_result["scores"], f, indent=2)

        selected_images.append(best_image)
        logs.append({
            "scene_id": scene_id,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "best_index": best_index,
            "scores": eval_result["scores"],
        })

        memory.update_memory(
            characters=chars_present,
            image=best_image,
            descriptions={
                name: all_char_descriptions.get(name, "")
                for name in chars_present
            },
        )

        previous_selected_image = best_image

    logger.info("run_pipeline_v2 complete | selected %d images", len(selected_images))

    return {
        "images": selected_images,
        "logs":   logs,
    }