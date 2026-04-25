"""
pipeline/story_pipeline.py
--------------------------
Pipeline Orchestrator — Phase 5 (NO MEMORY).

Responsibilities
----------------
- Accept parsed scenes and config
- For each scene: generate N candidate images, evaluate them, select the best
- Track previous selected image for temporal consistency
- Return ordered list of selected images with per-scene logs

Public interface
----------------
    run_pipeline(scenes, config) -> dict

Constraints
-----------
- NO memory integration (reference_image is always None)
- Does NOT modify evaluator or generator
- Does NOT implement constraint builder, memory manager, or logger
"""

import logging
from typing import Any, Optional

from PIL import Image

from generator.image_generator import generate_images
from evaluator.clip_evaluator import score_candidates

logger = logging.getLogger(__name__)


def run_pipeline(
    scenes: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the no-memory image pipeline for each parsed scene in order.

    Parameters
    ----------
    scenes : list of dict[str, Any]
        Scene records from the parser. Each dict should provide ``scene_id``
        (optional; defaults to ``scene_idx + 1``) and ``description`` (required
        non-empty string) used as both generation prompt and evaluation text.
    config : dict[str, Any]
        Application configuration. Must contain ``"generation"`` (passed to
        ``generate_images``) and ``config["evaluator"]["weights"]`` (passed to
        ``score_candidates``). Candidate count uses
        ``int(config["generation"].get("num_candidates", 3))``.

    Returns
    -------
    dict[str, Any]
        Mapping with keys ``"images"`` (list of ``PIL.Image.Image``, one best
        image per scene in order) and ``"logs"`` (list of per-scene dicts with
        ``scene_id``, ``scene_text``, ``best_index``, and ``scores``).

    Notes
    -----
    For every scene, calls ``generate_images`` then ``score_candidates`` with
    ``reference_image=None`` and ``previous_image`` set to the prior scene's
    selected image (``None`` for the first scene). Appends the evaluator's
    ``best_image`` to ``selected_images`` and updates ``previous_selected_image``.

    Raises
    ------
    ValueError
        If ``scenes`` is empty, required config keys are missing, a scene has no
        non-empty ``description``, or ``generate_images`` / ``score_candidates``
        preconditions fail.

    Edge cases
    ----------
    Empty ``scenes`` raises immediately. Missing ``"generation"`` or
    ``"evaluator"``/``"weights"`` raises ``ValueError``. Downstream failures
    from generation or evaluation propagate as raised exceptions from those
    calls.
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

    n_candidates = int(generation_config.get("num_candidates", 3))

    logger.info(
        "run_pipeline | scenes=%d  n_candidates=%d",
        len(scenes), n_candidates,
    )

    selected_images: list[Image.Image] = []
    logs: list[dict[str, Any]] = []

    previous_selected_image: Optional[Image.Image] = None

    for scene_idx, scene in enumerate(scenes):
        scene_id   = scene.get("scene_id", scene_idx + 1)
        scene_text = scene.get("description", "")
        if not scene_text:
            raise ValueError(f"Scene {scene_id} missing description")

        logger.info(
            "Scene %d/%d  scene_id=%s  text=%r...",
            scene_idx + 1, len(scenes), scene_id, scene_text[:80],
        )

        generated = generate_images(
            prompt=scene_text,
            n=n_candidates,
            config=generation_config,
        )
        candidate_images = [g.image for g in generated]

        logger.info(
            "Scene %d — generated %d candidates", scene_idx + 1, len(candidate_images)
        )

        eval_result = score_candidates(
            images=candidate_images,
            scene_text=scene_text,
            reference_image=None,
            previous_image=previous_selected_image,
            weights=weights,
        )

        best_index = eval_result["best_index"]
        best_image = eval_result["best_image"]

        logger.info(
            "Scene %d — best_index=%d  final_score=%.4f",
            scene_idx + 1,
            best_index,
            eval_result["scores"][best_index]["final_score"],
        )

        selected_images.append(best_image)
        logs.append({
            "scene_id":   scene_id,
            "scene_text": scene_text,
            "best_index": best_index,
            "scores":     eval_result["scores"],
        })

        previous_selected_image = best_image

    logger.info("run_pipeline complete | selected %d images", len(selected_images))

    return {
        "images": selected_images,
        "logs":   logs,
    }