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
    Run the full pipeline (no memory) over a list of parsed scenes.

    For each scene:
      1. Extract scene_text from scene["description"]
      2. Generate N candidate images via generate_images()
      3. Score candidates via score_candidates()
         - reference_image is always None (no memory in this phase)
         - previous_image is the previously selected image (or None for scene 1)
      4. Select best image and append to results
      5. Update previous_selected_image for next scene

    Parameters
    ----------
    scenes : List of scene dicts from the parser. Each must contain at minimum:
               "scene_id"    (int)
               "description" (str) — used as scene_text for generation + evaluation
    config : Master config dict (loaded from config.yaml). Expected sub-keys:
               config["generation"]         — passed to generate_images()
               config["evaluator"]["weights"] — passed to score_candidates()

    Returns
    -------
    {
        "images": [PIL.Image, ...],   # one selected image per scene, in order
        "logs":   [                   # one entry per scene
            {
                "scene_id":    int,
                "scene_text":  str,
                "best_index":  int,
                "scores":      list[dict],   # full score breakdown per candidate
            },
            ...
        ]
    }

    Raises
    ------
    ValueError  if scenes is empty or required config keys are missing.
    RuntimeError if generation or evaluation fails for any scene.
    """
    if not scenes:
        raise ValueError("scenes list must not be empty.")

    # -- Extract sub-configs -------------------------------------------------
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

    # -- Scene loop ----------------------------------------------------------
    for scene_idx, scene in enumerate(scenes):
        scene_id   = scene.get("scene_id", scene_idx + 1)
        scene_text = scene.get("description", "")
        if not scene_text:
            raise ValueError(f"Scene {scene_id} missing description")

        logger.info(
            "Scene %d/%d  scene_id=%s  text=%r...",
            scene_idx + 1, len(scenes), scene_id, scene_text[:80],
        )

        # Step 1: Generate N candidate images --------------------------------
        generated = generate_images(
            prompt=scene_text,
            n=n_candidates,
            config=generation_config,
        )
        candidate_images = [g.image for g in generated]

        logger.info(
            "Scene %d — generated %d candidates", scene_idx + 1, len(candidate_images)
        )

        # Step 2: Evaluate candidates ----------------------------------------
        #   reference_image = None  (no memory in Phase 5)
        #   previous_image  = last selected image (or None for first scene)
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

        # Step 3: Collect results --------------------------------------------
        selected_images.append(best_image)
        logs.append({
            "scene_id":   scene_id,
            "scene_text": scene_text,
            "best_index": best_index,
            "scores":     eval_result["scores"],
        })

        # Step 4: Update previous image for next scene -----------------------
        previous_selected_image = best_image

    logger.info("run_pipeline complete | selected %d images", len(selected_images))

    return {
        "images": selected_images,
        "logs":   logs,
    }