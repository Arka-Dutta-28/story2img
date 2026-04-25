import yaml
import os
from PIL import Image

from evaluator.clip_evaluator import score_candidates


def load_test_images():
    """
    Load fixed PNG paths from ``outputs`` for evaluator exercises.

    Parameters
    ----------
    None

    Returns
    -------
    list[PIL.Image.Image]
        Opened images for ``test_image_0.png`` and ``test_image_1.png``.

    Raises
    ------
    FileNotFoundError
        If either expected path is missing.

    Notes
    -----
    Opens files without context managers; relies on PIL lazy loading semantics.

    Edge cases
    ----------
    Requires prior generator sanity run to populate ``outputs``.
    """
    img_paths = [
        "outputs/test_image_0.png",
        "outputs/test_image_1.png",
    ]

    images = []
    for p in img_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing test image: {p}")
        images.append(Image.open(p))

    return images


def run():
    """
    Run three ``score_candidates`` configurations emphasising different weights.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Loads images via ``load_test_images``, prints raw result dicts for alignment-only,
    identity-only, and temporal-only weightings.

    Edge cases
    ----------
    Propagates missing-file errors from ``load_test_images``.
    """
    print("=== Evaluator Sanity Check ===")

    images = load_test_images()

    scene_text = "a knight standing in a dark forest with dramatic lighting"

    weights = {
        "w_align": 1.0,
        "w_identity": 0.0,
        "w_temporal": 0.0
    }

    print("\n--- Test 1: Alignment Only ---")
    result = score_candidates(
        images=images,
        scene_text=scene_text,
        reference_image=None,
        previous_image=None,
        weights=weights
    )

    print(result)

    print("\n--- Test 2: Identity Consistency ---")
    result = score_candidates(
        images=images,
        scene_text=scene_text,
        reference_image=images[0],
        previous_image=None,
        weights={
            "w_align": 0.0,
            "w_identity": 1.0,
            "w_temporal": 0.0
        }
    )

    print(result)

    print("\n--- Test 3: Temporal Consistency ---")
    result = score_candidates(
        images=images,
        scene_text=scene_text,
        reference_image=None,
        previous_image=images[0],
        weights={
            "w_align": 0.0,
            "w_identity": 0.0,
            "w_temporal": 1.0
        }
    )

    print(result)

    print("\nSanity check complete.")


if __name__ == "__main__":
    run()
