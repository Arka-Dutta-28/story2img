import yaml
import os
from PIL import Image

from evaluator.clip_evaluator import score_candidates


def load_test_images():
    """
    Load 2–3 sample images from outputs folder.
    Assumes generator already saved some images.
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
        reference_image=images[0],   # same image → should score high
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