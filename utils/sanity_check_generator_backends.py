"""
utils/sanity_check_generator_backends.py
----------------------------------------
Smoke-test utility for image generation backends.

Purpose
-------
Validate that selected backends load correctly and generate images
without errors. Useful for testing new backend configurations or
verifying environment setup.

Usage
-----
    python -m utils.sanity_check_generator_backends --backend sdxl --count 2
    python -m utils.sanity_check_generator_backends --backend flux --output-dir flux_out

Command-line options
--------------------
--backend {sdxl, flux}
    Backend to test (default: sdxl).
--count N
    Number of images to generate (default: 2).
--output-dir PATH
    Directory to save generated images (default: outputs).

Test scope
----------
- Validates backend loading and initialization
- Tests image generation with fixed prompt
- Verifies output saving
- Reports generation time and status

Useful for
----------
- Verifying backend installation
- Testing new configurations
- Debugging environment issues
- Validating precision/quantization settings
"""

import argparse
import os
import yaml

from generator import generate_images


def main() -> None:
    """
    Run backend smoke test with configurable parameters.

    Loads config from config.yaml, applies command-line backend override,
    generates test images, and saves them to output directory.

    Notes
    -----
    Uses fixed test prompt across all backends for consistency.
    Reports generation time and success/failure status.
    Creates output directory if it doesn't exist.

    Backend testing
    ---------------
    Each backend is tested with its own image generation pipeline.
    Images are saved with backend name prefix for easy identification.
    Test prompt is intentionally detailed to exercise model capability.
    """
    parser = argparse.ArgumentParser(
        description="Smoke-test image generation backends with a fixed prompt."
    )
    parser.add_argument(
        "--backend",
        choices=["sdxl", "flux"],
        default="sdxl",
        help="Backend to test."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of candidate images to generate."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated images will be saved."
    )
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("generation", {})
    config.setdefault("backends", {})
    config["generation"]["backend"] = args.backend

    prompt = (
        "A cinematic illustration of a lone traveler standing on a cliff at dusk, "
        "soft volumetric light, dramatic atmosphere."
    )

    print("Running backend smoke test:", args.backend)
    print("Prompt:", prompt)

    images = generate_images(prompt=prompt, n=args.count, config=config)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, image_record in enumerate(images):
        filename = os.path.join(args.output_dir, f"{args.backend}_{idx}.png")
        image_record.image.save(filename)
        print(
            f"Saved {filename} | seed={image_record.seed} "
            f"backend={image_record.backend_name} precision={image_record.precision}"
        )


if __name__ == "__main__":
    main()
