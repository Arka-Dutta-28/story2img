import yaml
import os
from generator import generate_images

def run():
    """
    Generate two SDXL candidates from ``config.yaml`` and save under ``outputs``.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Reads ``generation`` block, calls ``generate_images`` with a fixed forest
    prompt, saves each ``GeneratedImage.image`` as PNG.

    Edge cases
    ----------
    Fails if ``config.yaml`` or ``generation`` key is missing; overwrites
    ``outputs/test_image_*.png`` paths.
    """
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    gen_config = config["generation"]

    prompt = "a cinematic shot of a knight standing in a dark forest, dramatic lighting"

    images = generate_images(prompt, n=2, config=gen_config)
    os.makedirs("outputs", exist_ok=True)

    for i, img in enumerate(images):
        path = f"outputs/test_image_{i}.png"
        img.image.save(path)
        print(f"Saved: {path} | seed={img.seed}")

if __name__ == "__main__":
    run()
