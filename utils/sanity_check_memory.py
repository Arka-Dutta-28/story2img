from memory.memory_manager import MemoryManager
from PIL import Image
import os

def run():
    """
    Smoke-test ``MemoryManager`` storage and retrieval with a synthetic image.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Creates a red ``64x64`` RGB image, calls ``update_memory`` for two names,
    prints retrieved keys and descriptions, writes PNGs under
    ``outputs/sanity_check_memory_test``.

    Edge cases
    ----------
    Always overwrites outputs in the target directory; not suitable as a hermetic
    unit test without cleanup.
    """
    memory = MemoryManager()

    img = Image.new("RGB", (64, 64), color="red")

    characters = ["knight", "dragon"]
    descriptions = {
        "knight": "a warrior in silver armor",
        "dragon": "a small flying creature"
    }

    memory.update_memory(characters, img, descriptions)

    refs = memory.get_reference_images(["knight", "dragon"])
    descs = memory.get_character_descriptions(["knight", "dragon"])

    print("Reference images:", list(refs.keys()))
    print("Descriptions:", descs)

    os.makedirs("outputs/sanity_check_memory_test", exist_ok=True)

    for name, img in refs.items():
        path = f"outputs/sanity_check_memory_test/{name}.png"
        img.save(path)
        print(f"Saved {name} → {path}")


if __name__ == "__main__":
    run()
