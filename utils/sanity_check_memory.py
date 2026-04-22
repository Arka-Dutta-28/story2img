from memory.memory_manager import MemoryManager
from PIL import Image
import os

def run():
    memory = MemoryManager()

    # fake image
    img = Image.new("RGB", (64, 64), color="red")

    characters = ["knight", "dragon"]
    descriptions = {
        "knight": "a warrior in silver armor",
        "dragon": "a small flying creature"
    }

    # update
    memory.update_memory(characters, img, descriptions)

    # retrieve
    refs = memory.get_reference_images(["knight", "dragon"])
    descs = memory.get_character_descriptions(["knight", "dragon"])

    print("Reference images:", list(refs.keys()))
    print("Descriptions:", descs)

    os.makedirs("outputs/sanity_check_memory_test", exist_ok=True)

    # save reference images
    for name, img in refs.items():
        path = f"outputs/sanity_check_memory_test/{name}.png"
        img.save(path)
        print(f"Saved {name} → {path}")


if __name__ == "__main__":
    run()