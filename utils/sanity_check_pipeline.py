import yaml
import os

from llm.parser import StoryParser
from pipeline.story_pipeline import run_pipeline
from llm import build_llm_client


def run():
    print("=== Full Pipeline Sanity Check ===")

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Sample story
    story = """
    A knight walks through a dark forest at night. He carries a glowing sword.
    He reaches a river where mist rises. A shadow moves across the water.
    He prepares to fight as the moon shines above.
    """

    # Step 1: Parse
    llm_client = build_llm_client(config["llm"])
    parser = StoryParser(llm_client)
    parsed = parser.parse(story)

    print(f"\nParsed {len(parsed.scenes)} scenes")

    # Step 2: Run pipeline
    result = run_pipeline(parsed.scenes, config)

    images = result["images"]
    logs = result["logs"]

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save images
    for i, img in enumerate(images):
        path = f"outputs/pipeline_scene_{i}.png"
        img.save(path)
        print(f"Saved: {path}")

    print("\n=== Pipeline Logs ===")
    for log in logs:
        print({
            "scene_id": log["scene_id"],
            "best_index": log["best_index"],
            "top_score": log["scores"][log["best_index"]]["final_score"]
        })

    print("\nPipeline sanity check complete.")


if __name__ == "__main__":
    run()