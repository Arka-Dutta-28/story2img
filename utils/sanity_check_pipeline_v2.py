import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import shutil
import yaml

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm import build_llm_client, StoryParser
from pipeline.story_pipeline_v2 import run_pipeline_v2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sanity_check_pipeline_v2")

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STORIES_DIR = PROJECT_ROOT / "data" / "test_stories"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "pipeline_v2_runs"
LOG_PATH    = PROJECT_ROOT / "logs" / "pipeline_v2_sanity_checks.json"

from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------
# Load stories
# --------------------------------------------------

def _load_stories():
    """Load every ``*.txt`` under ``data/test_stories``; story id is ``path.stem`` (filename without extension, any name)."""
    txt_files = sorted(STORIES_DIR.glob("*.txt"))
    if not txt_files:
        print("No stories found.")
        sys.exit(1)

    return [
        {
            "id": path.stem,
            "text": path.read_text(encoding="utf-8").strip()
        }
        for path in txt_files
    ]


# --------------------------------------------------
# Grid builder
# --------------------------------------------------

def _create_image_grid(images: list[Image.Image], save_path: Path):
    if not images:
        return

    n = len(images)

    # auto layout (square-ish)
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols

    w, h = images[0].size

    grid = Image.new("RGB", (cols * w, rows * h))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * w, r * h))

    grid.save(save_path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def run():
    print("\n=== PIPELINE V2 SANITY CHECK ===")

    # -------------------------------
    # Load config + models
    # -------------------------------
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    llm = build_llm_client(config["llm"])
    parser = StoryParser(llm)

    stories = _load_stories()

    print(f"Loaded {len(stories)} stories:")
    for s in stories:
        print(" -", s["id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # -------------------------------
    # Story loop
    # -------------------------------
    for idx, story in enumerate(stories, 1):
        log_file = PROJECT_ROOT / "logs" / "prompt_debug.log"
        if log_file.exists():
            log_file.unlink()
            print(f"Prompt debug log deleted")

        print(f"\n[{idx}/{len(stories)}] Running: {story['id']}")

        t0 = time.monotonic()

        try:
            parsed = parser.parse(story["text"])

            # -------------------------------
            # Create versioned output dir
            # -------------------------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            story_root = OUTPUT_DIR / story["id"] / timestamp
            debug_dir  = story_root / "debug"
            final_dir  = story_root / "final"

            debug_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)
            # Save config snapshot
            with open(story_root / "config.yaml", "w") as f:
                yaml.dump(config, f, indent=2)

            # pass debug path to pipeline
            config["debug_output_dir"] = str(debug_dir)

            # -------------------------------
            # Run pipeline
            # -------------------------------
            pipeline_result = run_pipeline_v2(
                parsed.scenes,
                parsed.characters,
                config
            )

            elapsed = time.monotonic() - t0

            images = pipeline_result["images"]

            # -------------------------------
            # Save FINAL outputs
            # -------------------------------
            for i, img in enumerate(images):
                img.save(final_dir / f"scene_{i}.png")

            # grid
            grid_path = final_dir / "grid.png"
            _create_image_grid(images, grid_path)

            # -------------------------------
            # Log success
            # -------------------------------
            results.append({
                "story_id": story["id"],
                "scene_count": len(parsed.scenes),
                "elapsed_s": round(elapsed, 2),
                "status": "success",
                "output_dir": str(story_root)
            })

            print(f"Done ({elapsed:.2f}s, {len(images)} scenes)")
            print(f"Final outputs → {final_dir}")
            print(f"Debug data   → {debug_dir}")

            src_log = PROJECT_ROOT / "logs" / "prompt_debug.log"
            dst_log = debug_dir / "prompt_debug.log"

            if src_log.exists():
                shutil.copy(src_log, dst_log)
                print(f"Prompt debug log copied to {dst_log}")
            else:
                print(f"Prompt debug log not found at {src_log}")

        except Exception as e:
            elapsed = time.monotonic() - t0

            results.append({
                "story_id": story["id"],
                "elapsed_s": round(elapsed, 2),
                "status": "failed",
                "error": str(e)
            })

            print(f"Failed: {e}")

    # -------------------------------
    # Save run log
    # -------------------------------
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    print(f"\nLog saved → {LOG_PATH}")


if __name__ == "__main__":
    run()
