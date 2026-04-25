import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from llm import build_llm_client, StoryParser
from pipeline.story_pipeline import run_pipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sanity_check_pipeline")

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STORIES_DIR = PROJECT_ROOT / "data" / "test_stories"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "pipeline_runs"
LOG_PATH    = PROJECT_ROOT / "logs" / "pipeline_sanity_checks.json"


def _load_stories():
    """
    Load all non-empty ``*.txt`` stories from ``data/test_stories``.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict[str, str]]
        Dicts with keys ``id`` (stem) and ``text`` (stripped contents).

    Notes
    -----
    Exits the process with code ``1`` if no text files are found.

    Edge cases
    ----------
    Calls ``sys.exit`` on empty discovery; does not validate story encoding
    beyond UTF-8 read.
    """
    txt_files = sorted(STORIES_DIR.glob("*.txt"))
    if not txt_files:
        print("No stories found.")
        sys.exit(1)

    stories = []
    for path in txt_files:
        stories.append({
            "id": path.stem,
            "text": path.read_text(encoding="utf-8").strip()
        })

    return stories


def run():
    """
    Batch-parse stories and run Phase-5 ``run_pipeline`` with JSON run logging.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Writes per-story PNGs under ``outputs/pipeline_runs/<story_id>/`` and
    aggregates pass/fail metadata to ``logs/pipeline_sanity_checks.json``.

    Edge cases
    ----------
    Catches ``Exception`` per story to continue the batch; prints emoji status
    markers to stdout on success/failure.
    """
    print("\n=== FULL PIPELINE SANITY CHECK ===")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    llm = build_llm_client(config["llm"])
    parser = StoryParser(llm)

    stories = _load_stories()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, story in enumerate(stories, 1):
        print(f"\n[{idx}/{len(stories)}] Running: {story['id']}")

        t0 = time.monotonic()

        try:
            parsed = parser.parse(story["text"])
            pipeline_result = run_pipeline(parsed.scenes, config)

            story_out_dir = OUTPUT_DIR / story["id"]
            story_out_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(pipeline_result["images"]):
                img_path = story_out_dir / f"scene_{i}.png"
                img.save(img_path)

            elapsed = time.monotonic() - t0

            results.append({
                "story_id": story["id"],
                "scene_count": len(parsed.scenes),
                "elapsed_s": round(elapsed, 2),
                "status": "success"
            })

            print(f"  ✅ Done ({elapsed:.2f}s, {len(parsed.scenes)} scenes)")

        except Exception as e:
            elapsed = time.monotonic() - t0

            results.append({
                "story_id": story["id"],
                "elapsed_s": round(elapsed, 2),
                "status": "failed",
                "error": str(e)
            })

            print(f"  ❌ Failed: {e}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_payload = {
        "timestamp": datetime.now().isoformat(),
        "total_stories": len(results),
        "results": results
    }

    with open(LOG_PATH, "w") as f:
        json.dump(log_payload, f, indent=2)

    print(f"\nLog saved → {LOG_PATH}")


if __name__ == "__main__":
    run()
