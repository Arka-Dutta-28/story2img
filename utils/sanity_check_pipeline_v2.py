import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
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


def _load_stories():
    """
    Load every ``*.txt`` story from ``data/test_stories`` as id/text pairs.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict[str, str]]
        Each dict has ``id`` (filename stem) and ``text`` (UTF-8 stripped).

    Notes
    -----
    Exits with code ``1`` when no matching files exist.

    Edge cases
    ----------
    Uses lexicographic sort of paths; empty files still produce entries with
    empty ``text``.
    """
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


def _create_image_grid(images: list[Image.Image], save_path: Path):
    """
    Paste ``images`` into a near-square grid and save to ``save_path``.

    Parameters
    ----------
    images : list[PIL.Image.Image]
        Non-empty list assumed to share identical ``size``.
    save_path : pathlib.Path
        Output path for the combined RGB image.

    Returns
    -------
    None

    Notes
    -----
    Computes ``cols = int(sqrt(n))`` and ``rows = ceil(n / cols)``, creates a
    blank RGB canvas, pastes each image at grid coordinates.

    Edge cases
    ----------
    Returns immediately when ``images`` is empty without creating a file.
    """
    if not images:
        return

    n = len(images)

    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols

    w, h = images[0].size

    grid = Image.new("RGB", (cols * w, rows * h))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * w, r * h))

    grid.save(save_path)


def run():
    """
    Run memory-aware ``run_pipeline_v2`` across all test stories with artifacts.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Deletes ``logs/prompt_debug.log`` when present before each story, injects
    ``config["debug_output_dir"]`` per run, saves final PNGs and grid, copies
    prompt debug log into the debug folder, appends JSON summary to
    ``LOG_PATH``.

    Edge cases
    ----------
    Catches ``Exception`` per story to record failures; mutates shared ``config``
    dict with debug directory strings each iteration.
    """
    print("\n=== PIPELINE V2 SANITY CHECK ===")

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

    for idx, story in enumerate(stories, 1):
        log_file = PROJECT_ROOT / "logs" / "prompt_debug.log"
        if log_file.exists():
            log_file.unlink()
            print(f"Prompt debug log deleted")

        print(f"\n[{idx}/{len(stories)}] Running: {story['id']}")

        t0 = time.monotonic()

        try:
            parsed = parser.parse(story["text"])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            story_root = OUTPUT_DIR / story["id"] / timestamp
            debug_dir  = story_root / "debug"
            final_dir  = story_root / "final"

            debug_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)
            with open(story_root / "config.yaml", "w") as f:
                yaml.dump(config, f, indent=2)

            config["debug_output_dir"] = str(debug_dir)

            pipeline_result = run_pipeline_v2(
                parsed.scenes,
                parsed.characters,
                config
            )

            elapsed = time.monotonic() - t0

            images = pipeline_result["images"]

            for i, img in enumerate(images):
                img.save(final_dir / f"scene_{i}.png")

            grid_path = final_dir / "grid.png"
            _create_image_grid(images, grid_path)

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

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    print(f"\nLog saved → {LOG_PATH}")


if __name__ == "__main__":
    run()
