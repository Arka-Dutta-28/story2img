"""
utils/sanity_check_parser.py
----------------------------
Exercise `llm.parser.StoryParser` on every `*.txt` story under
`data/test_stories`.

Writes a full JSON report to `logs/parser_sanity_checks_<model>.json` where
`<model>` is the configured LLM id (filesystem-safe). The report includes
characters, scenes, style, raw_json, story text, timings, and errors. Also
prints a short console summary.

Usage (from repo root):
    python -m utils.sanity_check_parser
    # or
    python utils/sanity_check_parser.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from llm import StoryParser, build_llm_client

load_dotenv()

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STORIES_DIR = PROJECT_ROOT / "data" / "test_stories"
LOGS_DIR = PROJECT_ROOT / "logs" / "parser"


@dataclass
class StoryRecord:
    """
    One discovered test story file and its UTF-8 text payload.

    Attributes
    ----------
    story_id : str
        Filename stem used as identifier.
    text : str
        Stripped file contents.
    source_path : pathlib.Path
        Absolute-compatible path to the ``.txt`` file.
    """
    story_id: str
    text: str
    source_path: Path


def _discover_stories() -> tuple[list[StoryRecord], Path]:
    """
    Enumerate non-empty text stories under the test data directory.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[list[StoryRecord], pathlib.Path]
        Records sorted by path and the base directory (possibly non-existent).

    Notes
    -----
    Reads each ``*.txt`` twice when non-empty check passes (once in filter,
    once in comprehension). Strips whitespace for ``text`` field.

    Edge cases
    ----------
    If ``STORIES_DIR`` is not a directory, ``paths`` is empty and records list is
    empty.
    """
    base = STORIES_DIR
    paths = sorted(base.glob("*.txt")) if base.is_dir() else []

    records = [
        StoryRecord(
            story_id=p.stem,
            text=p.read_text(encoding="utf-8").strip(),
            source_path=p,
        )
        for p in paths
        if p.read_text(encoding="utf-8").strip()
    ]
    return records, base


def _load_llm_config() -> dict[str, Any]:
    """
    Load the ``llm`` subsection from the repository ``config.yaml``.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, Any]
        The value at key ``"llm"`` in the parsed YAML.

    Raises
    ------
    FileNotFoundError
        If ``CONFIG_PATH`` is missing.

    Edge cases
    ----------
    Relies on YAML containing an ``llm`` key; ``KeyError`` would propagate if absent.
    """
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["llm"]


def _llm_meta(llm_config: dict[str, Any]) -> tuple[str, str]:
    """
    Extract provider name and model string for filenames and reports.

    Parameters
    ----------
    llm_config : dict[str, Any]
        The ``llm`` configuration block.

    Returns
    -------
    tuple[str, str]
        Lowercased ``provider`` and nested ``model`` string (empty if unknown).

    Notes
    -----
    Reads ``gemini``, ``groq``, or ``nvidia`` sub-dicts based on ``provider``.

    Edge cases
    ----------
    Unknown provider yields empty model id.
    """
    provider = str(llm_config.get("provider", "")).lower()
    if provider == "gemini":
        model = str(llm_config.get("gemini", {}).get("model", ""))
    elif provider == "groq":
        model = str(llm_config.get("groq", {}).get("model", ""))
    elif provider == "nvidia":
        model = str(llm_config.get("nvidia", {}).get("model", ""))
    else:
        model = ""
    return provider, model


def _log_path_for_model(model: str) -> Path:
    """
    Build a JSON log path under ``logs/parser`` with sanitised model stem.

    Parameters
    ----------
    model : str
        Raw model name possibly containing filesystem-hostile characters.

    Returns
    -------
    pathlib.Path
        ``LOGS_DIR / f"{stem}.json"`` after character replacement.

    Notes
    -----
    Replaces ``/\\:*?"<>|`` with underscores; blank model becomes ``unknown``.

    Edge cases
    ----------
    Does not collapse repeated underscores or enforce max path length.
    """
    stem = model.strip() or "unknown"
    for bad, repl in (("/", "_"), ("\\", "_"), (":", "_"), ("*", "_"), ("?", "_"), ('"', "_"), ("<", "_"), (">", "_"), ("|", "_")):
        stem = stem.replace(bad, repl)
    return LOGS_DIR / f"{stem}.json"


def _summarise_parsed(result: Any) -> dict[str, Any]:
    """
    Summarise a ``ParsedStory``-like object for logging.

    Parameters
    ----------
    result : Any
        Object with optional ``scenes``, ``characters``, ``style`` attributes.

    Returns
    -------
    dict[str, Any]
        Counts, scene id min/max among dict scenes, and style preview substring.

    Notes
    -----
    Treats missing attributes as empty. Scene ids read via ``dict.get`` on each
    scene when dict-shaped.

    Edge cases
    ----------
    Non-dict entries in ``scenes`` are ignored for min/max computation.
    """
    scenes = getattr(result, "scenes", []) or []
    scene_ids = [s.get("scene_id") for s in scenes if isinstance(s, dict)]
    return {
        "n_characters": len(getattr(result, "characters", []) or []),
        "n_scenes": len(scenes),
        "scene_id_min": min(scene_ids) if scene_ids else None,
        "scene_id_max": max(scene_ids) if scene_ids else None,
        "style_preview": (getattr(result, "style", "") or "")[:120],
    }


def run() -> int:
    """
    Execute parser smoke tests across ``data/test_stories`` and write JSON logs.

    Parameters
    ----------
    None

    Returns
    -------
    int
        ``0`` if every story parses, ``1`` if none found or any failure occurred.

    Notes
    -----
    Instantiates ``StoryParser`` with ``build_llm_client``, loops stories,
    captures timing and errors, prints summaries, writes detailed JSON to
    ``_log_path_for_model``.

    Edge cases
    ----------
    Catches broad ``Exception`` per story to record failure rows. Uses
    ``ascii()`` for style preview printing to avoid console encoding issues.
    """
    print("\n=== StoryParser sanity check ===\n")

    stories, stories_dir = _discover_stories()
    if not stories:
        print(f"No non-empty .txt stories under:\n  {STORIES_DIR}")
        return 1

    print(f"Stories directory: {stories_dir}")
    print(f"Stories loaded: {len(stories)}\n")

    llm_config = _load_llm_config()
    llm_provider, llm_model = _llm_meta(llm_config)
    log_path = _log_path_for_model(llm_model)
    llm = build_llm_client(llm_config)
    parser = StoryParser(llm=llm)

    rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    failures = 0
    scene_counts: list[int] = []

    for rec in stories:
        t0 = time.perf_counter()
        err: Optional[str] = None
        stats: Optional[dict[str, Any]] = None
        parsed: Any = None
        try:
            parsed = parser.parse(rec.text)
            stats = _summarise_parsed(parsed)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failures += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        elapsed_s = round(elapsed_ms / 1000.0, 3)
        rel_file = str(rec.source_path.relative_to(PROJECT_ROOT))

        row: dict[str, Any] = {
            "story_id": rec.story_id,
            "file": rel_file,
            "ok": err is None,
            "wall_ms": round(elapsed_ms, 1),
            "chars_in_story": len(rec.text),
        }
        if stats:
            row.update(stats)
        if err:
            row["error"] = err

        rows.append(row)

        if parsed is not None and err is None:
            scene_counts.append(len(parsed.scenes))
            detailed_results.append(
                {
                    "story_id": rec.story_id,
                    "title": rec.story_id,
                    "file": rel_file,
                    "story_text": rec.text,
                    "characters": parsed.characters,
                    "scenes": parsed.scenes,
                    "style": parsed.style,
                    "raw_json": parsed.raw_json,
                    "valid": True,
                    "warning": None,
                    "scene_count": len(parsed.scenes),
                    "elapsed_s": elapsed_s,
                    "wall_ms": round(elapsed_ms, 1),
                }
            )
        else:
            detailed_results.append(
                {
                    "story_id": rec.story_id,
                    "title": rec.story_id,
                    "file": rel_file,
                    "story_text": rec.text,
                    "characters": [],
                    "scenes": [],
                    "style": "",
                    "raw_json": {},
                    "valid": False,
                    "warning": None,
                    "scene_count": 0,
                    "elapsed_s": elapsed_s,
                    "wall_ms": round(elapsed_ms, 1),
                    "error": err,
                }
            )

        status = "OK" if err is None else "FAIL"
        print(f"[{status}] {rec.story_id} | {elapsed_ms:.1f} ms | story_chars={len(rec.text)}")
        if stats:
            print(
                f"         characters={stats['n_characters']} "
                f"scenes={stats['n_scenes']} "
                f"scene_id_range=({stats['scene_id_min']}, {stats['scene_id_max']})"
            )
            print(f"         style: {ascii(stats['style_preview'])}")
        if err:
            print(f"         {err}")
        print()

    passed = len(stories) - failures
    avg_scenes = round(sum(scene_counts) / len(scene_counts), 2) if scene_counts else 0.0

    log_payload: dict[str, Any] = {
        "run_timestamp": datetime.now().isoformat(),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "stories_dir": str(stories_dir.resolve()),
        "total_stories": len(stories),
        "passed": passed,
        "failed": failures,
        "avg_scenes_per_story": avg_scenes,
        "results": detailed_results,
        "summary_rows": rows,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)

    print("--- Summary (JSON) ---")
    print(json.dumps(rows, indent=2, ensure_ascii=True))
    print(f"\nDetailed report written to {log_path}")

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        return 1

    print("\nAll stories parsed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
