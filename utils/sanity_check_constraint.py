"""
utils/sanity_check_constraint.py
--------------------------------
Build constraint dicts (and prompt-from-constraints strings) for every scene
using cached parser output — **no Gemini call**. Scenes and characters are read
from `logs/parser/gemini-2.5-flash.json` by default. Layout planning uses
`config.yaml` (Groq / `layout` section) when `use_llm_layout` is true.

Usage (from repo root):

    python utils/sanity_check_constraint.py
    python utils/sanity_check_constraint.py --parser-log logs/parser/gemini-2.5-flash.json
    python utils/sanity_check_constraint.py --no-layout-llm
    python utils/sanity_check_constraint.py --verbose

By default, per-scene print() output from the constraint builder is suppressed
(console stays clean); use `--verbose` to show it. Full data is always in the JSON.

Writes `logs/constraint_sanity_<timestamp>.json` and prints a short summary.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        """
        No-op stub when ``python-dotenv`` is not installed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Defined only in the ``ImportError`` fallback branch.

        Edge cases
        ----------
        Matches the callable shape expected by downstream ``load_dotenv()`` use.
        """
        return

from constraints import constraint_builder as cb
from constraints.constraint_builder import (
    build_constraints,
    build_prompt_from_constraints,
)

load_dotenv()

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_PARSER_LOG = PROJECT_ROOT / "logs" / "parser" / "gemini-2.5-flash.json"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "constraint"


def _char_descriptions_for_scene(
    characters: list[dict[str, Any]],
    chars_present: list[str],
) -> dict[str, str]:
    """
    Map scene character names to parser-provided descriptions.

    Parameters
    ----------
    characters : list[dict[str, Any]]
        Character records with ``name`` and optional ``description``.
    chars_present : list[str]
        Names appearing in the current scene.

    Returns
    -------
    dict[str, str]
        ``name -> description`` for each name in ``chars_present`` (default ``""``).

    Notes
    -----
    Builds an intermediate dict from ``characters`` then projects onto
    ``chars_present``.

    Edge cases
    ----------
    Names absent from ``characters`` yield empty string values.
    """
    by_name = {
        c["name"]: c.get("description", "")
        for c in characters
        if c.get("name")
    }
    return {name: by_name.get(name, "") for name in chars_present}


def _load_config(path: Path, no_layout_llm: bool) -> dict[str, Any]:
    """
    Load ``config.yaml`` and optionally disable LLM layout planning.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path to YAML config.
    no_layout_llm : bool
        When ``True``, copies the config and forces ``layout.use_llm_layout`` false.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping, possibly shallow-copied and mutated for layout.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.

    Edge cases
    ----------
    Shallow copy means nested dicts are shared unless replaced (layout dict is
    replaced with a new dict).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if no_layout_llm:
        cfg = dict(cfg)
        layout = dict(cfg.get("layout") or {})
        layout["use_llm_layout"] = False
        cfg["layout"] = layout
    return cfg


def _build_one_scene(
    scene: dict[str, Any],
    characters: list[dict[str, Any]],
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """
    Build constraints and flattened prompt for one scene dict.

    Parameters
    ----------
    scene : dict[str, Any]
        Single scene record from parser output.
    characters : list[dict[str, Any]]
        Full story character list for description lookup.
    config : dict[str, Any]
        Master configuration passed to ``build_constraints``.
    verbose : bool
        When ``False``, redirects stdout during builder calls.

    Returns
    -------
    dict[str, Any]
        Row with scene metadata, ``constraints``, ``prompt_from_constraints``,
        captured Groq raw text, and optional captured stdout.

    Notes
    -----
    Reads ``cb._LAST_GROQ_LAYOUT_RESPONSE_TEXT`` after building. Uses
    ``contextlib.redirect_stdout`` to a ``StringIO`` when not verbose.

    Edge cases
    ----------
    Propagates exceptions from builder functions to the caller.
    """
    chars_present = scene.get("characters_present", [])
    char_desc = _char_descriptions_for_scene(characters, chars_present)

    sink = io.StringIO()
    ctx = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(sink)
    )

    with ctx:
        constraints = build_constraints(scene, char_desc, config)
        prompt = build_prompt_from_constraints(constraints)

    groq_raw = cb._LAST_GROQ_LAYOUT_RESPONSE_TEXT
    captured_prints = "" if verbose else sink.getvalue()

    return {
        "scene_id": scene.get("scene_id"),
        "characters_present": chars_present,
        "constraints": constraints,
        "prompt_from_constraints": prompt,
        "layout_groq_raw": groq_raw,
        "captured_stdout": captured_prints,
    }


def run(argv: list[str] | None = None) -> int:
    """
    CLI entry: rebuild constraints from a cached parser JSON artifact.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument vector; ``None`` uses ``sys.argv`` via ``argparse``.

    Returns
    -------
    int
        Process exit code ``0`` on success, ``1`` on input errors.

    Notes
    -----
    Parses CLI flags, loads parser bundle, iterates stories and scenes calling
    ``_build_one_scene``, writes aggregated JSON under ``logs/constraint`` by
    default.

    Edge cases
    ----------
    Invalid ``results`` shape prints to stderr and returns ``1``. Per-scene
    errors are captured inside scene rows without aborting the full run.
    """
    parser = argparse.ArgumentParser(
        description="Build constraints from cached Gemini parser JSON (no Gemini API)."
    )
    parser.add_argument(
        "--parser-log",
        type=Path,
        default=DEFAULT_PARSER_LOG,
        help=f"Path to parser JSON (default: {DEFAULT_PARSER_LOG})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Master config.yaml",
    )
    parser.add_argument(
        "--no-layout-llm",
        action="store_true",
        help="Force layout fallback only (no Groq layout calls).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-scene print() from constraint_builder (default: suppress).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: logs/constraint_sanity_<timestamp>.json)",
    )

    args = parser.parse_args(argv)

    parser_log: Path = args.parser_log
    if not parser_log.is_file():
        print(f"ERROR: Parser log not found: {parser_log}", file=sys.stderr)
        return 1

    config = _load_config(args.config, no_layout_llm=args.no_layout_llm)

    with parser_log.open(encoding="utf-8") as f:
        bundle = json.load(f)

    results_in = bundle.get("results")
    if not isinstance(results_in, list):
        print("ERROR: Expected top-level 'results' list in parser JSON.", file=sys.stderr)
        return 1

    run_ts = datetime.now().isoformat()
    stories_out: list[dict[str, Any]] = []

    for story in results_in:
        story_id = story.get("story_id", "?")
        characters = story.get("characters") or []
        scenes = story.get("scenes") or []

        if not isinstance(characters, list) or not isinstance(scenes, list):
            stories_out.append({
                "story_id": story_id,
                "error": "invalid characters or scenes shape",
                "scenes": [],
            })
            continue

        scene_rows: list[dict[str, Any]] = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            try:
                row = _build_one_scene(scene, characters, config, verbose=args.verbose)
                scene_rows.append(row)
            except Exception as exc:
                scene_rows.append({
                    "scene_id": scene.get("scene_id"),
                    "error": str(exc),
                })

        stories_out.append({
            "story_id": story_id,
            "title": story.get("title"),
            "file": story.get("file"),
            "scene_count": len(scene_rows),
            "scenes": scene_rows,
        })

        ok = sum(1 for s in scene_rows if "error" not in s)
        print(f"[{story_id}] scenes={len(scene_rows)} built_ok={ok}")

    out_path = args.output
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{stamp}.json"

    payload: dict[str, Any] = {
        "run_timestamp": run_ts,
        "source_parser_log": str(parser_log.resolve()),
        "config_path": str(args.config.resolve()),
        "layout_llm_enabled": bool((config.get("layout") or {}).get("use_llm_layout")),
        "stories": stories_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
