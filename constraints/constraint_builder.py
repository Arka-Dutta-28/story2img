"""
constraints/constraint_builder.py
---------------------------------

Constraint Builder — constructs the final prompt string for image generation.

## Responsibilities

* Combine scene description, character descriptions, and style config
  into a single clean prompt string ready for the image generator.

## Public interface

* build_constraints(scene, character_descriptions, config) -> dict — structured layer (+ optional layout)
* build_prompt_from_constraints(constraints, include_layout=True) -> str — collapse constraints to text
* compress_prompt(constraints) -> str — short prompt for ControlNet + SDXL
* build_prompt(scene, character_descriptions, style_prefix, style_suffix) -> str — legacy assembler

Optional layout planning via config["layout"] and Groq (see generate_layout_with_llm); always falls back on failure.
No negative prompts. No complex prompt engineering beyond layout JSON.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_CACHE_PATH = REPO_ROOT / "cache" / "layout_cache.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# --------------------------------------------------
# Logging setup
# --------------------------------------------------

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.FileHandler(LOG_DIR / "prompt_debug.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
file_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _truncate(text: str, max_words: int):
    return " ".join(text.split()[:max_words])


def _load_layout_cache() -> dict[str, Any]:
    try:
        LAYOUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LAYOUT_CACHE_PATH.is_file():
            with LAYOUT_CACHE_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("layout cache load failed: %s", exc)
    return {}


def _save_layout_cache(cache: dict[str, Any]) -> None:
    try:
        LAYOUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LAYOUT_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("layout cache save failed: %s", exc)


def _layout_cache_key(scene: dict, character_names: list[str]) -> str:
    desc = (scene.get("description") or "").strip()
    key_str = desc + str(sorted(character_names))
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


# Last Groq layout response text for mandatory debug logging (sequential pipeline only).
_LAST_GROQ_LAYOUT_RESPONSE_TEXT: Optional[str] = None


def _parse_strict_layout_json(text: str) -> Optional[dict[str, Any]]:
    """Parse LLM output into a dict; return None if not valid JSON object."""
    if not text or not text.strip():
        return None
    s = text.strip()
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def validate_layout(layout: Any, character_names: list[str]) -> bool:
    VALID_POS = {"left", "center", "right"}
    VALID_DEPTH = {"foreground", "midground", "background"}

    if not layout or not isinstance(layout, dict) or "characters" not in layout:
        return False

    chars = layout["characters"]
    if not isinstance(chars, list):
        return False

    if len(chars) != len(character_names):
        return False

    names = []
    for c in chars:
        if not isinstance(c, dict):
            return False
        n = c.get("name")
        if n is None:
            return False
        names.append(n)

    if set(names) != set(character_names):
        return False

    for c in chars:
        pos = c.get("position")
        depth = c.get("depth")
        if pos not in VALID_POS or depth not in VALID_DEPTH:
            return False

    return True


def fallback_layout(character_names: list[str]) -> dict[str, Any]:
    positions = ["center", "left", "right"]
    depths = ["foreground", "midground", "background"]

    layout = {"characters": []}

    for i, name in enumerate(character_names):
        pos = positions[i % len(positions)]
        depth = depths[i % len(depths)]

        layout["characters"].append({
            "name": name,
            "position": pos,
            "depth": depth
        })

    return layout


def generate_layout_with_llm(constraints: dict, config: dict) -> Optional[dict[str, Any]]:
    """
    Calls Groq LLM to generate layout.
    Returns parsed layout dict OR None if failure.
    Never raises.
    """
    global _LAST_GROQ_LAYOUT_RESPONSE_TEXT
    _LAST_GROQ_LAYOUT_RESPONSE_TEXT = None

    layout_cfg = (config or {}).get("layout") or {}
    if not layout_cfg.get("use_llm_layout"):
        return None

    try:
        from llm.groq_client import GroqClient
    except Exception:
        return None

    characters = [c["name"] for c in constraints.get("characters", [])]
    scene_description = " ".join(constraints.get("actions", []))

    system_prompt = """
You are a deterministic scene layout planner.

You must assign layout based on importance and interaction.

---

## PRIORITY RULES (STRICT ORDER)

1. VISUAL DOMINANCE (HIGHEST PRIORITY)

* Large, powerful, or central characters dominate layout
* Example: lion > mouse, wolf > sheep

→ Dominant character MUST be:
center foreground OR center midground

---

2. INTERACTION (SECOND PRIORITY)

* If one character acts on another:

  acting character → foreground
  receiving character → midground

BUT:

* DO NOT override dominance rule
* If dominant character is involved, it stays central

---

3. SECONDARY CHARACTERS

* Place on left/right or midground/background

---

4. GENERAL RULES

* Use ONLY given characters
* Assign EXACTLY one position and depth per character
* Allowed positions: left, center, right
* Allowed depth: foreground, midground, background
* Avoid symmetric layouts
* DO NOT hallucinate
* Output STRICT JSON only

---

5. SPATIAL UNIQUENESS (MANDATORY)

* Each character MUST have a unique (position, depth) pair

* DO NOT assign the same combination to multiple characters

* If conflict occurs:
  → keep dominant character in center
  → move others to left/right or different depth

---

## CRITICAL CONSTRAINT

Dominant character MUST NOT be placed in background unless explicitly inactive.

---
""".strip()

    user_prompt = f"""
Characters:
{characters}

Scene description:
{scene_description}

---

IMPORTANT:
The scene description contains the interaction and must be used to decide importance.

---

Output format:
{{
"characters": [
{{"name": "...", "position": "...", "depth": "..."}}
]
}}
""".strip()

    try:
        client = GroqClient(
            model=layout_cfg["model"],
            temperature=float(layout_cfg.get("temperature", 0.0)),
            max_tokens=int(layout_cfg.get("max_tokens", 512)),
        )

        response = client.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        raw = response.text if response and response.text is not None else ""
        _LAST_GROQ_LAYOUT_RESPONSE_TEXT = raw

        parsed = _parse_strict_layout_json(raw)
        if not isinstance(parsed, dict):
            return None

        return parsed

    except Exception:
        return None


# --------------------------------------------------
# Structured constraints → prompt (preferred path)
# --------------------------------------------------


def compress_prompt(constraints: dict) -> str:
    """
    Short prompt for SDXL when ControlNet carries spatial layout: names, main action, setting only.
    """
    parts: list[str] = []

    for char in constraints.get("characters", []):
        if isinstance(char, dict) and char.get("name"):
            parts.append(str(char["name"]))

    actions = constraints.get("actions") or []
    if isinstance(actions, list) and actions:
        parts.append(str(actions[0]))

    if constraints.get("setting"):
        parts.append(str(constraints["setting"]))

    return ", ".join(parts)


def build_constraints(
    scene: dict,
    character_descriptions: dict[str, str],
    config: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Build a structured constraint dict from a parsed scene.

    Parameters
    ----------
    scene : dict
        Scene from the parser (expects characters_present, optional description, etc.).
    character_descriptions : dict
        name → description for grounding.
    config : dict, optional
        Master config; uses config["layout"] for optional LLM layout (defaults if missing).

    Returns
    -------
    dict
        Structured constraints before collapsing to a single prompt string.
    """
    cfg = config or {}
    layout_cfg = cfg.get("layout") or {}

    constraints: dict = {
        "characters": [],
        "actions": [],
        "setting": None,
        "time": None,
        "mood": None,
    }

    for name in scene.get("characters_present", []):
        constraints["characters"].append({
            "name": name,
            "description": character_descriptions.get(name, ""),
        })

    if "description" in scene:
        constraints["actions"].append(scene["description"])

    constraints["setting"] = scene.get("setting")
    constraints["time"] = scene.get("time_of_day")
    constraints["mood"] = scene.get("mood")

    character_names = [c["name"] for c in constraints["characters"]]

    llm_raw: Optional[str] = None
    from_cache = False

    if layout_cfg.get("use_llm_layout"):
        layout_cache = _load_layout_cache()
        cache_key = _layout_cache_key(scene, character_names)
        layout: dict[str, Any]

        if cache_key in layout_cache:
            cached_layout = layout_cache[cache_key]
            if isinstance(cached_layout, dict) and validate_layout(cached_layout, character_names):
                print("CACHE HIT: layout")
                layout = cached_layout
                from_cache = True
            else:
                layout_cache.pop(cache_key, None)
                _save_layout_cache(layout_cache)
                layout_candidate = generate_layout_with_llm(constraints, cfg)
                llm_raw = _LAST_GROQ_LAYOUT_RESPONSE_TEXT
                if layout_candidate is not None and validate_layout(layout_candidate, character_names):
                    layout = layout_candidate
                    layout_cache[cache_key] = layout
                    _save_layout_cache(layout_cache)
                else:
                    logger.warning(
                        "layout planner: invalid JSON, validation failed, or LLM error — using fallback_layout"
                    )
                    layout = fallback_layout(character_names)
        else:
            layout_candidate = generate_layout_with_llm(constraints, cfg)
            llm_raw = _LAST_GROQ_LAYOUT_RESPONSE_TEXT
            if layout_candidate is not None and validate_layout(layout_candidate, character_names):
                layout = layout_candidate
                layout_cache[cache_key] = layout
                _save_layout_cache(layout_cache)
            else:
                logger.warning(
                    "layout planner: invalid JSON, validation failed, or LLM error — using fallback_layout"
                )
                layout = fallback_layout(character_names)
    else:
        layout = fallback_layout(character_names)

    if layout_cfg.get("use_llm_layout"):
        if from_cache:
            print("LLM RAW:", "(skipped; layout from cache)")
        else:
            print("LLM RAW:", llm_raw if llm_raw is not None else "(no response)")
    else:
        print("LLM RAW:", "(skipped; use_llm_layout is false)")

    print("FINAL LAYOUT:", layout)

    constraints["layout"] = layout

    return constraints


def build_prompt_from_constraints(
    constraints: dict,
    include_layout: bool = True,
) -> str:
    """
    Convert structured constraints into a single prompt string.

    Parameters
    ----------
    constraints : dict
        Output of build_constraints().
    include_layout : bool
        If False, omit spatial layout phrases (e.g. when ControlNet carries layout).

    Returns
    -------
    str
        Comma-joined prompt text for the image generator.
    """
    parts: list[str] = []

    for char in constraints["characters"]:
        parts.append(f'{char["name"]}: {char["description"]}')

    parts.extend(constraints["actions"])

    if constraints["setting"]:
        parts.append(f'setting: {constraints["setting"]}')
    if constraints["time"]:
        parts.append(f'time: {constraints["time"]}')
    if constraints["mood"]:
        parts.append(f'mood: {constraints["mood"]}')

    if include_layout:
        layout = constraints.get("layout") or {}
        layout_chars = layout.get("characters") if isinstance(layout, dict) else None
        if isinstance(layout_chars, list):
            for c in layout_chars:
                if isinstance(c, dict) and "name" in c and "position" in c and "depth" in c:
                    parts.insert(0, f'{c["name"]} at {c["position"]} {c["depth"]}')

    return ", ".join(parts)


# --------------------------------------------------
# Legacy prompt assembly (fallback)
# --------------------------------------------------


def build_prompt(
    scene: dict,
    character_descriptions: dict[str, str],
    style_prefix: str,
    style_suffix: str,
):
    MAX_CHARS = 320

    description = scene.get("description", "").strip()
    if not description:
        raise ValueError("Scene missing description")

    setting = scene.get("setting", "").strip()
    time = scene.get("time_of_day", "").strip()
    mood = scene.get("mood", "").strip()

    # --------------------------------------------------
    # 1. Build CHARACTER-GROUNDED ACTION
    # --------------------------------------------------
    grounded_sentences = []

    for name, desc in character_descriptions.items():
        desc = desc.strip()
        if desc:
            grounded_sentences.append(f"{name} ({desc})")

    char_context = ", ".join(grounded_sentences)

    # Merge with action
    if char_context:
        main_block = f"{char_context}. {description}"
    else:
        main_block = description

    # --------------------------------------------------
    # 2. KEY OBJECTS (implicitly inside description)
    # --------------------------------------------------
    # (Already handled via description — keep simple)

    # --------------------------------------------------
    # 3. SETTING (LOW PRIORITY)
    # --------------------------------------------------
    setting_parts = []

    if setting:
        setting_parts.append(setting)
    if time:
        setting_parts.append(time)

    setting_block = ", ".join(setting_parts)

    # --------------------------------------------------
    # 4. MOOD (LOWEST PRIORITY)
    # --------------------------------------------------
    mood_block = mood if mood else ""

    # --------------------------------------------------
    # 5. Assemble FULL prompt (priority order)
    # --------------------------------------------------
    parts = [main_block]

    if setting_block:
        parts.append(setting_block)

    if mood_block:
        parts.append(mood_block)

    full_prompt = ". ".join(parts)
    whole_prompt = full_prompt

    # --------------------------------------------------
    # 6. SMART TRIMMING (NO IDENTITY LOSS)
    # --------------------------------------------------
    if len(full_prompt) > MAX_CHARS:
        # Step 1: remove mood
        if mood_block:
            parts = [main_block]
            if setting_block:
                parts.append(setting_block)
            full_prompt = ". ".join(parts)

    if len(full_prompt) > MAX_CHARS:
        # Step 2: shorten setting (NOT characters)
        if setting_block:
            setting_short = setting[:50]  # safe trim
            parts = [main_block, setting_short]
            full_prompt = ". ".join(parts)

    if len(full_prompt) > MAX_CHARS:
        # Step 3: last resort → trim ONLY tail
        full_prompt = full_prompt[:MAX_CHARS]

    # --------------------------------------------------
    # 7. Logging 
    # --------------------------------------------------
    logger.info(
        "\n[Prompt Debug - NEW]\n"
        f"[SCENE ID]: {scene.get('scene_id', 'unknown')}\n\n"
        f"WHOLE ({len(whole_prompt)} chars): {whole_prompt}\n\n"
        f"FINAL ({len(full_prompt)} chars): {full_prompt}\n"
    )

    return full_prompt