"""
constraints/constraint_builder.py
---------------------------------

Constraint Builder — constructs the final prompt string for image generation.

## Responsibilities

* Combine scene description, character descriptions, and style config
  into a single clean prompt string ready for the image generator.

## Public interface

* build_constraints(scene, character_descriptions, config) -> dict — structured layer (+ optional layout)
* build_prompt_from_constraints(constraints, include_layout=True) -> str — minimal text (names, action, setting; optional layout)
* compress_prompt(constraints) -> str — minimal text for ControlNet + SDXL (no layout phrases)
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
    """
    Return the first ``max_words`` whitespace-separated tokens of ``text``.

    Parameters
    ----------
    text : str
        Source string split with ``str.split()``.
    max_words : int
        Upper bound on token count (slice length).

    Returns
    -------
    str
        Space-joined prefix of tokens; empty string if ``text`` has no words.

    Notes
    -----
    Does not normalise whitespace beyond ``split`` behaviour.

    Edge cases
    ----------
    ``max_words`` larger than token count returns all tokens.
    """
    return " ".join(text.split()[:max_words])


def _load_layout_cache() -> dict[str, Any]:
    """
    Read the layout JSON cache from disk or return an empty dict.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, Any]
        Parsed object when root is a dict; otherwise ``{}``.

    Notes
    -----
    Uses ``LAYOUT_CACHE_PATH``; ensures parent directory exists on read attempt.

    Edge cases
    ----------
    Logs warning and returns ``{}`` on IO/JSON errors.
    """
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
    """
    Write ``cache`` to ``LAYOUT_CACHE_PATH`` as formatted JSON.

    Parameters
    ----------
    cache : dict[str, Any]
        Serializable mapping to persist.

    Returns
    -------
    None

    Notes
    -----
    Creates parent directories as needed.

    Edge cases
    ----------
    Logs warning on failure without raising.
    """
    try:
        LAYOUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LAYOUT_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("layout cache save failed: %s", exc)


def _layout_cache_key(scene: dict, character_names: list[str]) -> str:
    """
    Build a deterministic SHA-256 key from scene description and character names.

    Parameters
    ----------
    scene : dict
        Uses ``scene.get("description")`` stripped or empty string.
    character_names : list[str]
        Sorted when stringified for stability.

    Returns
    -------
    str
        Hex digest of UTF-8 encoded concatenation.

    Notes
    -----
    Concatenates description text with ``str(sorted(character_names))``.

    Edge cases
    ----------
    Same description with different name orderings still collide only if sorted
    lists match.
    """
    desc = (scene.get("description") or "").strip()
    key_str = desc + str(sorted(character_names))
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


# Last Groq layout response text for mandatory debug logging (sequential pipeline only).
_LAST_GROQ_LAYOUT_RESPONSE_TEXT: Optional[str] = None


def _parse_strict_layout_json(text: str) -> Optional[dict[str, Any]]:
    """
    Parse model text into a JSON object dict when possible.

    Parameters
    ----------
    text : str
        Raw LLM output possibly containing extra prose.

    Returns
    -------
    dict[str, Any] or None
        Parsed dict if root JSON is an object; else ``None``.

    Notes
    -----
    First tries ``json.loads`` on stripped text; on failure searches for a
    ``{...}`` substring and parses that.

    Edge cases
    ----------
    Returns ``None`` for empty input, non-object JSON, or unparseable braces.
    """
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
    """
    Check structural and naming constraints for a layout dict.

    Parameters
    ----------
    layout : Any
        Candidate layout object.
    character_names : list[str]
        Expected multiset of character names for the scene.

    Returns
    -------
    bool
        ``True`` if ``layout`` matches schema and names; else ``False``.

    Notes
    -----
    Requires dict with ``characters`` list length equal to
    ``len(character_names)``, exact set equality of names, each entry dict with
    allowed ``position`` and ``depth`` values.

    Edge cases
    ----------
    Returns ``False`` on type mismatches or duplicate handling is implicit in set
    comparison of names.
    """
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
    """
    Construct a deterministic default layout cycling positions and depths.

    Parameters
    ----------
    character_names : list[str]
        Names placed in input order.

    Returns
    -------
    dict[str, Any]
        ``{"characters": [{"name", "position", "depth"}, ...]}``.

    Notes
    -----
    Cycles ``positions`` and ``depths`` by index modulo list length.

    Edge cases
    ----------
    Empty input yields ``{"characters": []}``.
    """
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
    Optionally query Groq for a JSON layout plan.

    Parameters
    ----------
    constraints : dict
        Structured constraints; reads ``characters`` and ``actions``.
    config : dict
        Master config; uses ``config.get("layout")`` for flags and Groq params.

    Returns
    -------
    dict[str, Any] or None
        Parsed layout dict on success; ``None`` on disabled feature, import
        errors, API errors, or invalid JSON.

    Notes
    -----
    Sets module global ``_LAST_GROQ_LAYOUT_RESPONSE_TEXT`` to raw text or
    ``None``. Instantiates ``GroqClient`` from layout config when
    ``use_llm_layout`` is true.

    Edge cases
    ----------
    Swallows broad ``Exception`` and returns ``None``. Returns ``None`` when
    ``use_llm_layout`` is false or Groq import fails.
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
    Build a short comma-joined prompt for ControlNet-conditioned generation.

    Parameters
    ----------
    constraints : dict
        Structured constraints with ``characters``, ``actions``, ``setting``.

    Returns
    -------
    str
        Prompt of joined fragments, truncated to 200 characters when longer.

    Notes
    -----
    Includes character names absent from the lowercased first action sentence,
    first clause of first action, and first clause of setting. Prints final
    length to stdout.

    Edge cases
    ----------
    Missing lists default to empty contributions; ``print`` executes even for
    empty ``parts`` after join.
    """
    parts: list[str] = []

    char_names: list[str] = []
    for c in constraints.get("characters", []):
        if isinstance(c, dict) and c.get("name"):
            char_names.append(str(c["name"]).strip())

    actions = constraints.get("actions", [])
    action_line = ""
    if isinstance(actions, list) and actions:
        action_line = str(actions[0]).split(".")[0].strip()

    action_lower = action_line.lower()
    names_only = [n for n in char_names if n and n.lower() not in action_lower]

    if names_only:
        parts.append(", ".join(names_only))

    if action_line:
        parts.append(action_line)

    setting = constraints.get("setting")
    if setting:
        phrase = str(setting).split(".")[0].strip()
        if phrase:
            parts.append(phrase)

    prompt = ", ".join(parts)
    if len(prompt) > 200:
        prompt = prompt[:200]

    print("FINAL PROMPT LENGTH:", len(prompt))
    return prompt


def build_constraints(
    scene: dict,
    character_descriptions: dict[str, str],
    config: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Assemble structured constraints and attach a per-scene layout plan.

    Parameters
    ----------
    scene : dict
        Parser scene with ``characters_present`` and optional description,
        setting, mood, time fields.
    character_descriptions : dict[str, str]
        Maps character name to textual description for constraint payload.
    config : dict[str, Any] or None, optional
        When ``None``, treated as empty dict for layout settings.

    Returns
    -------
    dict
        Constraint dict with ``characters``, ``actions``, ``setting``, ``time``,
        ``mood``, and ``layout`` keys.

    Notes
    -----
    Populates ``characters`` from ``characters_present``, adds scene description
    to ``actions`` when present, copies setting/time/mood. Layout selection:
    optional LLM via ``generate_layout_with_llm`` with JSON disk cache keyed by
    scene; falls back to ``fallback_layout`` on validation failure. Prints cache
    and layout diagnostics to stdout.

    Edge cases
    ----------
    When cached layout invalidates, removes stale cache entry before fallback.
    Global ``_LAST_GROQ_LAYOUT_RESPONSE_TEXT`` reflects last Groq raw output in
    LLM path.
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
    Flatten structured constraints into a compact natural-language prompt.

    Parameters
    ----------
    constraints : dict
        Same structure produced by ``build_constraints``.
    include_layout : bool, optional
        When ``True``, prefixes with non-redundant ``name at position depth``
        fragments drawn from ``constraints["layout"]``.

    Returns
    -------
    str
        Comma-joined prompt truncated to 200 characters if needed.

    Notes
    -----
    Skips character names already present in the lowercased action clause when
    building layout fragments and name list. Logs final length via print.

    Edge cases
    ----------
    If ``layout`` missing or malformed, layout insertion is skipped silently
    (guarded by isinstance checks).
    """
    parts: list[str] = []

    char_names: list[str] = []
    for c in constraints.get("characters", []):
        if isinstance(c, dict) and c.get("name"):
            char_names.append(str(c["name"]).strip())

    actions = constraints.get("actions", [])
    action_line = ""
    if isinstance(actions, list) and actions:
        action_line = str(actions[0]).split(".")[0].strip()

    action_lower = action_line.lower()
    names_only = [n for n in char_names if n and n.lower() not in action_lower]

    if include_layout:
        layout = constraints.get("layout") or {}
        layout_chars = layout.get("characters") if isinstance(layout, dict) else None
        if isinstance(layout_chars, list):
            layout_frags: list[str] = []
            for c in layout_chars:
                if isinstance(c, dict) and "name" in c and "position" in c and "depth" in c:
                    nm = str(c["name"]).strip()
                    if nm.lower() in action_lower:
                        continue
                    layout_frags.append(f'{nm} at {c["position"]} {c["depth"]}')
            for frag in reversed(layout_frags):
                parts.insert(0, frag)

    if names_only:
        parts.append(", ".join(names_only))

    if action_line:
        parts.append(action_line)

    setting = constraints.get("setting")
    if setting:
        phrase = str(setting).split(".")[0].strip()
        if phrase:
            parts.append(phrase)

    prompt = ", ".join(parts)
    if len(prompt) > 200:
        prompt = prompt[:200]

    print("FINAL PROMPT LENGTH:", len(prompt))
    return prompt


# --------------------------------------------------
# Legacy prompt assembly (fallback)
# --------------------------------------------------


def build_prompt(
    scene: dict,
    character_descriptions: dict[str, str],
    style_prefix: str,
    style_suffix: str,
):
    """
    Legacy prompt assembly with progressive truncation to a character budget.

    Parameters
    ----------
    scene : dict
        Must include non-empty stripped ``description``; may include ``setting``,
        ``time_of_day``, ``mood``.
    character_descriptions : dict[str, str]
        Used to prefix ``name (desc)`` fragments before the scene description.
    style_prefix : str
        Accepted for API compatibility; not referenced in the function body.
    style_suffix : str
        Accepted for API compatibility; not referenced in the function body.

    Returns
    -------
    str
        Final prompt after optional removal of mood, shortening setting, or hard
        slice to ``MAX_CHARS`` (320).

    Notes
    -----
    Builds ``main_block`` from character context plus description, appends
    setting and time, then mood. If over budget, drops mood; if still over,
    replaces setting with first 50 characters of raw ``setting`` string; if
    still over, truncates tail to ``MAX_CHARS``. Logs pre- and post-trim strings.

    Raises
    ------
    ValueError
        If ``description`` is missing or whitespace-only after strip.

    Edge cases
    ----------
    When ``setting_block`` is falsey, the second trimming stage is skipped.
    ``style_prefix`` and ``style_suffix`` do not affect the returned string.
    """
    MAX_CHARS = 320

    description = scene.get("description", "").strip()
    if not description:
        raise ValueError("Scene missing description")

    setting = scene.get("setting", "").strip()
    time = scene.get("time_of_day", "").strip()
    mood = scene.get("mood", "").strip()

    grounded_sentences = []

    for name, desc in character_descriptions.items():
        desc = desc.strip()
        if desc:
            grounded_sentences.append(f"{name} ({desc})")

    char_context = ", ".join(grounded_sentences)

    if char_context:
        main_block = f"{char_context}. {description}"
    else:
        main_block = description

    setting_parts = []

    if setting:
        setting_parts.append(setting)
    if time:
        setting_parts.append(time)

    setting_block = ", ".join(setting_parts)

    mood_block = mood if mood else ""

    parts = [main_block]

    if setting_block:
        parts.append(setting_block)

    if mood_block:
        parts.append(mood_block)

    full_prompt = ". ".join(parts)
    whole_prompt = full_prompt

    if len(full_prompt) > MAX_CHARS:
        if mood_block:
            parts = [main_block]
            if setting_block:
                parts.append(setting_block)
            full_prompt = ". ".join(parts)

    if len(full_prompt) > MAX_CHARS:
        if setting_block:
            setting_short = setting[:50]
            parts = [main_block, setting_short]
            full_prompt = ". ".join(parts)

    if len(full_prompt) > MAX_CHARS:
        full_prompt = full_prompt[:MAX_CHARS]

    logger.info(
        "\n[Prompt Debug - NEW]\n"
        f"[SCENE ID]: {scene.get('scene_id', 'unknown')}\n\n"
        f"WHOLE ({len(whole_prompt)} chars): {whole_prompt}\n\n"
        f"FINAL ({len(full_prompt)} chars): {full_prompt}\n"
    )

    return full_prompt