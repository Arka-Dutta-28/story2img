"""
llm/parser.py
-------------
Story Parser — converts a raw story string into structured JSON.

Responsibilities
----------------
- Send the story to the LLM with a strict JSON-forcing prompt
- Parse and validate the returned JSON
- Retry up to MAX_RETRIES times if the output is malformed
- Return a validated ParsedStory dataclass

Output contract (strict)
------------------------
{
    "characters": [
        {
            "name": str,
            "description": str        # physical appearance, clothing, distinguishing features
        },
        ...
    ],
    "scenes": [
        {
            "scene_id": int,          # 1-indexed
            "description": str,       # what is happening
            "setting": str,           # where / environment
            "characters_present": [str, ...],   # names only, subset of characters[]
            "mood": str,              # emotional tone
            "time_of_day": str        # morning / afternoon / night / unknown
        },
        ...
    ],
    "style": str                      # overall visual style inferred from the story
}

Does NOT modify any existing module.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from llm.base import LLMBase, LLMResponse

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedStory:
    """
    Validated output from the Parser.

    Attributes
    ----------
    characters : List of character dicts with 'name' and 'description'.
    scenes     : List of scene dicts (see module docstring for full schema).
    style      : Overall visual style string inferred from the story.
    raw_json   : The raw dict returned by the LLM (before dataclass wrapping).
    """
    characters: list[dict[str, str]]
    scenes: list[dict[str, Any]]
    style: str
    raw_json: dict[str, Any]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict JSON planner for a short story-to-image pipeline.

Your job is to convert a short story into a structured multi-frame image plan.

STRICT RULES:
- Output ONLY valid JSON
- No explanations, no markdown, no code fences
- Do not include ``` or ```json
- Do not add extra text before or after JSON
- Use EXACT same character names across all frames
- Do NOT rename or shorten character names
- Do not invent new characters not implied by the story
- Infer a reasonable frame count N from pacing and events
- Output EXACTLY N frames
- Every frame must be visually filmable and image-generation-ready
- If unsure, use "unknown" or empty list
"""

def _build_user_prompt(story: str) -> str:
    """
    Format the user message that requests JSON story structure from the LLM.

    Parameters
    ----------
    story : str
        Raw story text embedded at the end of the template.

    Returns
    -------
    str
        Multi-line instruction string including schema and ``STORY:`` section.

    Notes
    -----
    Static template defines required JSON keys and content rules; appends the
    provided ``story`` verbatim inside the template.

    Edge cases
    ----------
    Does not validate ``story`` length or encoding.
    """
    return f"""
IMPORTANT:
- This is SHORT STORY -> MULTIPLE STILL IMAGE SCENES (NOT video)
- Infer a reasonable scene count N from story pacing and event density
- Output EXACTLY N scenes in order
- Each scene must represent a distinct visual beat
- Do not merge all events into one scene
- Use EXACT same character names in characters[] and in every scene's characters_present

IMPORTANT FOR DOWNSTREAM IMAGE GENERATION:
- Character descriptions must be concise but visually informative (max ~12 words each)
- Focus on clothing, silhouette, colors, age cues, distinctive features
- Avoid backstory and abstract personality traits
- Scene descriptions must be concrete, visible, and drawable
- Include action, subjects, objects, setting, composition, lighting, and mood cues
- Avoid vague text like "something happens" or purely emotional statements
- Prefer physically observable details

Return a JSON object with EXACTLY this structure:

{{
  "characters": [
    {{
      "name": "<character name>",
      "description": "<concise visual description>"
    }}
  ],
  "scenes": [
    {{
      "scene_id": <integer starting at 1>,
      "description": "<what happens in this scene, concrete action>",
      "setting": "<location + environment details>",
      "characters_present": ["<name>", "..."],
      "mood": "<emotional tone>",
      "time_of_day": "<morning | afternoon | evening | night | unknown>"
    }}
  ],
  "style": "<overall visual style inferred from the story>"
}}

STORY:
{story}

Return ONLY the JSON. No other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """
    Strip optional Markdown code fences from model output.

    Parameters
    ----------
    text : str
        Raw model output, possibly wrapped in fenced blocks.

    Returns
    -------
    str
        ``text`` with leading `` ```json``/`` ``` `` and trailing `` ``` ``
        removed, then stripped.

    Notes
    -----
    Applies case-insensitive regex to the opening fence; trims whitespace
    before and after.

    Edge cases
    ----------
    If no fences match, returns stripped ``text`` unchanged aside from outer
    whitespace normalization from the initial ``strip``.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Slice the substring from the first ``{`` through the last ``}`` inclusive.

    Parameters
    ----------
    text : str
        String expected to contain a JSON object.

    Returns
    -------
    str
        Substring ``text[start : end + 1]`` for the first ``{`` and last ``}``.

    Notes
    -----
    Does not parse JSON; only locates delimiters.

    Raises
    ------
    ValueError
        If ``{`` or ``}`` is missing or their positions are invalid.

    Edge cases
    ----------
    If multiple objects exist, the slice spans from the first opening brace to
    the final closing brace in the string (may include extra content).
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM output.")
    return text[start : end + 1]


def _validate(data: dict[str, Any]) -> list[str]:
    """
    Check required keys and types for parser output contract.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed JSON object to validate in-place logically (read-only).

    Returns
    -------
    list[str]
        Human-readable error messages; empty if ``data`` satisfies the schema.

    Notes
    -----
    Validates ``characters`` (list of dicts with ``name`` and ``description``),
    non-empty ``scenes`` (each dict with fixed scene fields), and non-empty
    string ``style``.

    Edge cases
    ----------
    Returns multiple errors when several constraints fail; does not short-circuit
    on first error beyond natural control flow.
    """
    errors: list[str] = []

    if "characters" not in data:
        errors.append("Missing key: 'characters'")
    elif not isinstance(data["characters"], list):
        errors.append("'characters' must be a list")
    else:
        for i, c in enumerate(data["characters"]):
            if not isinstance(c, dict):
                errors.append(f"characters[{i}] must be a dict")
            else:
                for field in ("name", "description"):
                    if field not in c:
                        errors.append(f"characters[{i}] missing '{field}'")

    if "scenes" not in data:
        errors.append("Missing key: 'scenes'")
    elif not isinstance(data["scenes"], list):
        errors.append("'scenes' must be a list")
    elif len(data["scenes"]) == 0:
        errors.append("'scenes' list is empty")
    else:
        required_scene_fields = (
            "scene_id", "description", "setting",
            "characters_present", "mood", "time_of_day",
        )
        for i, s in enumerate(data["scenes"]):
            if not isinstance(s, dict):
                errors.append(f"scenes[{i}] must be a dict")
            else:
                for field in required_scene_fields:
                    if field not in s:
                        errors.append(f"scenes[{i}] missing '{field}'")

    if "style" not in data:
        errors.append("Missing key: 'style'")
    elif not isinstance(data["style"], str) or not data["style"].strip():
        errors.append("'style' must be a non-empty string")

    return errors


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------

class StoryParser:
    """
    Parses a raw story string into a structured ParsedStory using an LLM.

    Parameters
    ----------
    llm : Any LLMBase-compliant client (GeminiClient recommended).

    Usage
    -----
        parser = StoryParser(llm=gemini_client)
        result = parser.parse("Once upon a time ...")
        print(result.characters)
        print(result.scenes)
        print(result.style)
    """

    def __init__(self, llm: LLMBase) -> None:
        """
        Attach an ``LLMBase`` client used by ``parse``.

        Parameters
        ----------
        llm : LLMBase
            Provider client implementing ``complete``.

        Returns
        -------
        None

        Notes
        -----
        Stores ``self._llm`` and logs initialisation.

        Edge cases
        ----------
        None.
        """
        self._llm = llm
        logger.info("StoryParser initialised | llm=%r", self._llm)

    # ------------------------------------------------------------------
    def parse(self, story: str) -> ParsedStory:
        """
        Parse a story string into a validated ``ParsedStory`` via the LLM.

        Parameters
        ----------
        story : str
            Raw narrative text. Must be non-empty after stripping.

        Returns
        -------
        ParsedStory
            Dataclass with ``characters``, ``scenes``, ``style``, and ``raw_json``.

        Notes
        -----
        Builds a fixed user prompt, calls ``self._llm.complete`` with
        ``SYSTEM_PROMPT``, strips fences, extracts JSON, parses with ``json.loads``,
        and validates with ``_validate``. On success returns ``ParsedStory``; on
        failure logs and retries up to ``MAX_RETRIES``.

        Raises
        ------
        ValueError
            If ``story`` is empty or whitespace-only.
        RuntimeError
            If all attempts fail; message includes the last recorded error.

        Edge cases
        ----------
        LLM exceptions are caught per attempt and recorded in ``last_error``.
        Validation failures accumulate a list string in ``last_error``. If the
        final attempt fails, ``last_error`` may be ``None`` only if no body ran
        (not the case for normal loops).
        """
        if not story or not story.strip():
            raise ValueError("Story string is empty.")

        prompt = _build_user_prompt(story)
        last_error: Optional[str] = None

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info("StoryParser.parse | attempt %d/%d", attempt, MAX_RETRIES)

            try:
                response: LLMResponse = self._llm.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                )
            except Exception as exc:
                last_error = f"LLM call failed: {exc}"
                logger.warning("Attempt %d — LLM error: %s", attempt, exc)
                continue

            raw_text = response.text
            logger.debug("Raw LLM output (attempt %d):\n%s", attempt, raw_text)

            try:
                cleaned = _strip_code_fences(raw_text)
                json_str = _extract_json_object(cleaned)
                data = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning("Attempt %d — %s", attempt, last_error)
                continue

            errors = _validate(data)
            if errors:
                last_error = f"Validation errors: {errors}"
                logger.warning("Attempt %d — %s", attempt, last_error)
                continue

            logger.info(
                "StoryParser.parse | success on attempt %d | "
                "characters=%d scenes=%d style=%r",
                attempt,
                len(data["characters"]),
                len(data["scenes"]),
                data["style"],
            )

            return ParsedStory(
                characters=data["characters"],
                scenes=data["scenes"],
                style=data["style"],
                raw_json=data,
            )

        raise RuntimeError(
            f"StoryParser failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
