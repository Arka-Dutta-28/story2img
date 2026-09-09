"""
utils/sanity_check_entities.py
------------------------------
Standalone check for entity grounding. No config, no API key, no cached logs,
so it runs on a clean clone:

    python utils/sanity_check_entities.py

Covers the four things that actually break: whole-word matching,
case-insensitivity, the double-expansion trap, and the sentinel that makes a
repeat call a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constraints.constraint_builder import (  # noqa: E402
    _ENTITY_MAP,
    _substitute_entity_words,
    normalize_entities,
)


def main() -> int:
    # whole word only: a key inside a longer word must not match
    assert _substitute_entity_words("mousetrap", _ENTITY_MAP) == "mousetrap"
    assert _substitute_entity_words("potato", _ENTITY_MAP) == "potato"

    # case-insensitive, and the replacement is the canonical lower-case phrase
    assert _substitute_entity_words("Lion", _ENTITY_MAP) == "adult male lion"
    assert _substitute_entity_words("LION", _ENTITY_MAP) == "adult male lion"

    # one pass does not re-expand within its own output
    assert _substitute_entity_words("a net", _ENTITY_MAP) == "a rope net"

    # ...but the substitution alone is NOT idempotent, which is why the
    # sentinel below exists. If this assert ever starts failing, someone made
    # the substitution idempotent and the sentinel can go.
    twice = _substitute_entity_words(_substitute_entity_words("a net", _ENTITY_MAP), _ENTITY_MAP)
    assert twice == "a rope rope net", twice

    # normalize_entities rewrites characters, actions, setting and layout names
    c = {
        "characters": [{"name": "Mouse"}, {"name": "Lion"}],
        "actions": ["The mouse gnaws the net"],
        "setting": "beside a pot of grapes",
        "layout": {"characters": [{"name": "crow"}]},
    }
    out = normalize_entities(c)
    assert [x["name"] for x in out["characters"]] == [
        "small gray mouse animal",
        "adult male lion",
    ]
    assert out["actions"][0] == "The small gray mouse animal gnaws the rope net"
    assert out["setting"] == "beside a clay water pot of purple grape cluster"
    assert out["layout"]["characters"][0]["name"] == "black crow"

    # the guard: a second call changes nothing
    before = (out["actions"][0], out["setting"])
    normalize_entities(out)
    assert (out["actions"][0], out["setting"]) == before, "sentinel failed, text double-expanded"

    # unmapped nouns pass through untouched, which is the common case for the
    # 142 fables the map does not cover
    u = {"characters": [{"name": "stork"}], "actions": ["the stork waits"], "setting": ""}
    assert normalize_entities(u)["actions"][0] == "the stork waits"

    print(f"entity grounding OK: {len(_ENTITY_MAP)} entities mapped, 11 assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
