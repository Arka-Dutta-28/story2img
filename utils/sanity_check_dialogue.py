"""
utils/sanity_check_dialogue.py
------------------------------
Check the dialogue branch of the parser contract. No config, no API key, no
cached logs, so it runs on a clean clone:

    python utils/sanity_check_dialogue.py

Dialogue was added to the scene schema so speech balloons have text to carry,
and so the parser resolves pronouns to character names rather than emitting
"he" as a speaker. It is validated when present and tolerated when absent,
because requiring it would invalidate every cached parser log under logs/.
This asserts both halves of that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.parser import _validate  # noqa: E402


def base(dialogue=...):
    """Smallest parse that satisfies everything except, optionally, dialogue."""
    scene = {
        "scene_id": 1,
        "description": "The mouse gnaws the net",
        "setting": "a forest clearing",
        "characters_present": ["Mouse"],
        "mood": "tense",
        "time_of_day": "afternoon",
    }
    if dialogue is not ...:
        scene["dialogue"] = dialogue
    return {
        "characters": [{"name": "Mouse", "description": "small grey mouse"}],
        "scenes": [scene],
        "style": "storybook illustration",
    }


def errs(dialogue=...):
    return [e for e in _validate(base(dialogue)) if "dialogue" in e]


def main() -> int:
    # the contract without dialogue must still validate: cached parser logs
    # predate this field and the sanity checks read them
    assert _validate(base()) == [], _validate(base())

    # a scene with no speech is legitimate
    assert errs([]) == []

    # a well-formed turn passes
    assert errs([{"speaker": "Mouse", "line": "Please let me go"}]) == []
    assert errs([
        {"speaker": "Mouse", "line": "Please let me go"},
        {"speaker": "narrator", "line": "And so the lion relented."},
    ]) == []

    # wrong container type
    assert errs("Please let me go"), "a string should not pass as dialogue"
    assert errs({"speaker": "Mouse", "line": "hi"}), "a bare dict should not pass"

    # entry not a dict
    assert errs(["Please let me go"]), "a list of strings should not pass"

    # missing or empty fields
    assert errs([{"line": "Please let me go"}]), "missing speaker should fail"
    assert errs([{"speaker": "Mouse"}]), "missing line should fail"
    assert errs([{"speaker": "", "line": "hi"}]), "empty speaker should fail"
    assert errs([{"speaker": "Mouse", "line": "   "}]), "whitespace line should fail"
    assert errs([{"speaker": None, "line": "hi"}]), "non-string speaker should fail"

    # the error message must name the scene and turn, or debugging a 12-scene
    # parse becomes guesswork
    msg = errs([{"speaker": "Mouse"}])[0]
    assert "scenes[0]" in msg and "dialogue[0]" in msg, msg

    # a good turn alongside a bad one still reports the bad one
    both = errs([
        {"speaker": "Mouse", "line": "Please let me go"},
        {"speaker": "Lion"},
    ])
    assert len(both) == 1 and "dialogue[1]" in both[0], both

    print("dialogue contract OK: 15 assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
