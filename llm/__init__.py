"""
llm/__init__.py
---------------
Exports the LLM base class, response dataclass, and provider clients.
Provides a factory function `build_llm_client` to instantiate the
correct client from a config dict.
"""

from llm.base import LLMBase, LLMResponse
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient
from llm.nvidia_client import NvidiaClient
from llm.parser import StoryParser, ParsedStory

from typing import Any, Optional


def build_llm_client(llm_config: dict[str, Any]) -> LLMBase:
    """
    Construct a concrete ``LLMBase`` implementation from a config mapping.

    Parameters
    ----------
    llm_config : dict[str, Any]
        The ``llm`` section from ``config.yaml``. Must include ``provider`` and
        the matching nested block (``gemini``, ``groq``, or ``nvidia``).

    Returns
    -------
    LLMBase
        ``GeminiClient``, ``GroqClient``, or ``NvidiaClient`` instance.

    Notes
    -----
    Dispatches on ``llm_config["provider"]`` (case-insensitive). For NVIDIA,
    ``top_p`` may be ``None`` and is passed through as optional. ``base_url``
    defaults to the integrate API host when omitted.

    Raises
    ------
    ValueError
        If ``provider`` is not one of the supported strings.

    Edge cases
    ----------
    Unknown ``provider`` raises with a message listing supported values.
    Missing nested keys required by each client constructor will raise from
    that client's ``__init__``.
    """
    provider = llm_config.get("provider", "").lower()

    if provider == "gemini":
        cfg = llm_config["gemini"]
        return GeminiClient(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.4),
            max_tokens=cfg.get("max_output_tokens", 2048),
        )

    if provider == "groq":
        cfg = llm_config["groq"]
        return GroqClient(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.4),
            max_tokens=cfg.get("max_tokens", 2048),
        )

    if provider == "nvidia":
        cfg = llm_config["nvidia"]
        raw_top = cfg.get("top_p", 0.95)
        top_p_opt: Optional[float] = None if raw_top is None else float(raw_top)
        return NvidiaClient(
            model=cfg["model"],
            temperature=float(cfg.get("temperature", 0.4)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            top_p=top_p_opt,
            base_url=str(cfg.get("base_url", "https://integrate.api.nvidia.com/v1")),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Supported values: 'gemini', 'groq', 'nvidia'."
    )


__all__ = [
    "LLMBase",
    "LLMResponse",
    "GeminiClient",
    "GroqClient",
    "NvidiaClient",
    "StoryParser",
    "ParsedStory",
    "build_llm_client",
]
