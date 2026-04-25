"""
llm/base.py
-----------
Abstract base class for all LLM clients used in the pipeline.

Every LLM client (Gemini, Groq, NVIDIA, etc.) must inherit from LLMBase
and implement the `complete` method. This keeps the Parser fully
decoupled from any specific provider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standardised response returned by every LLM client.

    Attributes
    ----------
    text      : The raw text content returned by the model.
    model     : The model identifier that produced the response.
    prompt_tokens    : Number of tokens in the prompt (if reported by provider).
    completion_tokens: Number of tokens in the completion (if reported).
    raw       : The full, unmodified response object from the provider SDK.
    """
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw: Optional[object] = None


class LLMBase(ABC):
    """
    Abstract base class for LLM provider clients.

    Subclasses must implement:
        complete(prompt, system_prompt) -> LLMResponse

    Parameters
    ----------
    model       : Model identifier string (provider-specific).
    temperature : Sampling temperature (0.0 – 1.0).
    max_tokens  : Maximum tokens to generate in the completion.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        """
        Store generation parameters and log client construction.

        Parameters
        ----------
        model : str
            Provider-specific model identifier.
        temperature : float, optional
            Sampling temperature (default 0.4).
        max_tokens : int, optional
            Maximum tokens for the completion (default 2048).

        Returns
        -------
        None

        Notes
        -----
        Assigns ``self.model``, ``self.temperature``, and ``self.max_tokens`` and
        emits an INFO log line with the concrete subclass name.

        Edge cases
        ----------
        No validation is performed on ``temperature`` or ``max_tokens`` beyond
        assignment.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            "Initialised %s | model=%s | temperature=%s | max_tokens=%s",
            self.__class__.__name__, model, temperature, max_tokens,
        )

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Request a completion from the provider and return ``LLMResponse``.

        Parameters
        ----------
        prompt : str
            User or primary instruction text for the model.
        system_prompt : str or None, optional
            Optional system-level instructions when supported by the backend.

        Returns
        -------
        LLMResponse
            Normalised wrapper with text, model id, optional token counts, and raw
            provider payload.

        Notes
        -----
        Subclasses implement provider-specific HTTP/SDK calls. This base
        definition is abstract and not invoked directly.

        Edge cases
        ----------
        Contract for empty ``prompt`` or unsupported ``system_prompt`` is defined
        by each subclass.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a concise, eval-like string of the client configuration.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Form ``ClassName(model='...', temperature=..., max_tokens=...)``.

        Notes
        -----
        Uses the runtime class name and current attribute values.

        Edge cases
        ----------
        None.
        """
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
