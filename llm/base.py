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
        Send a prompt to the LLM and return a standardised LLMResponse.

        Parameters
        ----------
        prompt        : The user-turn prompt text.
        system_prompt : Optional system/instruction prompt (if provider supports it).

        Returns
        -------
        LLMResponse
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
