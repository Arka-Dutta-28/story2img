"""
llm/groq_client.py
------------------
Groq LLM client implementing LLMBase.

Uses the official `groq` Python SDK.
API key is read from the environment variable GROQ_API_KEY.

Usage
-----
    client = GroqClient(model="llama3-8b-8192", temperature=0.4, max_tokens=2048)
    response = client.complete(prompt="...", system_prompt="...")
    print(response.text)
"""

import logging
import os
from typing import Optional

from llm.base import LLMBase, LLMResponse

logger = logging.getLogger(__name__)


class GroqClient(LLMBase):
    """
    LLM client for Groq-hosted models (LLaMA 3, Mixtral, etc.).

    Parameters
    ----------
    model       : Groq model name, e.g. "llama3-8b-8192".
    temperature : Sampling temperature.
    max_tokens  : Maximum completion tokens.
    api_key     : Groq API key. Falls back to GROQ_API_KEY env var if None.
    """

    def __init__(
        self,
        model: str = "llama3-8b-8192",
        temperature: float = 0.4,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Groq API key not found. "
                "Set the GROQ_API_KEY environment variable or pass api_key=."
            )

        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "groq is not installed. "
                "Run: pip install groq"
            ) from exc

        self._client = Groq(api_key=resolved_key)
        logger.info("GroqClient ready | model=%s", self.model)

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a prompt to Groq and return a standardised LLMResponse.

        Groq's API is OpenAI-compatible and supports distinct system
        and user roles natively.

        Parameters
        ----------
        prompt        : User-turn prompt text.
        system_prompt : Optional system instruction.

        Returns
        -------
        LLMResponse
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "GroqClient.complete | messages=%d | prompt_length=%d chars",
            len(messages), len(prompt),
        )

        try:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            raise RuntimeError(f"Groq completion failed: {exc}") from exc

        choice = raw_response.choices[0]
        text = choice.message.content

        usage = getattr(raw_response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        logger.debug(
            "GroqClient.complete | prompt_tokens=%s completion_tokens=%s",
            prompt_tokens, completion_tokens,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw_response,
        )
