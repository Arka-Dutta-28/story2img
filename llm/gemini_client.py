"""
llm/gemini_client.py
--------------------
Gemini LLM client implementing LLMBase.

Uses the official `google-generativeai` SDK.
API key is read from the environment variable GEMINI_API_KEY.

Usage
-----
    client = GeminiClient(model="gemini-1.5-flash", temperature=0.4, max_tokens=2048)
    response = client.complete(prompt="...", system_prompt="...")
    print(response.text)
"""

import logging
import os
from typing import Optional

from llm.base import LLMBase, LLMResponse

logger = logging.getLogger(__name__)


class GeminiClient(LLMBase):
    """
    LLM client for Google Gemini models.

    Parameters
    ----------
    model       : Gemini model name, e.g. "gemini-1.5-flash".
    temperature : Sampling temperature.
    max_tokens  : Maximum output tokens.
    api_key     : Gemini API key. Falls back to GEMINI_API_KEY env var if None.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.4,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable or pass api_key=."
            )

        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=resolved_key)

        self._generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        # Model is instantiated once and reused across calls.
        self._genai = genai
        self._client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self._generation_config,
        )

        logger.info("GeminiClient ready | model=%s", self.model)

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a prompt to Gemini and return a standardised LLMResponse.

        Gemini does not support a separate system role in the basic
        GenerativeModel API; if a system_prompt is provided it is
        prepended to the user prompt with a clear delimiter.

        Parameters
        ----------
        prompt        : User-turn prompt text.
        system_prompt : Optional instruction context.

        Returns
        -------
        LLMResponse
        """
        full_prompt = (
            f"{system_prompt}\n\n---\n\n{prompt}"
            if system_prompt
            else prompt
        )

        logger.debug("GeminiClient.complete | prompt_length=%d chars", len(full_prompt))

        try:
            raw_response = self._client.generate_content(full_prompt)
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini completion failed: {exc}") from exc

        text = raw_response.text

        # Token counts are available on usage_metadata when reported.
        usage = getattr(raw_response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)

        logger.debug(
            "GeminiClient.complete | prompt_tokens=%s completion_tokens=%s",
            prompt_tokens, completion_tokens,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw_response,
        )
