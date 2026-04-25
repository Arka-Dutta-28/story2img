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
        """
        Configure the Gemini SDK and construct a reusable ``GenerativeModel``.

        Parameters
        ----------
        model : str, optional
            Gemini model name (default ``gemini-1.5-flash``).
        temperature : float, optional
            Passed to ``LLMBase`` and to ``GenerationConfig``.
        max_tokens : int, optional
            Mapped to ``max_output_tokens`` in ``GenerationConfig``.
        api_key : str or None, optional
            API key; if ``None``, uses ``GEMINI_API_KEY`` from the environment.

        Returns
        -------
        None

        Notes
        -----
        Calls ``genai.configure``, builds ``GenerationConfig``, and stores
        ``self._client`` as ``GenerativeModel`` for subsequent ``complete`` calls.

        Raises
        ------
        EnvironmentError
            If no API key is available.
        ImportError
            If ``google.generativeai`` cannot be imported.

        Edge cases
        ----------
        Instantiates the module-level client once per instance; does not validate
        remote model availability until ``generate_content`` runs.
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable or pass api_key=."
            )

        try:
            import google.generativeai as genai
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
        Call ``generate_content`` on the configured Gemini model.

        Parameters
        ----------
        prompt : str
            Primary user content.
        system_prompt : str or None, optional
            If given, concatenated before ``prompt`` with a ``---`` separator.

        Returns
        -------
        LLMResponse
            Text from ``raw_response.text``, token counts when present, and
            ``raw`` set to the SDK response object.

        Notes
        -----
        Logs debug line with character length; on API failure logs error and
        wraps exception in ``RuntimeError``. Reads optional token fields from
        ``usage_metadata`` via ``getattr``.

        Raises
        ------
        RuntimeError
            If ``generate_content`` raises.

        Edge cases
        ----------
        If the SDK omits ``usage_metadata``, token fields in ``LLMResponse`` are
        ``None``. Behavior for empty ``prompt`` depends on the Gemini API.
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
