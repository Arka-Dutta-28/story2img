"""
llm/nvidia_client.py
--------------------
NVIDIA NIM / AI Foundation Models client implementing LLMBase.

Uses the OpenAI-compatible Chat Completions API against NVIDIA's gateway.
API key is read from the environment variable NVIDIA_API_KEY.

Usage
-----
    client = NvidiaClient(
        model="minimaxai/minimax-m2.7",
        temperature=0.4,
        max_tokens=2048,
    )
    response = client.complete(prompt="...", system_prompt="...")
    print(response.text)

See https://build.nvidia.com/ for model catalogue and keys.
"""

import logging
import os
from typing import Any, Optional

from llm.base import LLMBase, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaClient(LLMBase):
    """
    LLM client for NVIDIA-hosted models (OpenAI-compatible HTTP API).

    Parameters
    ----------
    model       : NVIDIA catalog model id, e.g. "minimaxai/minimax-m2.7".
    temperature : Sampling temperature.
    max_tokens  : Maximum completion tokens.
    top_p       : Nucleus sampling; set on each request when not None.
    base_url    : API root (default: NVIDIA integrate endpoint).
    api_key     : API key. Falls back to NVIDIA_API_KEY env var if None.
    """

    def __init__(
        self,
        model: str = "minimaxai/minimax-m2.7",
        temperature: float = 0.4,
        max_tokens: int = 2048,
        top_p: Optional[float] = 0.95,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Create an OpenAI-compatible client targeting NVIDIA's gateway.

        Parameters
        ----------
        model : str, optional
            Catalog model id (default ``minimaxai/minimax-m2.7``).
        temperature : float, optional
            Sampling temperature forwarded to chat completions.
        max_tokens : int, optional
            Maximum completion tokens.
        top_p : float or None, optional
            Nucleus sampling parameter; if ``None``, omitted from requests.
        base_url : str, optional
            API root; defaults to ``DEFAULT_BASE_URL``.
        api_key : str or None, optional
            If ``None``, uses ``NVIDIA_API_KEY`` from the environment.

        Returns
        -------
        None

        Notes
        -----
        Stores ``self._top_p`` and ``self._client`` as ``OpenAI(...)``.

        Raises
        ------
        EnvironmentError
            If no API key is available.
        ImportError
            If ``openai`` is not installed.

        Edge cases
        ----------
        ``top_p`` may be explicitly ``None`` to disable the request field.
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "NVIDIA API key not found. "
                "Set the NVIDIA_API_KEY environment variable or pass api_key=."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. "
                "Run: pip install openai"
            ) from exc

        self._top_p = top_p
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)
        logger.info("NvidiaClient ready | model=%s | base_url=%s", self.model, base_url)

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Perform a non-streaming chat completion against the NVIDIA endpoint.

        Parameters
        ----------
        prompt : str
            User message body.
        system_prompt : str or None, optional
            Optional system message prepended when provided.

        Returns
        -------
        LLMResponse
            ``text`` from the first message content or ``""`` if missing;
            optional usage fields; ``raw`` response object.

        Notes
        -----
        Builds kwargs with ``stream=False`` and includes ``top_p`` only when
        ``self._top_p`` is not ``None``.

        Raises
        ------
        RuntimeError
            If the chat completion call raises.

        Edge cases
        ----------
        If ``msg.content`` is ``None``, returns empty string for ``text``.
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "NvidiaClient.complete | messages=%d | prompt_length=%d chars",
            len(messages),
            len(prompt),
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p

        try:
            raw_response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("NVIDIA API call failed: %s", exc)
            raise RuntimeError(f"NVIDIA completion failed: {exc}") from exc

        choice = raw_response.choices[0]
        msg = choice.message
        text = msg.content if msg and msg.content is not None else ""

        usage = getattr(raw_response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        logger.debug(
            "NvidiaClient.complete | prompt_tokens=%s completion_tokens=%s",
            prompt_tokens,
            completion_tokens,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw_response,
        )
