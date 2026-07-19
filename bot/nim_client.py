"""Async client for NVIDIA NIM (OpenAI-compatible) chat, vision and embeddings.

NIM exposes an OpenAI-compatible REST API at ``NIM_BASE_URL`` so we drive it with
the official ``openai`` async SDK. Everything the bot needs from an LLM funnels
through this one class, which keeps retries, error translation and model
selection in a single place.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Dict, List, Optional, Sequence

from openai import APIError, APIStatusError, AsyncOpenAI, RateLimitError

from .config import Settings

log = logging.getLogger("nim")

# Chat message shape reused across the bot.
Message = Dict[str, object]


class NimError(RuntimeError):
    """Raised when a NIM request fails after retries. Message is user-safe."""


class NimClient:
    """Thin, retrying wrapper over the NVIDIA NIM endpoints used by the bot."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.nim_base_url,
            api_key=settings.nvidia_api_key or "missing-key",
            timeout=60.0,
            max_retries=0,  # we implement our own backoff below
        )

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the assistant text for a chat completion."""
        model = model or self._settings.chat_model
        resp = await self._with_retries(
            lambda: self._client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=(
                    self._settings.temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens or self._settings.max_tokens,
            )
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()

    # ---------------------------------------------------------------- vision
    async def vision(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Ask a vision model a question about a single image."""
        model = model or self._settings.vision_model
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        resp = await self._with_retries(
            lambda: self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=max_tokens or self._settings.max_tokens,
            )
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------- embeddings
    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str = "passage",
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts.

        NVIDIA retrieval embedding models require an ``input_type`` of either
        ``"query"`` or ``"passage"``, passed through ``extra_body``.
        """
        if not texts:
            return []
        model = model or self._settings.embed_model
        resp = await self._with_retries(
            lambda: self._client.embeddings.create(
                model=model,
                input=list(texts),
                extra_body={"input_type": input_type, "truncate": "END"},
            )
        )
        # Preserve input order.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [list(item.embedding) for item in ordered]

    # ------------------------------------------------------------- internals
    async def _with_retries(self, call, *, attempts: int = 3):
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return await call()
            except RateLimitError as exc:
                last_exc = exc
                log.warning("NIM rate limited (attempt %d/%d)", attempt, attempts)
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code == 401:
                    raise NimError(
                        "NVIDIA rejected the API key (401). Check NVIDIA_API_KEY."
                    ) from exc
                if exc.status_code and exc.status_code < 500:
                    # 4xx other than 401/429 will not improve on retry.
                    raise NimError(
                        f"NIM request failed ({exc.status_code}). "
                        "The model name or request may be invalid."
                    ) from exc
                log.warning("NIM server error %s (attempt %d/%d)", exc.status_code, attempt, attempts)
            except APIError as exc:
                last_exc = exc
                log.warning("NIM API error (attempt %d/%d): %s", attempt, attempts, exc)
            if attempt < attempts:
                await asyncio.sleep(delay)
                delay *= 2
        raise NimError(
            "NIM is unavailable right now. Please try again in a moment."
        ) from last_exc
