"""Test doubles shared by the test modules (import with ``from fakes import ...``)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


class ScriptedNim:
    """Minimal fake of NimClient.

    * ``chat`` returns the queued replies in order (or ``default_reply``) and
      records every call.
    * ``embed`` maps text to a bag-of-words vector over ``vocab`` so retrieval
      is predictable in tests.
    """

    def __init__(
        self,
        replies: Optional[Sequence[str]] = None,
        vocab: Optional[Sequence[str]] = None,
        *,
        default_reply: str = "ok",
        embed_model_name: str = "fake/embedder",
        default_min_score: float = 0.2,
    ) -> None:
        self.replies: List[str] = list(replies or [])
        self.default_reply = default_reply
        self.vocab = list(vocab or [])
        self.chat_calls: List[Dict] = []
        self.embed_calls: List[Dict] = []
        self.vision_calls: List[Dict] = []
        self.embed_model_name = embed_model_name
        self.default_min_score = default_min_score

    async def chat(self, messages, **kwargs) -> str:
        self.chat_calls.append({"messages": list(messages), **kwargs})
        return self.replies.pop(0) if self.replies else self.default_reply

    async def embed(self, texts, *, input_type="passage", model=None):
        self.embed_calls.append({"texts": list(texts), "input_type": input_type})
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z0-9]+", text.lower())
            vectors.append([float(words.count(v)) for v in self.vocab])
        return vectors

    async def vision(self, prompt, image_bytes, mime_type="image/png", **kwargs) -> str:
        self.vision_calls.append({"prompt": prompt, "bytes": image_bytes, "mime": mime_type})
        return self.replies.pop(0) if self.replies else self.default_reply

    async def close(self) -> None:
        return None
