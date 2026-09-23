"""Test doubles shared by the test modules (import with ``from fakes import ...``).

* ``ScriptedNim`` — a NimClient stand-in with scripted replies.
* ``Fake*`` Discord objects — just enough of Interaction / Message / Channel /
  Guild for the command callbacks in ``bot/main.py`` to run without a gateway.
* ``minimal_pdf`` — a real, tiny PDF with extractable text.
"""
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

    offline = False

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


# ------------------------------------------------------------ Discord fakes
class FakeUser:
    def __init__(self, uid: int, name: str = "", *, bot: bool = False, roles=()) -> None:
        self.id = uid
        self.display_name = name or f"user{uid}"
        self.name = self.display_name
        self.bot = bot
        self.roles = [FakeRole(r) for r in roles]
        self.mention = f"<@{uid}>"

    def __str__(self) -> str:
        return self.name


class FakeRole:
    def __init__(self, rid: int) -> None:
        self.id = rid
        self.mention = f"<@&{rid}>"


class FakeAttachment:
    def __init__(self, aid: int, filename: str, data: bytes, content_type: Optional[str] = None):
        self.id = aid
        self.filename = filename
        self.content_type = content_type
        self.size = len(data)
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeMessage:
    def __init__(
        self,
        mid: int,
        author: FakeUser,
        content: str = "",
        *,
        channel=None,
        attachments=(),
        mentions=(),
        guild=None,
    ) -> None:
        self.id = mid
        self.author = author
        self.content = content
        self.clean_content = content
        self.attachments = list(attachments)
        self.mentions = list(mentions)
        self.channel = channel
        self.guild = guild
        self.jump_url = f"https://discord.com/channels/1/{getattr(channel, 'id', 0)}/{mid}"


class _AsyncList:
    """Async iterator that, like discord.py's _PinsIterator, is NOT awaitable."""

    def __init__(self, items) -> None:
        self._items = list(items)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self._items:
            yield item


class FakeChannel:
    def __init__(self, cid: int, name: str = "general", messages=(), pins=()) -> None:
        self.id = cid
        self.name = name
        self.mention = f"<#{cid}>"
        self.messages: List[FakeMessage] = list(messages)  # oldest first
        self._pins: List[FakeMessage] = list(pins)
        self.sent: List[Dict] = []
        self.history_calls: List[Dict] = []

    def history(self, *, limit=100, after=None, oldest_first=None):
        self.history_calls.append({"limit": limit, "after": after, "oldest_first": oldest_first})
        msgs = self.messages
        if after is not None:
            msgs = [m for m in msgs if m.id > after.id]
            oldest_first = True if oldest_first is None else oldest_first
        ordered = msgs if oldest_first else list(reversed(msgs))
        return _AsyncList(ordered[:limit] if limit is not None else ordered)

    def pins(self, *, limit=50, before=None, oldest_first=False):
        return _AsyncList(self._pins if limit is None else self._pins[:limit])

    async def send(self, content=None, *, embed=None):
        self.sent.append({"content": content, "embed": embed})


class FakeGuild:
    def __init__(self, gid: int, channels=(), threads=()) -> None:
        self.id = gid
        self.channels = list(channels)
        self.text_channels = list(channels)
        self.threads = list(threads)

    def get_channel(self, cid: int):
        return next((c for c in self.channels if c.id == cid), None)


class FakeResponse:
    def __init__(self) -> None:
        self._done = False
        self.deferred: Optional[Dict] = None
        self.sent: List[Dict] = []

    def is_done(self) -> bool:
        return self._done

    async def defer(self, *, thinking: bool = False, ephemeral: bool = False) -> None:
        self._done = True
        self.deferred = {"thinking": thinking, "ephemeral": ephemeral}

    async def send_message(self, content=None, *, embed=None, ephemeral: bool = False) -> None:
        self._done = True
        self.sent.append({"content": content, "embed": embed, "ephemeral": ephemeral})


class FakeFollowup:
    def __init__(self) -> None:
        self.sent: List[Dict] = []

    async def send(self, content=None, *, embed=None, ephemeral: bool = False) -> None:
        self.sent.append({"content": content, "embed": embed, "ephemeral": ephemeral})


class FakeInteraction:
    def __init__(self, guild: Optional[FakeGuild], channel: FakeChannel, user: FakeUser) -> None:
        self.guild = guild
        self.guild_id = guild.id if guild else None
        self.channel = channel
        self.channel_id = channel.id
        self.user = user
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.command = None

    @property
    def replies(self) -> List[Dict]:
        """Everything sent back, in order (initial response + followups)."""
        return self.response.sent + self.followup.sent

    @property
    def text(self) -> str:
        return "\n".join(r["content"] or "" for r in self.replies)


# ---------------------------------------------------------------- PDF bytes
def minimal_pdf(text: str) -> bytes:
    """A valid one-page PDF whose text layer contains ``text``."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
