"""Discord-free business logic shared by the bot and the terminal simulator.

Every feature the bot offers lives here as a small service that takes plain ids
and strings and returns a :class:`Reply`. ``bot/main.py`` only adapts Discord
interactions to these calls, and ``bot/cli.py`` drives the very same code from a
terminal. That keeps the logic testable without Discord objects.

    async with BotCore(settings) as core:          # connects SQLite
        reply = await core.ask.ask(guild_id=1, channel_id=2, prompt="hi")
        print(reply.text)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .imaging import ImageError, normalize_image
from .moderation import ModerationService, ReviewResult
from .nim_client import NimError, make_nim_client
from .persistence import Database, GuildConfig, UsageStats
from .personas import get_persona
from .prompts import (
    EXPLAIN_MESSAGE_TEMPLATE,
    IMAGE_IN_MESSAGE_TEMPLATE,
    SUMMARIZE_SYSTEM_PROMPT,
)
from .rag import IngestReport, RagAnswer, RagIndexError, RagService, extract_pdf_text
from .textutil import Transcript, build_transcript

log = logging.getLogger("services")

TEXT_EXTS = (".txt", ".md", ".markdown", ".text", ".log", ".csv", ".json", ".rst")
DOC_EXTS = TEXT_EXTS + (".pdf",)


@dataclass
class Reply:
    """What a command answers. ``note`` is a short footnote (e.g. image resized)."""

    text: str
    ok: bool = True
    note: str = ""
    data: Dict[str, object] = field(default_factory=dict)

    def render(self) -> str:
        """Discord markdown: the note becomes small '-#' subtext under the text."""
        return f"{self.text}\n-# {self.note}" if self.note else self.text


# --------------------------------------------------------------- documents
def is_document(filename: str) -> bool:
    return filename.lower().endswith(DOC_EXTS)


async def document_text(filename: str, data: bytes) -> str:
    """Decode a .txt/.md/... or .pdf attachment. PDFs are parsed off the loop."""
    name = filename.lower()
    if name.endswith(TEXT_EXTS):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        # pypdf is CPU-bound: keep it off the event loop (gateway heartbeat).
        return await asyncio.to_thread(extract_pdf_text, data)
    return ""


# ---------------------------------------------------------------- services
class AskService:
    def __init__(self, core: "BotCore") -> None:
        self._core = core

    async def ask(self, guild_id: int, channel_id: int, prompt: str) -> Reply:
        """Chat with persona + this channel's memory; store the exchange."""
        core = self._core
        cfg = await core.db.get_guild_config(guild_id)
        persona = get_persona(cfg.persona)
        history = await core.db.get_history(channel_id, cfg.history_window)
        messages = (
            [{"role": "system", "content": persona["system"]}]
            + history
            + [{"role": "user", "content": prompt}]
        )
        try:
            answer = await core.nim.chat(messages, model=core.chat_model_for(cfg))
        except NimError as exc:
            return Reply(str(exc), ok=False)
        await core.db.add_history(channel_id, "user", prompt, guild_id=guild_id)
        await core.db.add_history(channel_id, "assistant", answer, guild_id=guild_id)
        await core.db.trim_history(channel_id, cfg.history_window * 2)
        return Reply(answer, data={"history_used": len(history)})

    async def explain(self, guild_id: int, author: str, content: str) -> Reply:
        """Explain or answer someone else's message (context menu). No memory."""
        core = self._core
        cfg = await core.db.get_guild_config(guild_id)
        persona = get_persona(cfg.persona)
        prompt = EXPLAIN_MESSAGE_TEMPLATE.format(author=author, content=content[:4000])
        try:
            answer = await core.nim.chat(
                [
                    {"role": "system", "content": persona["system"]},
                    {"role": "user", "content": prompt},
                ],
                model=core.chat_model_for(cfg),
            )
        except NimError as exc:
            return Reply(str(exc), ok=False)
        return Reply(answer)


class SummarizeService:
    def __init__(self, core: "BotCore") -> None:
        self._core = core

    async def summarize(
        self,
        lines: Sequence[str],
        *,
        keep: str = "newest",
        anchor: Optional[str] = None,
    ) -> Reply:
        """Summarize ``lines`` ("name: text", oldest first).

        ``keep="newest"`` (``/summarize``) drops the oldest lines when over the
        character budget; ``keep="oldest"`` (*Summarize from here*) drops the
        latest ones so the selected starting message is always included.
        """
        if not lines:
            return Reply("Nothing to summarize here yet.", ok=False)
        transcript = build_transcript(
            lines, self._core.settings.summarize_char_budget, keep=keep
        )
        messages = [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": transcript.text},
        ]
        try:
            summary = await self._core.nim.chat(messages, temperature=0.3)
        except NimError as exc:
            return Reply(str(exc), ok=False)
        return Reply(
            f"{transcript.header(anchor=anchor)}\n{summary}",
            data={"included": transcript.included, "total": transcript.total},
        )


class DocsService:
    def __init__(self, core: "BotCore") -> None:
        self._core = core

    async def ingest(
        self, guild_id: int, documents: Sequence[Tuple[str, str]], *, where: str = ""
    ) -> Reply:
        if not documents:
            return Reply(
                "Found no readable pins or .txt/.md/.pdf documents to index.", ok=False
            )
        try:
            report: IngestReport = await self._core.rag.ingest_documents(guild_id, documents)
        except NimError as exc:
            return Reply(str(exc), ok=False)
        text = (
            f"Indexed **{report.chunks}** chunks from **{report.documents}** "
            f"document(s){' in ' + where if where else ''}."
        )
        if report.skipped:
            text += f"\nSkipped {len(report.skipped)} empty source(s)."
        return Reply(text, ok=report.chunks > 0, data={"report": report})

    async def ask(self, guild_id: int, question: str) -> Reply:
        try:
            result: RagAnswer = await self._core.rag.answer(guild_id, question)
        except (NimError, RagIndexError) as exc:
            return Reply(str(exc), ok=False)
        return Reply(result.formatted(), ok=result.indexed, data={"answer": result})

    def status_lines(self, guild_id: int, docs_channel: Optional[str] = None) -> List[str]:
        status = self._core.rag.status(guild_id)
        lines = [f"Indexed chunks: **{status['chunks']}** from {status['sources']} source(s)"]
        if docs_channel is not None:
            lines.append(f"Docs channel: {docs_channel}")
        if status["embed_model"]:
            lines.append(f"Embedding model: `{status['embed_model']}` ({status['dim']}-d)")
            if status["embed_model"] != status["current_embed_model"]:
                lines.append(
                    f"Warning: the bot now embeds with `{status['current_embed_model']}`; "
                    "re-run `/docs ingest`."
                )
        if status["error"]:
            lines.append(f"Problem: {status['error']}")
        return lines


class VisionService:
    def __init__(self, core: "BotCore") -> None:
        self._core = core

    async def describe(self, data: bytes, mime: Optional[str], question: str) -> Reply:
        """Normalise the image to fit the vision model, then ask about it."""
        settings = self._core.settings
        try:
            image = await asyncio.to_thread(
                normalize_image,
                data,
                mime,
                max_side=settings.image_max_side,
                max_b64=settings.image_max_b64,
            )
        except ImageError as exc:
            return Reply(str(exc), ok=False)
        try:
            answer = await self._core.nim.vision(question, image.data, image.mime)
        except NimError as exc:
            return Reply(str(exc), ok=False)
        return Reply(answer, note=image.note, data={"image": image})

    async def describe_in_message(
        self, data: bytes, mime: Optional[str], author: str, caption: str
    ) -> Reply:
        caption_part = f' with the text "{caption[:500]}"' if caption.strip() else ""
        prompt = IMAGE_IN_MESSAGE_TEMPLATE.format(author=author, caption=caption_part)
        return await self.describe(data, mime, prompt)


class ModerationFlow:
    """The per-message gate: opt-in, mod channel, exemptions, then review."""

    def __init__(self, core: "BotCore") -> None:
        self._core = core

    @staticmethod
    def is_exempt(author_id: int, role_ids: Iterable[int], allowlist) -> bool:
        allow_ids = {tid for tid, _ in allowlist}
        return author_id in allow_ids or bool(set(role_ids) & allow_ids)

    async def review(
        self,
        guild_id: int,
        channel_id: int,
        author_id: int,
        role_ids: Iterable[int],
        content: str,
        mention_count: int,
    ) -> Tuple[Optional[GuildConfig], Optional[ReviewResult]]:
        """Return ``(config, result)``; result is None when the gate skips it."""
        cfg = await self._core.db.get_guild_config(guild_id)
        if not cfg.moderation_on or not cfg.mod_channel_id:
            return cfg, None
        if channel_id == cfg.mod_channel_id:
            return cfg, None
        allow = await self._core.db.get_allowlist(guild_id)
        if self.is_exempt(author_id, role_ids, allow):
            return cfg, None
        result = await self._core.moderation.review_text(
            content, mention_count, cfg.mod_threshold
        )
        return cfg, result


def stats_lines(stats: UsageStats, label_user=lambda uid: f"<@{uid}>") -> Dict[str, str]:
    """Human-readable blocks for /stats and the CLI ``stats`` command."""
    commands = "\n".join(f"`/{name.replace('_', ' ')}` — {n}" for name, n in stats.per_command)
    users = "\n".join(
        f"{i}. {label_user(uid)} — {n}" for i, (uid, n) in enumerate(stats.top_users, 1)
    )
    return {
        "total": f"{stats.total} command(s) recorded",
        "commands": commands or "No commands recorded yet.",
        "users": users or "Nobody yet.",
        "memory": f"{stats.stored_messages} stored /ask message(s)",
    }


# -------------------------------------------------------------------- core
class BotCore:
    """Owns settings, storage, the model backend and every service."""

    def __init__(self, settings, nim=None) -> None:
        self.settings = settings
        self.nim = nim if nim is not None else make_nim_client(settings)
        self.db = Database(settings.db_path)
        self.rag = RagService(settings, self.nim)
        self.moderation = ModerationService(settings, self.nim)
        self.ask = AskService(self)
        self.summarizer = SummarizeService(self)
        self.docs = DocsService(self)
        self.vision = VisionService(self)
        self.mod_flow = ModerationFlow(self)

    @property
    def offline(self) -> bool:
        return bool(getattr(self.nim, "offline", False))

    def chat_model_for(self, cfg: GuildConfig) -> str:
        return cfg.chat_model or self.settings.chat_model

    async def start(self) -> None:
        self.settings.ensure_dirs()
        await self.db.connect()

    async def close(self) -> None:
        await self.nim.close()
        await self.db.close()

    async def __aenter__(self) -> "BotCore":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


__all__ = [
    "BotCore",
    "Reply",
    "Transcript",
    "document_text",
    "is_document",
    "stats_lines",
]
