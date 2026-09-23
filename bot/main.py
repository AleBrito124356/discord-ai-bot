"""Bot entrypoint and slash-command definitions.

Run with:  python -m bot.main   (from the repository root)

The bot is intentionally guild-only. Every AI command defers first (Discord shows
a "thinking" indicator) because NIM calls take longer than the 3-second initial
response budget.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from .config import SIGNUP_HINT, Settings, load_env_file, load_settings
from .middleware import GuildGuard, RateLimiter, build_error_handler, send_ephemeral
from .moderation import ModerationService
from .nim_client import NimClient, NimError
from .persistence import Database
from .personas import DEFAULT_PERSONA, PERSONAS, get_persona
from .rag import RagIndexError, RagService, extract_pdf_text
from .textutil import build_transcript, split_message

log = logging.getLogger("bot")

# Models offered as autocomplete for /model. Any NIM chat model id also works.
COMMON_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-405b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "qwen/qwen2.5-coder-32b-instruct",
]

TEXT_EXTS = (".txt", ".md", ".markdown", ".text", ".log", ".csv", ".json", ".rst")


# --------------------------------------------------------------------------- bot
class DiscordAIBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required for /summarize and moderation
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self.db = Database(settings.db_path)
        self.nim = NimClient(settings)
        self.rag = RagService(settings, self.nim)
        self.moderation = ModerationService(settings, self.nim)
        self.guard = GuildGuard(settings.allowed_guild_ids)
        self.limiter = RateLimiter(
            settings.cooldown_seconds, settings.rate_limit_per_minute
        )

    async def setup_hook(self) -> None:
        await self.db.connect()
        self.tree.on_error = build_error_handler(self.tree)
        register_commands(self)
        if self.settings.allowed_guild_ids:
            for gid in self.settings.allowed_guild_ids:
                guild = discord.Object(id=gid)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                log.info("Synced %d commands to guild %s", len(synced), gid)
        else:
            synced = await self.tree.sync()
            log.info("Synced %d global commands (rollout can take up to 1h)", len(synced))

    async def close(self) -> None:
        await self.nim.close()
        await self.db.close()
        await super().close()


# --------------------------------------------------------------- output helpers
# split_message now lives in bot.textutil (fence-aware); re-exported here.
__all__ = ["DiscordAIBot", "register_commands", "split_message", "main"]


async def send_long(interaction: discord.Interaction, text: str) -> None:
    """Send a possibly long response via followups (used after defer)."""
    for chunk in split_message(text):
        await interaction.followup.send(chunk)


# --------------------------------------------------------------- command wiring
def register_commands(bot: DiscordAIBot) -> None:
    settings = bot.settings

    async def precheck(
        interaction: discord.Interaction, command: str, *, rate_limit: bool = True
    ) -> bool:
        """Guild allowlist + rate limit gate. Sends its own reply when blocked."""
        if interaction.guild_id is None:
            await send_ephemeral(interaction, "This bot only works inside a server.")
            return False
        if not bot.guard.is_allowed(interaction.guild_id):
            await send_ephemeral(
                interaction, "This bot is not enabled for this server."
            )
            return False
        if rate_limit:
            allowed, retry = bot.limiter.check(interaction.user.id)
            if not allowed:
                await send_ephemeral(
                    interaction, f"Please wait {retry:.1f}s before trying again."
                )
                return False
        await bot.db.increment_usage(
            interaction.guild_id, interaction.user.id, command
        )
        return True

    # ------------------------------------------------------------------- /ask
    @bot.tree.command(
        name="ask", description="Ask the AI. Remembers recent chat in this channel."
    )
    @app_commands.guild_only()
    @app_commands.describe(prompt="Your question or message")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        if not await precheck(interaction, "ask"):
            return
        await interaction.response.defer(thinking=True)
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        persona = get_persona(cfg.persona)
        history = await bot.db.get_history(interaction.channel_id, cfg.history_window)
        messages = (
            [{"role": "system", "content": persona["system"]}]
            + history
            + [{"role": "user", "content": prompt}]
        )
        model = cfg.chat_model or settings.chat_model
        try:
            answer = await bot.nim.chat(messages, model=model)
        except NimError as exc:
            await interaction.followup.send(str(exc))
            return
        gid = interaction.guild_id
        await bot.db.add_history(interaction.channel_id, "user", prompt, guild_id=gid)
        await bot.db.add_history(interaction.channel_id, "assistant", answer, guild_id=gid)
        await bot.db.trim_history(interaction.channel_id, cfg.history_window * 2)
        await send_long(interaction, answer)

    # ------------------------------------------------------------- /summarize
    @bot.tree.command(
        name="summarize",
        description="Summarize the most recent messages in this channel into bullets.",
    )
    @app_commands.guild_only()
    @app_commands.describe(count="How many recent messages to read (5-200)")
    async def summarize(
        interaction: discord.Interaction,
        count: app_commands.Range[int, 5, 200] = 50,
    ) -> None:
        if not await precheck(interaction, "summarize"):
            return
        await interaction.response.defer(thinking=True)
        channel = interaction.channel
        collected = []
        async for msg in channel.history(limit=int(count)):
            if msg.author.bot:
                continue
            body = msg.clean_content.strip()
            if body:
                collected.append(f"{msg.author.display_name}: {body}")
        collected.reverse()
        if not collected:
            await interaction.followup.send("Nothing to summarize here yet.")
            return
        # Keep the NEWEST messages that fit the budget (never drop the latest).
        transcript = build_transcript(collected, settings.summarize_char_budget)
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following Discord conversation into 3-7 concise "
                    "bullet points capturing decisions, questions and action items. "
                    "Use Discord markdown bullets. Do not invent details."
                ),
            },
            {"role": "user", "content": transcript.text},
        ]
        try:
            summary = await bot.nim.chat(messages, temperature=0.3)
        except NimError as exc:
            await interaction.followup.send(str(exc))
            return
        header = transcript.header() + "\n"
        await send_long(interaction, header + summary)

    # ----------------------------------------------------------------- /image
    @bot.tree.command(
        name="image", description="Ask a question about an uploaded image (vision)."
    )
    @app_commands.guild_only()
    @app_commands.describe(
        image="The image to analyze",
        question="What do you want to know about it?",
    )
    async def image_cmd(
        interaction: discord.Interaction,
        image: discord.Attachment,
        question: str = "Describe this image in detail.",
    ) -> None:
        if not await precheck(interaction, "image"):
            return
        content_type = image.content_type or ""
        if not content_type.startswith("image/"):
            await send_ephemeral(
                interaction, "Please attach an image file (PNG, JPG, WEBP, GIF)."
            )
            return
        if image.size > 12 * 1024 * 1024:
            await send_ephemeral(
                interaction, "That image is too large (max 12 MB for vision)."
            )
            return
        await interaction.response.defer(thinking=True)
        data = await image.read()
        try:
            answer = await bot.nim.vision(question, data, content_type)
        except NimError as exc:
            await interaction.followup.send(str(exc))
            return
        await send_long(interaction, answer)

    # --------------------------------------------------------------- /persona
    @bot.tree.command(
        name="persona", description="Switch the bot's persona for this server."
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(name="Which persona to use")
    @app_commands.choices(
        name=[
            app_commands.Choice(name=meta["name"], value=key)
            for key, meta in PERSONAS.items()
        ]
    )
    async def persona_cmd(
        interaction: discord.Interaction, name: app_commands.Choice[str]
    ) -> None:
        if not await precheck(interaction, "persona", rate_limit=False):
            return
        await bot.db.set_guild_field(interaction.guild_id, "persona", name.value)
        meta = PERSONAS[name.value]
        await send_ephemeral(
            interaction,
            f"Persona set to **{meta['name']}** — {meta['description']}",
        )

    # ----------------------------------------------------------------- /model
    async def model_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        current = (current or "").lower()
        return [
            app_commands.Choice(name=m, value=m)
            for m in COMMON_MODELS
            if current in m.lower()
        ][:25]

    @bot.tree.command(
        name="model", description="View or set the chat model for this server."
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(name="NIM model id, or leave empty to see the current one")
    @app_commands.autocomplete(name=model_autocomplete)
    async def model_cmd(
        interaction: discord.Interaction, name: Optional[str] = None
    ) -> None:
        if not await precheck(interaction, "model", rate_limit=False):
            return
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        if name is None:
            active = cfg.chat_model or settings.chat_model
            await send_ephemeral(
                interaction,
                f"Current chat model: `{active}`\nDefault: `{settings.chat_model}`",
            )
            return
        if name.lower() in {"default", "reset"}:
            await bot.db.set_guild_field(interaction.guild_id, "chat_model", None)
            await send_ephemeral(
                interaction, f"Reset to the default model: `{settings.chat_model}`"
            )
            return
        await bot.db.set_guild_field(interaction.guild_id, "chat_model", name)
        await send_ephemeral(interaction, f"Chat model set to `{name}`.")

    # ---------------------------------------------------------------- /forget
    @bot.tree.command(
        name="forget",
        description="Wipe the bot's memory of this channel's conversation.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.checks.has_permissions(manage_messages=True)
    async def forget(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "forget", rate_limit=False):
            return
        removed = await bot.db.clear_history(interaction.channel_id)
        await send_ephemeral(
            interaction,
            f"Cleared {removed} stored message(s) of context for this channel.",
        )

    # ------------------------------------------------------------------ /help
    @bot.tree.command(name="help", description="How to use this bot.")
    @app_commands.guild_only()
    async def help_cmd(interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="discord-ai-bot",
            description="AI chat, summaries, vision and docs — on free NVIDIA NIM.",
            colour=discord.Color.blurple(),
        )
        embed.add_field(
            name="Everyone",
            value=(
                "`/ask` — chat with per-channel memory\n"
                "`/summarize` — bullet-summary of recent messages\n"
                "`/image` — ask about an uploaded image\n"
                "`/docs ask` — answer from the server's ingested docs\n"
                "`/docs status` — how many doc chunks are indexed\n"
                "`/help` — this message"
            ),
            inline=False,
        )
        embed.add_field(
            name="Moderators (Manage Server)",
            value=(
                "`/persona` — switch the bot's persona\n"
                "`/model` — view or set the chat model\n"
                "`/forget` — wipe this channel's memory (Manage Messages)\n"
                "`/docs ingest` — index the docs channel\n"
                "`/config …` — mod channel, docs channel, moderation, wipe"
            ),
            inline=False,
        )
        embed.set_footer(
            text="Moderation is assist-only: the bot suggests, humans decide."
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -------------------------------------------------------------- docs group
    docs = app_commands.Group(
        name="docs",
        description="Retrieval over the server's ingested documents",
        guild_only=True,
    )

    @docs.command(name="ask", description="Answer a question from the server's docs.")
    @app_commands.describe(question="What do you want to know?")
    async def docs_ask(interaction: discord.Interaction, question: str) -> None:
        if not await precheck(interaction, "docs_ask"):
            return
        await interaction.response.defer(thinking=True)
        try:
            result = await bot.rag.answer(interaction.guild_id, question)
        except (NimError, RagIndexError) as exc:
            await interaction.followup.send(str(exc))
            return
        # Only the excerpts the answer actually cites are listed as sources.
        await send_long(interaction, result.formatted())

    @docs.command(
        name="ingest", description="Index pinned messages and files in the docs channel."
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def docs_ingest(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "docs_ingest", rate_limit=False):
            return
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        if not cfg.docs_channel_id:
            await send_ephemeral(
                interaction,
                "Set a docs channel first: `/config docs-channel #your-docs`.",
            )
            return
        channel = interaction.guild.get_channel(cfg.docs_channel_id)
        if channel is None:
            await send_ephemeral(
                interaction, "The configured docs channel no longer exists."
            )
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        documents = await _collect_documents(channel)
        if not documents:
            await interaction.followup.send(
                "Found no readable pins or .txt/.md/.pdf attachments in that channel.",
                ephemeral=True,
            )
            return
        try:
            report = await bot.rag.ingest_documents(interaction.guild_id, documents)
        except NimError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        msg = (
            f"Indexed **{report.chunks}** chunks from **{report.documents}** "
            f"document(s) in {channel.mention}."
        )
        if report.skipped:
            msg += f"\nSkipped {len(report.skipped)} empty source(s)."
        await interaction.followup.send(msg, ephemeral=True)

    @docs.command(name="status", description="Show how many doc chunks are indexed.")
    async def docs_status(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "docs_status", rate_limit=False):
            return
        status = bot.rag.status(interaction.guild_id)
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        channel = (
            f"<#{cfg.docs_channel_id}>" if cfg.docs_channel_id else "not set"
        )
        lines = [
            f"Indexed chunks: **{status['chunks']}** from {status['sources']} source(s)",
            f"Docs channel: {channel}",
        ]
        if status["embed_model"]:
            lines.append(f"Embedding model: `{status['embed_model']}` ({status['dim']}-d)")
        if status["error"]:
            lines.append(f"Problem: {status['error']}")
        await send_ephemeral(interaction, "\n".join(lines))

    bot.tree.add_command(docs)

    # ------------------------------------------------------------ config group
    config = app_commands.Group(
        name="config",
        description="Server configuration (Manage Server only)",
        guild_only=True,
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @config.command(name="show", description="Show the current server configuration.")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def config_show(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "config_show", rate_limit=False):
            return
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        store = bot.rag.store_for(interaction.guild_id)
        allow = await bot.db.get_allowlist(interaction.guild_id)
        embed = discord.Embed(
            title="Server configuration", colour=discord.Color.blurple()
        )
        embed.add_field(name="Persona", value=cfg.persona, inline=True)
        embed.add_field(
            name="Model",
            value=cfg.chat_model or f"default ({settings.chat_model})",
            inline=True,
        )
        embed.add_field(name="History window", value=str(cfg.history_window), inline=True)
        embed.add_field(
            name="Docs channel",
            value=f"<#{cfg.docs_channel_id}>" if cfg.docs_channel_id else "not set",
            inline=True,
        )
        embed.add_field(name="Indexed chunks", value=str(store.size), inline=True)
        embed.add_field(
            name="Mod channel",
            value=f"<#{cfg.mod_channel_id}>" if cfg.mod_channel_id else "not set",
            inline=True,
        )
        embed.add_field(
            name="Moderation",
            value=f"{'on' if cfg.moderation_on else 'off'} (threshold {cfg.mod_threshold:.2f})",
            inline=True,
        )
        embed.add_field(
            name="Mod allowlist",
            value=str(len(allow)) + " entr" + ("y" if len(allow) == 1 else "ies"),
            inline=True,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @config.command(
        name="mod-channel", description="Set the channel that receives mod advisories."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(channel="Text channel for moderation advisories")
    async def config_mod_channel(
        interaction: discord.Interaction, channel: discord.TextChannel
    ) -> None:
        if not await precheck(interaction, "config_mod_channel", rate_limit=False):
            return
        await bot.db.set_guild_field(
            interaction.guild_id, "mod_channel_id", channel.id
        )
        await send_ephemeral(
            interaction, f"Moderation advisories will be posted to {channel.mention}."
        )

    @config.command(
        name="docs-channel", description="Set the channel to ingest docs from."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(channel="Text channel whose pins/files become the knowledge base")
    async def config_docs_channel(
        interaction: discord.Interaction, channel: discord.TextChannel
    ) -> None:
        if not await precheck(interaction, "config_docs_channel", rate_limit=False):
            return
        await bot.db.set_guild_field(
            interaction.guild_id, "docs_channel_id", channel.id
        )
        await send_ephemeral(
            interaction,
            f"Docs channel set to {channel.mention}. Run `/docs ingest` to index it.",
        )

    @config.command(
        name="moderation", description="Turn assist-only moderation on or off."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(enabled="Enable moderation advisories")
    async def config_moderation(
        interaction: discord.Interaction, enabled: bool
    ) -> None:
        if not await precheck(interaction, "config_moderation", rate_limit=False):
            return
        cfg = await bot.db.get_guild_config(interaction.guild_id)
        if enabled and not cfg.mod_channel_id:
            await send_ephemeral(
                interaction,
                "Set a mod channel first with `/config mod-channel`.",
            )
            return
        await bot.db.set_guild_field(interaction.guild_id, "moderation_on", enabled)
        state = "enabled" if enabled else "disabled"
        note = (
            " The bot will only *suggest*; it never bans, kicks or deletes."
            if enabled
            else ""
        )
        await send_ephemeral(interaction, f"Moderation {state}.{note}")

    @config.command(
        name="threshold", description="Set the moderation sensitivity (0.1-0.95)."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(value="Lower = more sensitive, more advisories")
    async def config_threshold(
        interaction: discord.Interaction,
        value: app_commands.Range[float, 0.1, 0.95],
    ) -> None:
        if not await precheck(interaction, "config_threshold", rate_limit=False):
            return
        await bot.db.set_guild_field(
            interaction.guild_id, "mod_threshold", float(value)
        )
        await send_ephemeral(
            interaction, f"Moderation threshold set to {float(value):.2f}."
        )

    @config.command(
        name="history-window", description="How many past messages /ask remembers."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(size="Number of prior messages kept as context (2-40)")
    async def config_history_window(
        interaction: discord.Interaction,
        size: app_commands.Range[int, 2, 40],
    ) -> None:
        if not await precheck(interaction, "config_history_window", rate_limit=False):
            return
        await bot.db.set_guild_field(
            interaction.guild_id, "history_window", int(size)
        )
        await send_ephemeral(
            interaction, f"History window set to {int(size)} messages."
        )

    @config.command(
        name="mod-allow", description="Exempt a user or role from moderation checks."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(
        member="User to exempt", role="Role to exempt (either field)"
    )
    async def config_mod_allow(
        interaction: discord.Interaction,
        member: Optional[discord.Member] = None,
        role: Optional[discord.Role] = None,
    ) -> None:
        if not await precheck(interaction, "config_mod_allow", rate_limit=False):
            return
        if member is None and role is None:
            await send_ephemeral(interaction, "Provide a member or a role to exempt.")
            return
        added = []
        if member is not None:
            await bot.db.add_allowlist(interaction.guild_id, member.id, "user")
            added.append(member.mention)
        if role is not None:
            await bot.db.add_allowlist(interaction.guild_id, role.id, "role")
            added.append(role.mention)
        await send_ephemeral(
            interaction, "Exempted from moderation: " + ", ".join(added)
        )

    @config.command(
        name="mod-unallow", description="Remove a user or role from the exempt list."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(
        member="User to remove", role="Role to remove (either field)"
    )
    async def config_mod_unallow(
        interaction: discord.Interaction,
        member: Optional[discord.Member] = None,
        role: Optional[discord.Role] = None,
    ) -> None:
        if not await precheck(interaction, "config_mod_unallow", rate_limit=False):
            return
        target_id = member.id if member else (role.id if role else None)
        if target_id is None:
            await send_ephemeral(interaction, "Provide a member or a role to remove.")
            return
        removed = await bot.db.remove_allowlist(interaction.guild_id, target_id)
        await send_ephemeral(
            interaction,
            "Removed from the exempt list." if removed else "That was not on the list.",
        )

    @config.command(
        name="wipe", description="Delete ALL data this bot stored for this server."
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(confirm="Type true to confirm this is irreversible")
    async def config_wipe(interaction: discord.Interaction, confirm: bool) -> None:
        if not await precheck(interaction, "config_wipe", rate_limit=False):
            return
        if not confirm:
            await send_ephemeral(
                interaction,
                "Nothing wiped. Re-run with `confirm: True` to erase all stored data.",
            )
            return
        # Memory is matched by guild_id (threads, forums, deleted channels too);
        # the channel ids only catch rows stored before schema v2.
        guild = interaction.guild
        channel_ids = [c.id for c in guild.channels] + [t.id for t in guild.threads]
        await bot.db.clear_guild_data(interaction.guild_id, channel_ids)
        bot.rag.forget(interaction.guild_id)
        await send_ephemeral(
            interaction,
            "Wiped: channel memory, usage counters, config and the docs index.",
        )

    bot.tree.add_command(config)

    # ---------------------------------------------------- moderation listener
    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        if not bot.guard.is_allowed(message.guild.id):
            return
        cfg = await bot.db.get_guild_config(message.guild.id)
        if not cfg.moderation_on or not cfg.mod_channel_id:
            return
        if message.channel.id == cfg.mod_channel_id:
            return
        allow = await bot.db.get_allowlist(message.guild.id)
        allow_ids = {tid for tid, _ in allow}
        if message.author.id in allow_ids:
            return
        author_role_ids = {r.id for r in getattr(message.author, "roles", [])}
        if author_role_ids & allow_ids:
            return
        try:
            verdict = await bot.moderation.review_message(message, cfg.mod_threshold)
        except Exception:  # noqa: BLE001 - a moderation hiccup must not crash on_message
            log.exception("Moderation review failed")
            return
        if verdict is None:
            return
        mod_channel = message.guild.get_channel(cfg.mod_channel_id)
        if mod_channel is None:
            return
        try:
            await mod_channel.send(
                embed=bot.moderation.build_advisory_embed(message, verdict)
            )
        except discord.HTTPException:
            log.exception("Could not post moderation advisory")

    @bot.event
    async def on_ready() -> None:
        log.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "?")
        log.info("In %d guild(s)", len(bot.guilds))


async def _collect_documents(channel: discord.TextChannel):
    """Gather (source_label, text) tuples from a docs channel.

    Reads every pinned message (its text and any .txt/.md/.pdf attachments) plus
    document attachments found in the channel's recent history.
    """
    documents = []
    seen_attachment_ids = set()

    async def add_attachment(att: discord.Attachment, label_prefix: str) -> None:
        if att.id in seen_attachment_ids:
            return
        seen_attachment_ids.add(att.id)
        name = att.filename.lower()
        text = ""
        if name.endswith(TEXT_EXTS):
            data = await att.read()
            text = data.decode("utf-8", errors="replace")
        elif name.endswith(".pdf"):
            data = await att.read()
            # pypdf is CPU-bound: keep it off the event loop (gateway heartbeat).
            text = await asyncio.to_thread(extract_pdf_text, data)
        if text.strip():
            documents.append((f"{label_prefix}{att.filename}", text))

    # ``await channel.pins()`` is deprecated and stops at 50; iterate them all.
    pins = []
    try:
        async for msg in channel.pins(limit=None):
            pins.append(msg)
    except discord.HTTPException:
        log.warning("Could not read pins in #%s", getattr(channel, "name", "?"))
    for msg in pins:
        if msg.content.strip():
            documents.append(
                (f"pin by {msg.author.display_name}", msg.content.strip())
            )
        for att in msg.attachments:
            await add_attachment(att, "pinned: ")

    try:
        async for msg in channel.history(limit=500):
            for att in msg.attachments:
                await add_attachment(att, "file: ")
    except discord.HTTPException:
        pass

    return documents


# --------------------------------------------------------------------- main()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    load_env_file()  # <repo>/.env only; never a parent directory's .env
    settings = load_settings()

    if not settings.discord_token:
        print(
            "DISCORD_BOT_TOKEN is not set.\n"
            "Create a bot at https://discord.com/developers/applications, then copy "
            "its token into your .env file. See docs/setup.md for the walkthrough."
        )
        sys.exit(1)
    if not settings.has_nim_key():
        print("NVIDIA_API_KEY is not set (or is still the placeholder).\n" + SIGNUP_HINT)
        sys.exit(1)

    bot = DiscordAIBot(settings)
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
