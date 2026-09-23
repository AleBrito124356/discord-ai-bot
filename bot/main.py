"""Bot entrypoint and slash-command definitions.

Run with:  python -m bot.main   (from the repository root)

The bot is intentionally guild-only. Every AI command defers first (Discord shows
a "thinking" indicator) because NIM calls take longer than the 3-second initial
response budget.

The handlers here are thin adapters: they read what Discord gives them, call a
service from ``bot/services.py`` with plain ids and strings, and render the
:class:`~bot.services.Reply`. The same services power ``python -m bot.cli``.
"""
from __future__ import annotations

import logging
import sys
from typing import List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

from .config import SIGNUP_HINT, Settings, load_env_file, load_settings
from .middleware import GuildGuard, RateLimiter, build_error_handler, send_ephemeral
from .personas import PERSONAS
from .services import BotCore, document_text, is_document
from .textutil import split_message

__all__ = ["DiscordAIBot", "register_commands", "split_message", "main"]

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

# Anything bigger is refused before download; smaller images are converted and
# downscaled to what the vision model accepts (see bot/imaging.py).
MAX_IMAGE_UPLOAD = 25 * 1024 * 1024
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- bot
class DiscordAIBot(commands.Bot):
    def __init__(self, settings: Settings, *, nim=None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required for /summarize and moderation
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self.core = BotCore(settings, nim=nim)
        # Shortcuts kept for backwards compatibility with 1.0 code.
        self.db = self.core.db
        self.nim = self.core.nim
        self.rag = self.core.rag
        self.moderation = self.core.moderation
        self.guard = GuildGuard(settings.allowed_guild_ids)
        self.limiter = RateLimiter(
            settings.cooldown_seconds, settings.rate_limit_per_minute
        )

    async def setup_hook(self) -> None:
        await self.core.start()
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
        await self.core.close()
        await super().close()


# --------------------------------------------------------------- output helpers
async def send_long(
    interaction: discord.Interaction, text: str, *, ephemeral: bool = False
) -> None:
    """Send a possibly long response via followups (used after defer)."""
    for chunk in split_message(text):
        await interaction.followup.send(chunk, ephemeral=ephemeral)


def _is_image(attachment: discord.Attachment) -> bool:
    content_type = (attachment.content_type or "").lower()
    return content_type.startswith("image/") or attachment.filename.lower().endswith(IMAGE_EXTS)


# --------------------------------------------------------------- command wiring
def register_commands(bot: DiscordAIBot) -> None:
    settings = bot.settings
    core = bot.core

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
        await core.db.increment_usage(
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
        reply = await core.ask.ask(interaction.guild_id, interaction.channel_id, prompt)
        await send_long(interaction, reply.render())

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
        lines = []
        async for msg in interaction.channel.history(limit=int(count)):
            line = _transcript_line(msg)
            if line:
                lines.append(line)
        lines.reverse()  # history() is newest-first; transcripts are oldest-first
        reply = await core.summarizer.summarize(lines)
        await send_long(interaction, reply.render())

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
        if not _is_image(image):
            await send_ephemeral(
                interaction, "Please attach an image file (PNG, JPG, WEBP, GIF, BMP)."
            )
            return
        if image.size > MAX_IMAGE_UPLOAD:
            await send_ephemeral(
                interaction,
                f"That image is too large (max {MAX_IMAGE_UPLOAD // (1024 * 1024)} MB).",
            )
            return
        await interaction.response.defer(thinking=True)
        data = await image.read()
        reply = await core.vision.describe(data, image.content_type, question)
        await send_long(interaction, reply.render())

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
        await core.db.set_guild_field(interaction.guild_id, "persona", name.value)
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
        cfg = await core.db.get_guild_config(interaction.guild_id)
        if name is None:
            active = cfg.chat_model or settings.chat_model
            backend = " (offline backend)" if core.offline else ""
            await send_ephemeral(
                interaction,
                f"Current chat model: `{active}`{backend}\nDefault: `{settings.chat_model}`",
            )
            return
        if name.lower() in {"default", "reset"}:
            await core.db.set_guild_field(interaction.guild_id, "chat_model", None)
            await send_ephemeral(
                interaction, f"Reset to the default model: `{settings.chat_model}`"
            )
            return
        await core.db.set_guild_field(interaction.guild_id, "chat_model", name)
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
        removed = await core.db.clear_history(interaction.channel_id)
        await send_ephemeral(
            interaction,
            f"Cleared {removed} stored message(s) of context for this channel.",
        )

    # ------------------------------------------------------------------ /help
    @bot.tree.command(name="help", description="How to use this bot.")
    @app_commands.guild_only()
    async def help_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(embed=build_help_embed(core.offline), ephemeral=True)

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
        # Only the excerpts the answer actually cites are listed as sources.
        reply = await core.docs.ask(interaction.guild_id, question)
        await send_long(interaction, reply.render())

    @docs.command(
        name="ingest", description="Index pinned messages and files in the docs channel."
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def docs_ingest(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "docs_ingest", rate_limit=False):
            return
        cfg = await core.db.get_guild_config(interaction.guild_id)
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
        documents = await collect_documents(channel)
        reply = await core.docs.ingest(
            interaction.guild_id, documents, where=channel.mention
        )
        await interaction.followup.send(reply.render(), ephemeral=True)

    @docs.command(name="status", description="Show how many doc chunks are indexed.")
    async def docs_status(interaction: discord.Interaction) -> None:
        if not await precheck(interaction, "docs_status", rate_limit=False):
            return
        cfg = await core.db.get_guild_config(interaction.guild_id)
        channel = f"<#{cfg.docs_channel_id}>" if cfg.docs_channel_id else "not set"
        await send_ephemeral(
            interaction, "\n".join(core.docs.status_lines(interaction.guild_id, channel))
        )

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
        cfg = await core.db.get_guild_config(interaction.guild_id)
        status = core.rag.status(interaction.guild_id)
        allow = await core.db.get_allowlist(interaction.guild_id)
        embed = discord.Embed(
            title="Server configuration", colour=discord.Color.blurple()
        )
        embed.add_field(name="Persona", value=cfg.persona, inline=True)
        embed.add_field(
            name="Model",
            value=cfg.chat_model or f"default ({settings.chat_model})",
            inline=True,
        )
        embed.add_field(
            name="History window", value=f"{cfg.history_window} messages", inline=True
        )
        embed.add_field(
            name="Docs channel",
            value=f"<#{cfg.docs_channel_id}>" if cfg.docs_channel_id else "not set",
            inline=True,
        )
        embed.add_field(name="Indexed chunks", value=str(status["chunks"]), inline=True)
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
        if core.offline:
            embed.set_footer(text="Running on the offline backend (no NVIDIA NIM calls).")
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
        await core.db.set_guild_field(interaction.guild_id, "mod_channel_id", channel.id)
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
        await core.db.set_guild_field(interaction.guild_id, "docs_channel_id", channel.id)
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
        cfg = await core.db.get_guild_config(interaction.guild_id)
        if enabled and not cfg.mod_channel_id:
            await send_ephemeral(
                interaction,
                "Set a mod channel first with `/config mod-channel`.",
            )
            return
        await core.db.set_guild_field(interaction.guild_id, "moderation_on", enabled)
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
        await core.db.set_guild_field(interaction.guild_id, "mod_threshold", float(value))
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
        await core.db.set_guild_field(interaction.guild_id, "history_window", int(size))
        await send_ephemeral(
            interaction,
            f"History window set to {int(size)} messages "
            f"(about {int(size) // 2} question/answer exchanges).",
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
            await core.db.add_allowlist(interaction.guild_id, member.id, "user")
            added.append(member.mention)
        if role is not None:
            await core.db.add_allowlist(interaction.guild_id, role.id, "role")
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
        removed = await core.db.remove_allowlist(interaction.guild_id, target_id)
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
        removed = await core.db.clear_guild_data(interaction.guild_id, channel_ids)
        core.rag.forget(interaction.guild_id)
        await send_ephemeral(
            interaction,
            f"Wiped: {removed['channel_history']} memory message(s) across all channels "
            "and threads, usage counters, the moderation allowlist, config and the "
            "docs index.",
        )

    bot.tree.add_command(config)

    # ---------------------------------------------------- moderation listener
    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        if not bot.guard.is_allowed(message.guild.id):
            return
        try:
            cfg, result = await core.mod_flow.review(
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                author_id=message.author.id,
                role_ids=[r.id for r in getattr(message.author, "roles", [])],
                content=message.content,
                mention_count=len(message.mentions),
            )
        except Exception:  # noqa: BLE001 - a moderation hiccup must not crash on_message
            log.exception("Moderation review failed")
            return
        if result is None or not result.report:
            return
        mod_channel = message.guild.get_channel(cfg.mod_channel_id)
        if mod_channel is None:
            return
        try:
            await mod_channel.send(
                embed=core.moderation.build_advisory_embed(message, result.verdict)
            )
        except discord.HTTPException:
            log.exception("Could not post moderation advisory")

    @bot.event
    async def on_ready() -> None:
        log.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "?")
        log.info("In %d guild(s)", len(bot.guilds))


def build_help_embed(offline: bool = False) -> discord.Embed:
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
    footer = "Moderation is assist-only: the bot suggests, humans decide."
    if offline:
        footer += " Offline backend active: answers are extractive, not generated."
    embed.set_footer(text=footer)
    return embed


def _transcript_line(msg: discord.Message) -> Optional[str]:
    """``"Name: text"`` for a human message with text, else None."""
    if msg.author.bot:
        return None
    body = (msg.clean_content or "").strip()
    return f"{msg.author.display_name}: {body}" if body else None


async def collect_documents(channel) -> List[Tuple[str, str]]:
    """Gather (source_label, text) tuples from a docs channel.

    Reads every pinned message (its text and any .txt/.md/.pdf attachments) plus
    document attachments found in the channel's recent history.
    """
    documents: List[Tuple[str, str]] = []
    seen_attachment_ids = set()

    async def add_attachment(att: discord.Attachment, label_prefix: str) -> None:
        if att.id in seen_attachment_ids or not is_document(att.filename):
            return
        seen_attachment_ids.add(att.id)
        text = await document_text(att.filename, await att.read())
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


# Backwards-compatible name from 1.0.
_collect_documents = collect_documents


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
            "its token into your .env file. See docs/setup.md for the walkthrough.\n"
            "To try the bot without Discord at all: python -m bot.cli --help"
        )
        sys.exit(1)
    if settings.offline:
        log.warning(
            "BOT_OFFLINE is set: running on the deterministic offline backend. "
            "No NVIDIA NIM calls will be made."
        )
    elif not settings.has_nim_key():
        print("NVIDIA_API_KEY is not set (or is still the placeholder).\n" + SIGNUP_HINT)
        sys.exit(1)

    bot = DiscordAIBot(settings)
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
