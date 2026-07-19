"""Cross-cutting concerns: cooldowns, rate limits, guild allowlist, errors.

None of this touches the network. It is pure in-memory bookkeeping plus a single
error handler that turns any command failure into a friendly, ephemeral reply and
a log line.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Set, Tuple

import discord
from discord import app_commands

log = logging.getLogger("middleware")


class RateLimiter:
    """Per-user cooldown plus a sliding-window rate limit.

    * ``cooldown_seconds`` — minimum gap between two calls by the same user.
    * ``per_minute`` — max calls per user in any rolling 60-second window.
    """

    def __init__(self, cooldown_seconds: float, per_minute: int) -> None:
        self.cooldown = max(0.0, cooldown_seconds)
        self.per_minute = max(1, per_minute)
        self._last_call: Dict[int, float] = {}
        self._windows: Dict[int, Deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> Tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` and record the call if allowed."""
        now = time.monotonic()

        last = self._last_call.get(user_id)
        if last is not None and (now - last) < self.cooldown:
            return False, round(self.cooldown - (now - last), 1)

        window = self._windows[user_id]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.per_minute:
            retry = round(60.0 - (now - window[0]), 1)
            return False, max(retry, 0.1)

        window.append(now)
        self._last_call[user_id] = now
        return True, 0.0


class GuildGuard:
    """Allowlist gate. Empty allowlist means 'every guild is allowed'."""

    def __init__(self, allowed_guild_ids: Set[int]) -> None:
        self._allowed = set(allowed_guild_ids)

    @property
    def restricted(self) -> bool:
        return bool(self._allowed)

    def is_allowed(self, guild_id: Optional[int]) -> bool:
        if not self._allowed:
            return True
        return guild_id in self._allowed


async def send_ephemeral(interaction: discord.Interaction, content: str) -> None:
    """Reply privately whether or not the interaction was already deferred."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True)
        else:
            await interaction.response.send_message(content, ephemeral=True)
    except discord.HTTPException:
        log.exception("Failed to deliver ephemeral message")


def build_error_handler(tree: app_commands.CommandTree):
    """Return an on_error coroutine that apologizes to the user and logs."""

    async def on_error(
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        if isinstance(error, app_commands.CommandOnCooldown):
            await send_ephemeral(
                interaction,
                f"Slow down a moment — try again in {error.retry_after:.0f}s.",
            )
            return
        if isinstance(error, app_commands.MissingPermissions):
            await send_ephemeral(
                interaction,
                "You do not have permission to use that command here.",
            )
            return
        if isinstance(error, app_commands.CheckFailure):
            await send_ephemeral(
                interaction,
                "That command is not available in this server or channel.",
            )
            return

        # Unwrap invoke errors for a cleaner log while keeping the reply generic.
        original = getattr(error, "original", error)
        command = interaction.command.qualified_name if interaction.command else "?"
        log.error("Unhandled error in /%s: %s", command, original, exc_info=original)
        await send_ephemeral(
            interaction,
            "Sorry — something went wrong handling that. The error has been logged.",
        )

    return on_error
