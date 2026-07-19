"""SQLite persistence: guild config, per-channel memory, usage counters.

Uses ``aiosqlite`` so nothing blocks the event loop. One long-lived connection is
opened in :meth:`Database.connect` and shared for the process. All writes commit
immediately; the traffic a chat bot generates does not need batching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

from .personas import DEFAULT_PERSONA

SCHEMA = """
CREATE TABLE IF NOT EXISTS guild_config (
    guild_id        INTEGER PRIMARY KEY,
    persona         TEXT    NOT NULL DEFAULT 'assistant',
    chat_model      TEXT,
    history_window  INTEGER NOT NULL DEFAULT 12,
    mod_channel_id  INTEGER,
    docs_channel_id INTEGER,
    moderation_on   INTEGER NOT NULL DEFAULT 0,
    mod_threshold   REAL    NOT NULL DEFAULT 0.6
);

CREATE TABLE IF NOT EXISTS channel_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_channel
    ON channel_history (channel_id, id);

CREATE TABLE IF NOT EXISTS usage (
    guild_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    command  TEXT    NOT NULL,
    count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id, command)
);

CREATE TABLE IF NOT EXISTS mod_allowlist (
    guild_id  INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    kind      TEXT    NOT NULL,          -- 'user' or 'role'
    PRIMARY KEY (guild_id, target_id, kind)
);
"""


@dataclass
class GuildConfig:
    guild_id: int
    persona: str = DEFAULT_PERSONA
    chat_model: Optional[str] = None
    history_window: int = 12
    mod_channel_id: Optional[int] = None
    docs_channel_id: Optional[int] = None
    moderation_on: bool = False
    mod_threshold: float = 0.6


# Columns that may be updated through set_guild_field, mapped to a coercion fn.
_ALLOWED_FIELDS = {
    "persona": str,
    "chat_model": lambda v: None if v is None else str(v),
    "history_window": int,
    "mod_channel_id": lambda v: None if v is None else int(v),
    "docs_channel_id": lambda v: None if v is None else int(v),
    "moderation_on": lambda v: 1 if v else 0,
    "mod_threshold": float,
}


class Database:
    """Async SQLite wrapper. Call :meth:`connect` once at startup."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() was not awaited")
        return self._conn

    # -------------------------------------------------------- guild config
    async def get_guild_config(self, guild_id: int) -> GuildConfig:
        async with self._db.execute(
            "SELECT * FROM guild_config WHERE guild_id = ?", (guild_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            await self._db.execute(
                "INSERT INTO guild_config (guild_id) VALUES (?)", (guild_id,)
            )
            await self._db.commit()
            return GuildConfig(guild_id=guild_id)
        return GuildConfig(
            guild_id=row["guild_id"],
            persona=row["persona"],
            chat_model=row["chat_model"],
            history_window=row["history_window"],
            mod_channel_id=row["mod_channel_id"],
            docs_channel_id=row["docs_channel_id"],
            moderation_on=bool(row["moderation_on"]),
            mod_threshold=row["mod_threshold"],
        )

    async def set_guild_field(self, guild_id: int, field: str, value) -> None:
        if field not in _ALLOWED_FIELDS:
            raise ValueError(f"Unknown guild config field: {field}")
        coerced = _ALLOWED_FIELDS[field](value)
        # Ensure the row exists first.
        await self.get_guild_config(guild_id)
        await self._db.execute(
            f"UPDATE guild_config SET {field} = ? WHERE guild_id = ?",
            (coerced, guild_id),
        )
        await self._db.commit()

    # ---------------------------------------------------- channel history
    async def add_history(self, channel_id: int, role: str, content: str) -> None:
        await self._db.execute(
            "INSERT INTO channel_history (channel_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (channel_id, role, content, time.time()),
        )
        await self._db.commit()

    async def get_history(self, channel_id: int, limit: int) -> List[Dict[str, str]]:
        """Return the last ``limit`` messages for a channel in chronological order."""
        async with self._db.execute(
            "SELECT role, content FROM channel_history "
            "WHERE channel_id = ? ORDER BY id DESC LIMIT ?",
            (channel_id, max(0, limit)),
        ) as cur:
            rows = await cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def trim_history(self, channel_id: int, keep: int) -> None:
        """Delete all but the newest ``keep`` messages for a channel."""
        await self._db.execute(
            "DELETE FROM channel_history WHERE channel_id = ? AND id NOT IN ("
            "  SELECT id FROM channel_history WHERE channel_id = ? "
            "  ORDER BY id DESC LIMIT ?"
            ")",
            (channel_id, channel_id, max(0, keep)),
        )
        await self._db.commit()

    async def clear_history(self, channel_id: int) -> int:
        cur = await self._db.execute(
            "DELETE FROM channel_history WHERE channel_id = ?", (channel_id,)
        )
        await self._db.commit()
        return cur.rowcount

    async def clear_guild_data(self, guild_id: int, channel_ids: List[int]) -> None:
        """Wipe everything the bot stored for a guild (privacy: /config wipe)."""
        for cid in channel_ids:
            await self._db.execute(
                "DELETE FROM channel_history WHERE channel_id = ?", (cid,)
            )
        await self._db.execute("DELETE FROM usage WHERE guild_id = ?", (guild_id,))
        await self._db.execute(
            "DELETE FROM mod_allowlist WHERE guild_id = ?", (guild_id,)
        )
        await self._db.execute(
            "DELETE FROM guild_config WHERE guild_id = ?", (guild_id,)
        )
        await self._db.commit()

    # ------------------------------------------------------------- usage
    async def increment_usage(self, guild_id: int, user_id: int, command: str) -> None:
        await self._db.execute(
            "INSERT INTO usage (guild_id, user_id, command, count) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(guild_id, user_id, command) DO UPDATE SET count = count + 1",
            (guild_id, user_id, command),
        )
        await self._db.commit()

    async def usage_totals(self, guild_id: int) -> List[Tuple[str, int]]:
        async with self._db.execute(
            "SELECT command, SUM(count) AS total FROM usage "
            "WHERE guild_id = ? GROUP BY command ORDER BY total DESC",
            (guild_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["command"], r["total"]) for r in rows]

    # -------------------------------------------------- moderation allowlist
    async def add_allowlist(self, guild_id: int, target_id: int, kind: str) -> None:
        await self._db.execute(
            "INSERT OR IGNORE INTO mod_allowlist (guild_id, target_id, kind) "
            "VALUES (?, ?, ?)",
            (guild_id, target_id, kind),
        )
        await self._db.commit()

    async def remove_allowlist(self, guild_id: int, target_id: int) -> int:
        cur = await self._db.execute(
            "DELETE FROM mod_allowlist WHERE guild_id = ? AND target_id = ?",
            (guild_id, target_id),
        )
        await self._db.commit()
        return cur.rowcount

    async def get_allowlist(self, guild_id: int) -> List[Tuple[int, str]]:
        async with self._db.execute(
            "SELECT target_id, kind FROM mod_allowlist WHERE guild_id = ?",
            (guild_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["target_id"], r["kind"]) for r in rows]
