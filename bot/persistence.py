"""SQLite persistence: guild config, per-channel memory, usage counters.

Uses ``aiosqlite`` so nothing blocks the event loop. One long-lived connection is
opened in :meth:`Database.connect` and shared for the process. All writes commit
immediately; the traffic a chat bot generates does not need batching.

Schema versions
---------------
1. Original layout. ``channel_history`` had no ``guild_id``, so a server wipe
   could only delete memory for channels it could still enumerate.
2. ``channel_history.guild_id`` (+ index) and a ``schema_version`` table. A v1
   database is migrated in place on :meth:`Database.connect`; old rows keep
   ``guild_id = NULL`` until the channel is used again (then they are
   back-filled) or a wipe matches them by channel id.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import aiosqlite

from .personas import DEFAULT_PERSONA

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

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
    guild_id   INTEGER,
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

# Created after the migration step so it also works on a v1 table.
_POST_MIGRATION = """
CREATE INDEX IF NOT EXISTS idx_history_guild
    ON channel_history (guild_id);
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


@dataclass
class UsageStats:
    """Aggregated usage counters for one guild (rendered by /stats)."""

    total: int
    per_command: List[Tuple[str, int]]
    top_users: List[Tuple[int, int]]
    stored_messages: int


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
        await self._migrate()
        await self._conn.executescript(_POST_MIGRATION)
        await self._conn.commit()

    async def _migrate(self) -> None:
        """Bring an older database up to ``SCHEMA_VERSION`` in place."""
        async with self._db.execute("PRAGMA table_info(channel_history)") as cur:
            columns = {row["name"] for row in await cur.fetchall()}
        if "guild_id" not in columns:
            await self._db.execute(
                "ALTER TABLE channel_history ADD COLUMN guild_id INTEGER"
            )
        async with self._db.execute("SELECT version FROM schema_version") as cur:
            rows = await cur.fetchall()
        if not rows:
            await self._db.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        elif rows[0]["version"] < SCHEMA_VERSION:
            await self._db.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )

    async def schema_version(self) -> int:
        async with self._db.execute("SELECT version FROM schema_version") as cur:
            row = await cur.fetchone()
        return int(row["version"]) if row else 0

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
    async def _select_config(self, guild_id: int):
        async with self._db.execute(
            "SELECT * FROM guild_config WHERE guild_id = ?", (guild_id,)
        ) as cur:
            return await cur.fetchone()

    async def get_guild_config(self, guild_id: int) -> GuildConfig:
        # Read first (on_message calls this for every message: no write, no
        # open transaction on the hot path). On a miss, INSERT OR IGNORE: two
        # events for a brand-new guild can both miss, and a plain INSERT made
        # the second one fail with "UNIQUE constraint failed".
        row = await self._select_config(guild_id)
        if row is None:
            await self._db.execute(
                "INSERT OR IGNORE INTO guild_config (guild_id) VALUES (?)", (guild_id,)
            )
            await self._db.commit()
            row = await self._select_config(guild_id)
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
    async def add_history(
        self,
        channel_id: int,
        role: str,
        content: str,
        *,
        guild_id: Optional[int] = None,
    ) -> None:
        """Store one message of /ask memory.

        ``guild_id`` ties the row to its server so ``clear_guild_data`` can wipe
        memory kept in threads, forum posts or since-deleted channels. Passing it
        also back-fills rows written before schema v2 for the same channel.
        """
        if guild_id is not None:
            await self._db.execute(
                "UPDATE channel_history SET guild_id = ? "
                "WHERE channel_id = ? AND guild_id IS NULL",
                (guild_id, channel_id),
            )
        await self._db.execute(
            "INSERT INTO channel_history (guild_id, channel_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (guild_id, channel_id, role, content, time.time()),
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

    async def count_history(self, guild_id: int) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) AS n FROM channel_history WHERE guild_id = ?",
            (guild_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"])

    async def clear_guild_data(
        self, guild_id: int, channel_ids: Iterable[int] = ()
    ) -> Dict[str, int]:
        """Wipe everything the bot stored for a guild (privacy: /config wipe).

        Memory rows are matched by ``guild_id``, which covers threads, forum
        posts, voice-channel chats and deleted channels. ``channel_ids`` is only
        needed for rows written before schema v2 that were never back-filled.
        Returns how many rows were removed per table.
        """
        removed: Dict[str, int] = {}
        cur = await self._db.execute(
            "DELETE FROM channel_history WHERE guild_id = ?", (guild_id,)
        )
        history = cur.rowcount
        legacy_ids = sorted({int(c) for c in channel_ids})
        for start in range(0, len(legacy_ids), 500):
            batch = legacy_ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            cur = await self._db.execute(
                f"DELETE FROM channel_history WHERE guild_id IS NULL "
                f"AND channel_id IN ({marks})",
                batch,
            )
            history += cur.rowcount
        removed["channel_history"] = history
        for table in ("usage", "mod_allowlist", "guild_config"):
            cur = await self._db.execute(
                f"DELETE FROM {table} WHERE guild_id = ?", (guild_id,)
            )
            removed[table] = cur.rowcount
        await self._db.commit()
        return removed

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
            "WHERE guild_id = ? GROUP BY command ORDER BY total DESC, command ASC",
            (guild_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["command"], r["total"]) for r in rows]

    async def top_users(self, guild_id: int, limit: int = 5) -> List[Tuple[int, int]]:
        async with self._db.execute(
            "SELECT user_id, SUM(count) AS total FROM usage WHERE guild_id = ? "
            "GROUP BY user_id ORDER BY total DESC, user_id ASC LIMIT ?",
            (guild_id, max(1, limit)),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["user_id"], r["total"]) for r in rows]

    async def usage_stats(self, guild_id: int, top: int = 5) -> UsageStats:
        per_command = await self.usage_totals(guild_id)
        return UsageStats(
            total=sum(n for _, n in per_command),
            per_command=per_command,
            top_users=await self.top_users(guild_id, top),
            stored_messages=await self.count_history(guild_id),
        )

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
