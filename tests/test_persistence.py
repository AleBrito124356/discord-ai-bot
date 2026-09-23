"""SQLite persistence: config, memory, wipe, usage, allowlist and migration."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from bot.persistence import SCHEMA_VERSION, Database

# The exact schema shipped in version 1.0 (channel_history had no guild_id).
V1_SCHEMA = """
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
    kind      TEXT    NOT NULL,
    PRIMARY KEY (guild_id, target_id, kind)
);
"""


async def test_guild_config_defaults_and_updates(db):
    cfg = await db.get_guild_config(1)
    assert (cfg.persona, cfg.chat_model, cfg.history_window) == ("assistant", None, 12)
    assert (cfg.moderation_on, cfg.mod_threshold) == (False, 0.6)
    await db.set_guild_field(1, "persona", "coder")
    await db.set_guild_field(1, "moderation_on", True)
    await db.set_guild_field(1, "mod_threshold", "0.8")
    cfg = await db.get_guild_config(1)
    assert (cfg.persona, cfg.moderation_on, cfg.mod_threshold) == ("coder", True, 0.8)


async def test_set_guild_field_rejects_unknown_columns(db):
    with pytest.raises(ValueError):
        await db.set_guild_field(1, "guild_id; DROP TABLE usage", 1)


async def test_concurrent_get_guild_config_does_not_race(db):
    # Regression: SELECT-then-INSERT raised "UNIQUE constraint failed".
    configs = await asyncio.gather(*(db.get_guild_config(7) for _ in range(10)))
    assert {c.guild_id for c in configs} == {7}


async def test_history_is_chronological_and_trimmed(db):
    for i in range(10):
        await db.add_history(100, "user", f"m{i}", guild_id=1)
    assert [m["content"] for m in await db.get_history(100, 3)] == ["m7", "m8", "m9"]
    await db.trim_history(100, 4)
    assert [m["content"] for m in await db.get_history(100, 50)] == ["m6", "m7", "m8", "m9"]
    assert await db.clear_history(100) == 4
    assert await db.get_history(100, 50) == []


async def test_wipe_erases_memory_in_threads_and_deleted_channels(db):
    # Regression: the wipe only deleted memory for guild.text_channels.
    await db.add_history(100, "user", "in a text channel", guild_id=1)
    await db.add_history(200, "user", "secret in a thread", guild_id=1)
    await db.add_history(300, "user", "in a channel deleted since", guild_id=1)
    await db.add_history(900, "user", "another server", guild_id=2)
    await db.increment_usage(1, 5, "ask")
    await db.add_allowlist(1, 5, "user")
    await db.set_guild_field(1, "persona", "coder")

    removed = await db.clear_guild_data(1, channel_ids=[100])
    assert removed["channel_history"] == 3
    for channel in (100, 200, 300):
        assert await db.get_history(channel, 10) == []
    assert await db.get_history(900, 10) == [{"role": "user", "content": "another server"}]
    assert await db.usage_totals(1) == []
    assert await db.get_allowlist(1) == []
    assert (await db.get_guild_config(1)).persona == "assistant"


async def test_usage_stats(db):
    for user, command, times in [(1, "ask", 3), (2, "ask", 1), (2, "summarize", 2), (3, "docs_ask", 1)]:
        for _ in range(times):
            await db.increment_usage(10, user, command)
    await db.increment_usage(99, 1, "ask")  # another guild
    await db.add_history(5, "user", "x", guild_id=10)
    stats = await db.usage_stats(10, top=2)
    assert stats.total == 7
    assert stats.per_command == [("ask", 4), ("summarize", 2), ("docs_ask", 1)]
    assert stats.top_users == [(1, 3), (2, 3)]
    assert stats.stored_messages == 1


async def test_allowlist_roundtrip(db):
    await db.add_allowlist(1, 42, "user")
    await db.add_allowlist(1, 42, "user")  # idempotent
    await db.add_allowlist(1, 7, "role")
    assert sorted(await db.get_allowlist(1)) == [(7, "role"), (42, "user")]
    assert await db.remove_allowlist(1, 42) == 1
    assert await db.remove_allowlist(1, 42) == 0


async def test_migrates_a_v1_database_in_place(tmp_path):
    path = tmp_path / "v1.db"
    raw = sqlite3.connect(path)
    raw.executescript(V1_SCHEMA)
    raw.execute("INSERT INTO guild_config (guild_id, persona) VALUES (1, 'teacher')")
    raw.executemany(
        "INSERT INTO channel_history (channel_id, role, content, created_at) VALUES (?, ?, ?, 0)",
        [(100, "user", "old text-channel memory"), (200, "user", "old thread memory")],
    )
    raw.execute("INSERT INTO usage VALUES (1, 5, 'ask', 3)")
    raw.commit()
    raw.close()

    db = Database(path)
    await db.connect()
    try:
        assert await db.schema_version() == SCHEMA_VERSION
        # Old rows survived the migration.
        assert (await db.get_guild_config(1)).persona == "teacher"
        assert await db.get_history(100, 5) == [{"role": "user", "content": "old text-channel memory"}]
        assert await db.usage_totals(1) == [("ask", 3)]
        # Using a channel again back-fills its guild_id.
        await db.add_history(200, "assistant", "new reply", guild_id=1)
        # The wipe removes back-filled rows by guild and legacy rows by channel id.
        removed = await db.clear_guild_data(1, channel_ids=[100])
        assert removed["channel_history"] == 3
        assert await db.get_history(100, 5) == []
        assert await db.get_history(200, 5) == []
    finally:
        await db.close()

    # Re-connecting an already migrated DB is a no-op.
    db = Database(path)
    await db.connect()
    try:
        assert await db.schema_version() == SCHEMA_VERSION
    finally:
        await db.close()


async def test_using_db_before_connect_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="connect"):
        await Database(tmp_path / "x.db").get_guild_config(1)
