"""Shared fixtures. Every test runs offline, with no keys and a temp DATA_DIR."""
from __future__ import annotations

import pytest

from bot.config import load_settings
from bot.persistence import Database

# Every variable the bot reads. They are cleared so a developer's shell (or a
# stray .env) can never change test behaviour.
BOT_ENV_VARS = (
    "DISCORD_BOT_TOKEN",
    "NVIDIA_API_KEY",
    "NIM_BASE_URL",
    "NIM_MODEL",
    "NIM_VISION_MODEL",
    "NIM_EMBED_MODEL",
    "NIM_MAX_TOKENS",
    "ALLOWED_GUILD_IDS",
    "DATA_DIR",
    "BOT_OFFLINE",
    "RAG_MIN_SCORE",
    "COOLDOWN_SECONDS",
    "RATE_LIMIT_PER_MINUTE",
    "NIM_IMAGE_MAX_SIDE",
    "NIM_IMAGE_MAX_B64",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for name in BOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    yield


@pytest.fixture
def settings(tmp_path):
    return load_settings()


@pytest.fixture
async def db(settings):
    """A connected Database that is ALWAYS closed (an open one hangs pytest)."""
    database = Database(settings.db_path)
    await database.connect()
    try:
        yield database
    finally:
        await database.close()
