"""Settings loading (no .env walk-up) and the rate limiter's memory bound."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from bot.config import _parse_ids, load_env_file, load_settings
from bot.middleware import GuildGuard, RateLimiter

REPO = Path(__file__).resolve().parent.parent


def test_parse_ids_handles_mixed_separators_and_junk():
    assert _parse_ids("1, 2;3 , x, 4") == {1, 2, 3, 4}
    assert _parse_ids("") == set()
    assert _parse_ids(None) == set()


def test_importing_config_does_not_load_a_parent_directory_env(tmp_path):
    # Regression: load_dotenv() at import walked up and read ../.env files.
    parent = tmp_path / "unrelated_parent"
    checkout = parent / "checkout"
    shutil.copytree(REPO / "bot", checkout / "bot")
    (parent / ".env").write_text("NIM_MODEL=leaked-from-a-parent-dir\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NIM_", "NVIDIA", "DISCORD"))}
    env["DATA_DIR"] = str(tmp_path / "data")
    code = (
        "import os, bot.config as c; c.load_env_file(); "
        "print(os.getenv('NIM_MODEL')); print(c.load_settings().chat_model)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=checkout / "bot",  # a cwd *below* the leaked file, like find_dotenv
        env={**env, "PYTHONPATH": str(checkout)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["None", "meta/llama-3.3-70b-instruct"]


def test_load_env_file_reads_only_the_given_file_and_env_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NIM_MODEL=from-file\nNIM_MAX_TOKENS=321\n", encoding="utf-8")
    monkeypatch.setenv("NIM_MAX_TOKENS", "111")
    assert load_env_file(env_file) is True
    try:
        settings = load_settings()
        assert settings.chat_model == "from-file"
        assert settings.max_tokens == 111  # real environment variables win
    finally:
        os.environ.pop("NIM_MODEL", None)
    assert load_env_file(tmp_path / "missing.env") is False


def test_settings_parse_optional_values(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.35")
    monkeypatch.setenv("COOLDOWN_SECONDS", "not-a-number")
    settings = load_settings()
    assert settings.rag_min_score == 0.35
    assert settings.cooldown_seconds == 5.0
    assert settings.data_dir.is_dir() and settings.rag_dir.is_dir()
    assert not settings.has_nim_key()
    settings.nvidia_api_key = "nvapi-XXXXXXXX"
    assert not settings.has_nim_key()
    settings.nvidia_api_key = "nvapi-real"
    assert settings.has_nim_key()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_enforces_cooldown_then_allows():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=5.0, per_minute=10, clock=clock)
    assert limiter.check(1) == (True, 0.0)
    allowed, retry = limiter.check(1)
    assert not allowed and retry == 5.0
    clock.now += 5.0
    assert limiter.check(1)[0]


def test_rate_limiter_sliding_window():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=0.0, per_minute=2, clock=clock)
    assert limiter.check(42)[0] and limiter.check(42)[0]
    allowed, retry = limiter.check(42)
    assert not allowed and retry == 60.0
    clock.now += 60.5
    assert limiter.check(42)[0]


def test_rate_limiter_evicts_idle_users():
    # Regression: _last_call/_windows grew forever, one entry per user ever seen.
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=5.0, per_minute=12, clock=clock, prune_interval=60)
    for user in range(1000):
        limiter.check(user)
    assert limiter.tracked_users == 1000
    clock.now += 61
    limiter.check(5000)  # triggers the periodic prune
    assert limiter.tracked_users == 1
    assert limiter.check(1)[0]  # an evicted user is simply "new" again


def test_guild_guard():
    assert GuildGuard(set()).is_allowed(123) and not GuildGuard(set()).restricted
    guard = GuildGuard({1, 2})
    assert guard.restricted and guard.is_allowed(1) and not guard.is_allowed(3)
