"""The terminal simulator (python -m bot.cli), fully offline."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from bot.cli import main

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def sim(tmp_path, capsys):
    data_dir = tmp_path / "sim"

    def run(*argv):
        code = main([*argv, "--data-dir", str(data_dir)])
        out = capsys.readouterr()
        return code, out.out

    return run


def test_docs_ingest_and_ask_the_repository_guide(sim):
    code, out = sim("docs", "ingest", str(REPO / "docs"))
    assert code == 0 and "Indexed" in out and "offline/hashed-bow-512 (512-d)" in out
    code, out = sim("docs", "ask", "what permission integer does the invite URL use?", "--explain")
    assert code == 0
    assert "84992" in out and "Sources" in out and "setup.md" in out
    assert "retrieved (cosine similarity" in out
    code, out = sim("docs", "status")
    assert "Indexed chunks:" in out and "offline/hashed-bow-512" in out


def test_moderate_scam_is_reported_and_paradox_is_free(sim):
    code, out = sim("moderate", "free nitro click this link discord.gg/x")
    assert code == 0 and '"category": "scam"' in out and "mod channel: YES" in out
    code, out = sim("moderate", "what a paradox")
    assert "skipped (below the prefilter floor" in out and "mod channel: no" in out


def test_summarize_file_and_from_line(sim):
    transcript = str(REPO / "examples" / "standup.txt")
    code, out = sim("summarize", transcript)
    assert code == 0 and "Summary of the last 12 messages" in out and "• Decision:" in out
    code, out = sim("summarize", transcript, "--from-line", "6")
    assert "Summary of 7 messages from line 6" in out


def test_image_reports_normalisation(sim, tmp_path):
    path = tmp_path / "big.webp"
    Image.new("RGB", (4000, 3000), (40, 80, 200)).save(path, format="WEBP")
    code, out = sim("image", str(path), "--question", "what is this?")
    assert code == 0
    assert "input: 4000x3000 WEBP" in out
    assert "sent to vision model: 1568x1176 image/jpeg" in out
    assert "blue" in out


def test_ask_stats_and_forget(sim):
    sim("ask", "hello")
    sim("ask", "again", "--user", "2")
    code, out = sim("stats")
    assert code == 0 and "3 command(s) recorded" in out and "4 stored /ask message(s)" in out
    assert "/ask — 2" in out and "1. user 1" in out
    code, out = sim("forget")
    assert "Cleared 4 stored message(s)" in out


def test_failures_have_nonzero_exit_codes(sim, tmp_path):
    code, out = sim("docs", "ask", "anything")
    assert code == 1 and "No documents have been ingested" in out
    code, _ = sim("image", str(tmp_path / "missing.png"))
    assert code == 2
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    code, out = sim("image", str(bad))
    assert code == 1 and "not an image format" in out


def test_live_without_a_key_explains_how_to_get_one(sim, capsys):
    code = main(["--live", "ask", "hi", "--data-dir", "unused"])
    err = capsys.readouterr().err
    assert code == 2 and "build.nvidia.com" in err


def _clean_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DISCORD", "NVIDIA", "NIM_", "BOT_"))}
    env["DATA_DIR"] = str(tmp_path / "data")
    env["BOT_ENV_FILE"] = str(tmp_path / "no-such.env")  # never read a developer's .env
    return env


def test_module_entry_points_run(tmp_path):
    help_out = subprocess.run(
        [sys.executable, "-m", "bot.cli", "--help"], cwd=REPO, env=_clean_env(tmp_path),
        capture_output=True, text=True,
    )
    assert help_out.returncode == 0 and "docs" in help_out.stdout and "moderate" in help_out.stdout
    bot = subprocess.run(
        [sys.executable, "-m", "bot.main"], cwd=REPO, env=_clean_env(tmp_path),
        capture_output=True, text=True,
    )
    assert bot.returncode == 1 and "DISCORD_BOT_TOKEN is not set" in bot.stdout
    no_key = subprocess.run(
        [sys.executable, "-m", "bot.main"], cwd=REPO,
        env={**_clean_env(tmp_path), "DISCORD_BOT_TOKEN": "x.y.z"},
        capture_output=True, text=True,
    )
    assert no_key.returncode == 1 and "NVIDIA_API_KEY is not set" in no_key.stdout
    assert "BOT_OFFLINE=1" in no_key.stdout
