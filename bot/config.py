"""Central configuration loaded once from environment variables.

Everything runtime-tunable lives here so the rest of the bot imports a single
``Settings`` object. This module never talks to Discord or NVIDIA NIM; it only
reads, parses and validates environment variables (optionally from a ``.env``
file via python-dotenv).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

# Load a local .env if present. Real environment variables always win.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SIGNUP_HINT = (
    "Get a free NVIDIA NIM key in about two minutes:\n"
    "  1. Open https://build.nvidia.com and sign in.\n"
    "  2. Pick any model, click 'Get API Key'.\n"
    "  3. Copy the key that starts with 'nvapi-' into NVIDIA_API_KEY."
)


def _parse_ids(raw: str | None) -> Set[int]:
    """Parse a comma/semicolon separated list of integer IDs, skipping junk."""
    if not raw:
        return set()
    ids: Set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            continue
    return ids


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Immutable-ish view of all runtime configuration."""

    discord_token: str
    nvidia_api_key: str
    nim_base_url: str
    chat_model: str
    vision_model: str
    embed_model: str
    allowed_guild_ids: Set[int] = field(default_factory=set)

    # Storage
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")

    # Defaults applied to a guild the first time it is seen.
    default_history_window: int = 12
    default_mod_threshold: float = 0.6

    # Rate limiting.
    cooldown_seconds: float = 5.0
    rate_limit_per_minute: int = 12

    # Generation limits.
    max_tokens: int = 900
    temperature: float = 0.5

    # RAG tuning.
    rag_chunk_chars: int = 1200
    rag_chunk_overlap: int = 150
    rag_top_k: int = 4

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def rag_dir(self) -> Path:
        return self.data_dir / "rag"

    def has_nim_key(self) -> bool:
        key = self.nvidia_api_key.strip()
        return bool(key) and not key.upper().startswith("NVAPI-XXXX")


def load_settings() -> Settings:
    """Build a Settings instance from the current environment.

    Raises nothing here for missing keys; ``main`` prints a friendly message and
    exits so imports stay side-effect free and testable.
    """
    data_dir = Path(os.getenv("DATA_DIR") or (BASE_DIR / "data")).expanduser()
    settings = Settings(
        discord_token=os.getenv("DISCORD_BOT_TOKEN", "").strip(),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        nim_base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").strip(),
        chat_model=os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct").strip(),
        vision_model=os.getenv("NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct").strip(),
        embed_model=os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5").strip(),
        allowed_guild_ids=_parse_ids(os.getenv("ALLOWED_GUILD_IDS")),
        data_dir=data_dir,
        cooldown_seconds=_env_float("COOLDOWN_SECONDS", 5.0),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 12),
        max_tokens=_env_int("NIM_MAX_TOKENS", 900),
    )
    # Ensure storage directories exist up front.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_dir.mkdir(parents=True, exist_ok=True)
    return settings
