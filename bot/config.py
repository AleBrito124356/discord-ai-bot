"""Central configuration loaded once from environment variables.

Everything runtime-tunable lives here so the rest of the bot imports a single
``Settings`` object. This module never talks to Discord or NVIDIA NIM; it only
reads, parses and validates environment variables.

Importing it has no side effects. The ``.env`` file is loaded explicitly by the
entry points through :func:`load_env_file`, and only from the repository root —
never from a parent directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = BASE_DIR / ".env"

# NIM_BASE_URL=offline (or BOT_OFFLINE=1) selects the deterministic offline
# backend in bot/offline.py instead of NVIDIA NIM.
OFFLINE_BASE_URL = "offline"
_TRUTHY = {"1", "true", "yes", "on", "y"}

SIGNUP_HINT = (
    "Get a free NVIDIA NIM key in about two minutes:\n"
    "  1. Open https://build.nvidia.com and sign in.\n"
    "  2. Pick any model, click 'Get API Key'.\n"
    "  3. Copy the key that starts with 'nvapi-' into NVIDIA_API_KEY.\n"
    "No key yet? Set BOT_OFFLINE=1 to run on the deterministic offline backend,\n"
    "or try everything from a terminal with: python -m bot.cli --help"
)


def load_env_file(path: Optional[Path] = None) -> bool:
    """Load ``path`` (default: ``$BOT_ENV_FILE`` or ``<repo>/.env``) into ``os.environ``.

    Real environment variables always win. Unlike ``load_dotenv()`` with no
    argument, this never walks up parent directories looking for a ``.env``, so
    an unrelated file above the checkout can not leak settings into the bot.
    Returns True when a file was found and read.
    """
    if path is None:
        path = os.getenv("BOT_ENV_FILE") or DEFAULT_ENV_FILE
    target = Path(path)
    if not target.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return False
    return bool(load_dotenv(target, override=False))


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


def _env_bool(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in _TRUTHY


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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

    # Offline mode: bot/offline.py answers instead of NVIDIA NIM (no key needed).
    offline: bool = False

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
    # Minimum cosine similarity for a retrieved chunk to be used at all.
    # None means "use the embedding backend's own default".
    rag_min_score: Optional[float] = None

    # /summarize: max characters of transcript sent to the model.
    summarize_char_budget: int = 12_000

    # Vision: images are converted/downscaled to fit these before upload.
    # 180 000 base64 characters is the inline-image limit used in NVIDIA's
    # hosted API examples ("use the assets API" beyond that).
    image_max_side: int = 1568
    image_max_b64: int = 180_000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def rag_dir(self) -> Path:
        return self.data_dir / "rag"

    def has_nim_key(self) -> bool:
        key = self.nvidia_api_key.strip()
        return bool(key) and not key.upper().startswith("NVAPI-XXXX")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rag_dir.mkdir(parents=True, exist_ok=True)


def load_settings(*, create_dirs: bool = True) -> Settings:
    """Build a Settings instance from the current environment.

    Raises nothing for missing keys; ``main`` prints a friendly message and
    exits. This only reads ``os.environ``: call :func:`load_env_file` first if a
    ``.env`` file should be applied.
    """
    data_dir = Path(os.getenv("DATA_DIR") or (BASE_DIR / "data")).expanduser()
    base_url = os.getenv("NIM_BASE_URL", "").strip() or "https://integrate.api.nvidia.com/v1"
    settings = Settings(
        discord_token=os.getenv("DISCORD_BOT_TOKEN", "").strip(),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        nim_base_url=base_url,
        offline=_env_bool("BOT_OFFLINE") or base_url.lower() == OFFLINE_BASE_URL,
        chat_model=os.getenv("NIM_MODEL", "").strip() or "meta/llama-3.3-70b-instruct",
        vision_model=os.getenv("NIM_VISION_MODEL", "").strip()
        or "meta/llama-3.2-90b-vision-instruct",
        embed_model=os.getenv("NIM_EMBED_MODEL", "").strip() or "nvidia/nv-embedqa-e5-v5",
        allowed_guild_ids=_parse_ids(os.getenv("ALLOWED_GUILD_IDS")),
        data_dir=data_dir,
        cooldown_seconds=_env_float("COOLDOWN_SECONDS", 5.0),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 12),
        max_tokens=_env_int("NIM_MAX_TOKENS", 900),
        rag_min_score=_env_optional_float("RAG_MIN_SCORE"),
        image_max_side=max(64, _env_int("NIM_IMAGE_MAX_SIDE", 1568)),
        image_max_b64=max(10_000, _env_int("NIM_IMAGE_MAX_B64", 180_000)),
    )
    if create_dirs:
        settings.ensure_dirs()
    return settings
