"""discord-ai-bot: a complete Discord AI bot on discord.py + free NVIDIA NIM.

Modules
-------
config       Environment-driven settings (single ``load_settings`` entry point).
personas     Named system prompts the bot can switch between.
prompts      System prompts shared by the services and the offline backend.
nim_client   Async client for NVIDIA NIM chat, vision and embedding endpoints.
offline      Deterministic offline backend with the NimClient interface.
imaging      Pillow normalisation of uploads to what the vision model accepts.
persistence  SQLite storage: guild config, per-channel memory, usage counters.
middleware   Cooldowns, rate limits, guild allowlist and the error handler.
rag          Optional retrieval over a server's docs channel (numpy vector store).
moderation   Assist-only flagging of likely rule-breaking messages (never acts).
textutil     Fence-aware message splitting and summarize transcripts.
services     Discord-free business logic (BotCore) used by the bot and the CLI.
main         Bot entrypoint and slash-command definitions (thin adapters).
cli          Terminal simulator: every feature without Discord or keys.
"""

__version__ = "1.1.0"
