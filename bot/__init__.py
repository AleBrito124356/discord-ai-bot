"""discord-ai-bot: a complete Discord AI bot on discord.py + free NVIDIA NIM.

Modules
-------
config       Environment-driven settings (single ``load_settings`` entry point).
personas     Named system prompts the bot can switch between.
nim_client   Async client for NVIDIA NIM chat, vision and embedding endpoints.
persistence  SQLite storage: guild config, per-channel memory, usage counters.
middleware   Cooldowns, rate limits, guild allowlist and the error handler.
rag          Optional retrieval over a server's docs channel (numpy vector store).
moderation   Assist-only flagging of likely rule-breaking messages (never acts).
main         Bot entrypoint and slash-command definitions.
"""

__version__ = "1.0.0"
