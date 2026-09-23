# Changelog

## 1.1.0

### Fixed
- **`/config wipe` now erases all `/ask` memory of a server**, including threads,
  forum posts, voice-channel chats and channels deleted since. Memory rows store
  their server id (schema v2); a 1.0 database is migrated in place on start-up.
- **`/summarize` keeps the newest messages** when the transcript exceeds the
  model budget (it used to keep the oldest and drop the latest), and the header
  reports what was really summarized: `Summary of the last 150 of 200 messages`.
- **`/docs ask` lists only the sources the answer cites.** A "not found" answer
  has no Sources footer, and chunks below `RAG_MIN_SCORE` never reach the model;
  if none clear it, the model is not called.
- **Moderation prefilter matches on word boundaries**: "paradox", "orthodox",
  "slurped", "fire retardant" or "skyscraper" no longer score as threats or slurs
  (each of those used to cost an LLM call).
- **Moderation verdicts are parsed strictly**: `"flag": "false"` is false (it was
  true), severities like `85`, `8` or `"85%"` are normalised to 0..1 and clamped,
  unknown categories become `other`, and a reply with two JSON-looking brace
  groups or surrounding prose still parses.
- **No more `UNIQUE constraint failed`** when two events for a brand-new server
  arrive together (`get_guild_config` race).
- **Long code answers keep valid code fences** when split into several Discord
  messages; every part re-opens the fence with its language tag.
- **The docs index is consistent on disk**: one `.npz` per server, written to a
  temp file and renamed, with the embedding model, dimension and row count
  validated on load. A mismatched or foreign-model index is refused with a
  "re-run `/docs ingest`" message instead of silently mis-citing or crashing.
- **Importing `bot.config` has no side effects**: `.env` is loaded explicitly by
  the entry points, from the repository root only (it used to walk up parent
  directories).
- `/docs ingest` reads **every** pin with the paginated iterator (it stopped at
  50 and used an API deprecated in discord.py 2.6) and parses PDFs off the event
  loop.
- The rate limiter evicts idle users instead of keeping one entry per user
  forever.
- Docs: the history window counts messages, not turns; the invite no longer asks
  for Add Reactions / Attach Files, which the bot never used (84992 instead of
  117824).

### Added
- **Offline backend** (`BOT_OFFLINE=1` or `NIM_BASE_URL=offline`): deterministic
  hashed embeddings and extractive answers for docs, summaries and moderation,
  Pillow measurements for images, and a clearly labelled placeholder for `/ask`.
- **Terminal simulator** `python -m bot.cli` / `discord-ai-bot-sim`: `ask`,
  `summarize`, `docs ingest|ask|status`, `moderate`, `image`, `stats`, `forget`,
  offline by default, `--live` for NVIDIA NIM.
- **Image normalisation for vision**: GIF (first frame), WEBP, BMP, transparent
  PNG, EXIF-rotated and oversized photos are converted to JPEG, downscaled to
  `NIM_IMAGE_MAX_SIDE` (1568) and re-encoded until the base64 payload fits
  `NIM_IMAGE_MAX_B64` (180 000). Fitting JPEG/PNG images are sent unchanged. The
  reply says when an image was changed. Uploads up to 25 MB are accepted.
- **Message context menus**: *Ask AI about this* and *Summarize from here*.
- **`/stats`** (Manage Server): commands per type, top users, stored memory and
  indexed chunks.
- `bot/services.py`: Discord-free services behind every command (`BotCore`).
- `pyproject.toml` with console scripts and a `[dev]` extra.
- Test suite grown from 11 to 157 offline tests.

### Changed (compatibility notes)
- Requires `discord.py>=2.6` (paginated pins) and the new `Pillow` dependency.
- `Database.add_history(channel_id, role, content, *, guild_id=None)` takes an
  optional keyword `guild_id`; `clear_guild_data(guild_id, channel_ids=())` now
  deletes by server and returns per-table counts.
- `RagService.answer()` returns a `RagAnswer` (text, cited sources, hits) instead
  of a `(text, chunks)` tuple; `RagAnswer.formatted()` renders the reply.
- Docs indexes are now `data/rag/<guild>.npz`. Existing 1.0 `.npy` + `.json`
  pairs are still read (and validated) until the next `/docs ingest` replaces
  them.
- NIM 404 errors now say the model name is probably wrong.
