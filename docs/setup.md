# Setup guide

End-to-end: create the Discord application, get your keys, invite the bot, and
run it locally, under systemd, or in Docker.

> Just evaluating? `python -m bot.cli --help` runs every feature from a terminal
> with no Discord app and no NVIDIA key (see "Try it with no keys" in the README).

---

## 1. Create the Discord application and bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   and click **New Application**. Give it a name.
2. Open the **Bot** tab. The application already has a bot user attached.
3. Click **Reset Token**, then **Copy**. This is your `DISCORD_BOT_TOKEN`.
   Treat it like a password — anyone with it can control your bot.

### Enable the required intent

This bot needs the **Message Content Intent** so it can read message text for
`/summarize`, the *Summarize from here* menu and assist-only moderation.

- In the **Bot** tab, scroll to **Privileged Gateway Intents**.
- Turn on **Message Content Intent**. Save.

> The bot works without the Members intent. If you later want richer member data
> in moderation advisories, enable **Server Members Intent** too — it is optional.

---

## 2. Get a free NVIDIA NIM API key

1. Open [build.nvidia.com](https://build.nvidia.com) and sign in.
2. Pick any model (for example `meta/llama-3.3-70b-instruct`).
3. Click **Get API Key** and copy the key that starts with `nvapi-`.

This is your `NVIDIA_API_KEY`. The free tier is enough to run the bot for a small
to medium server.

---

## 3. Invite the bot to your server

Build an OAuth2 invite URL with both scopes the bot needs:

- `bot` — lets it join the server.
- `applications.commands` — lets it register slash commands.

You can generate the URL in the **OAuth2 -> URL Generator** tab (tick both scopes,
then tick the channel permissions below), or build it by hand:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_APPLICATION_ID&permissions=84992&scope=bot%20applications.commands
```

Replace `YOUR_APPLICATION_ID` with the **Application ID** from the **General
Information** tab.

### Permission integer 84992

That integer requests exactly the channel permissions the bot uses:

| Permission            | Why                                   |
| --------------------- | ------------------------------------- |
| View Channels         | See the channels it operates in       |
| Send Messages         | Reply to commands                     |
| Embed Links           | `/help`, `/config show`, mod advisories |
| Read Message History  | `/summarize`, *Summarize from here*, docs ingestion (all pins) |

The two message context menus (*Ask AI about this*, *Summarize from here*) are
registered together with the slash commands and need no extra permission.

The bot requests **no** moderation powers (no Ban, Kick, Manage Messages). That is
deliberate — see the moderation philosophy in the README. If you want it to *only*
watch specific channels, restrict its role's channel access in Server Settings.

---

## 4. Configure and run locally

```bash
git clone https://github.com/AleBrito124356/discord-ai-bot.git
cd discord-ai-bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env with your token and NVIDIA key
python -m bot.main
```

On first start the bot syncs its slash commands. If you set `ALLOWED_GUILD_IDS`,
the commands appear in those servers instantly. Global sync (empty allowlist) can
take up to an hour to propagate the first time.

Then in your server:

```
/config docs-channel #docs        (optional, for RAG)
/docs ingest                      (optional, indexes the docs channel)
/config mod-channel #mod-log      (optional, for moderation advisories)
/config moderation enabled: true  (optional, opt-in)
/ask prompt: hello!
```

---

## 5. Run under systemd (Linux VPS)

Create `/etc/systemd/system/discord-ai-bot.service`:

```ini
[Unit]
Description=discord-ai-bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/opt/discord-ai-bot
EnvironmentFile=/opt/discord-ai-bot/.env
ExecStart=/opt/discord-ai-bot/.venv/bin/python -m bot.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now discord-ai-bot
journalctl -u discord-ai-bot -f      # follow logs
```

---

## 6. Run in Docker

```bash
docker build -t discord-ai-bot .
docker run -d --name discord-ai-bot \
  --env-file .env \
  -v discord_ai_bot_data:/app/data \
  --restart unless-stopped \
  discord-ai-bot
```

The named volume persists the SQLite database and the RAG vector stores across
restarts and image rebuilds.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| ------- | ------------------ |
| Slash commands do not appear | Global sync is slow; set `ALLOWED_GUILD_IDS` for instant per-guild sync, or wait up to an hour. |
| `PrivilegedIntentsRequired` on startup | Enable the Message Content Intent in the Bot tab. |
| Every NIM call fails with 401 | Wrong or expired `NVIDIA_API_KEY`. Regenerate at build.nvidia.com. |
| `/summarize` reads nothing | The bot lacks Read Message History in that channel, or the channel is empty of human messages. |
| Vision returns an error | The attachment is not a readable image or is over 25 MB. Anything else is converted and downscaled automatically (`NIM_IMAGE_MAX_SIDE`, `NIM_IMAGE_MAX_B64`). |
| `/docs ask` says the index was built with another model | You changed `NIM_EMBED_MODEL` (or switched to/from offline mode). Run `/docs ingest` again. |
| `/docs ask` cites unrelated docs, or finds nothing | Tune `RAG_MIN_SCORE` (default 0.2 with NIM): raise it to drop weak matches, lower it if good matches are being filtered. |
| Want to test the setup before getting an NVIDIA key | Start with `BOT_OFFLINE=1`: every command works on the deterministic offline backend. |
