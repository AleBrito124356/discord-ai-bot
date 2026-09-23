"""Discord adapters: the command tree and the callbacks, driven with fakes."""
from __future__ import annotations

import io
import threading

import discord
import pytest
from discord import app_commands
from PIL import Image

import bot.rag as rag_module
from bot.main import (
    ASK_ABOUT_MENU,
    SUMMARIZE_FROM_MENU,
    DiscordAIBot,
    build_help_embed,
    collect_documents,
    register_commands,
)
from bot.middleware import build_error_handler
from bot.offline import OfflineNimClient
from bot.services import document_text
from fakes import (
    FakeAttachment,
    FakeChannel,
    FakeGuild,
    FakeInteraction,
    FakeMessage,
    FakeUser,
    ScriptedNim,
    minimal_pdf,
)

MANAGE_GUILD = discord.Permissions(manage_guild=True).value
MANAGE_MESSAGES = discord.Permissions(manage_messages=True).value


@pytest.fixture
async def make_bot(settings):
    created = []

    async def factory(nim=None, **overrides):
        for key, value in overrides.items():
            setattr(settings, key, value)
        bot = DiscordAIBot(settings, nim=nim or OfflineNimClient())
        await bot.core.start()
        register_commands(bot)
        created.append(bot)
        return bot

    yield factory
    for bot in created:
        await bot.core.close()


def _cmd(bot, name, sub=None):
    command = bot.tree.get_command(name)
    return command.get_command(sub) if sub else command


def _menu(bot, name):
    return bot.tree.get_command(name, type=discord.AppCommandType.message)


def _world(messages=(), *, extra_channels=(), threads=()):
    channel = FakeChannel(100, "general", messages=messages)
    guild = FakeGuild(1, channels=[channel, *extra_channels], threads=threads)
    return guild, channel


# ------------------------------------------------------------------- tree
async def test_command_tree_serialises_with_valid_descriptions(make_bot):
    bot = await make_bot()
    top = {c.name: c for c in bot.tree.get_commands(type=discord.AppCommandType.chat_input)}
    assert set(top) == {
        "ask", "summarize", "image", "persona", "model", "forget", "stats", "help", "docs", "config",
    }
    assert {c.name for c in top["docs"].commands} == {"ask", "ingest", "status"}
    assert {c.name for c in top["config"].commands} == {
        "show", "mod-channel", "docs-channel", "moderation", "threshold",
        "history-window", "mod-allow", "mod-unallow", "wipe",
    }

    def walk(payload):
        assert 1 <= len(payload["description"]) <= 100, payload["name"]
        for option in payload.get("options", []):
            if option["type"] in (1, 2):  # subcommand / group
                walk(option)
            else:
                assert 1 <= len(option["description"]) <= 100, option["name"]

    for command in top.values():
        payload = command.to_dict(bot.tree)
        assert payload["type"] == 1 and payload["dm_permission"] is False
        walk(payload)

    perms = {name: top[name].to_dict(bot.tree)["default_member_permissions"] for name in top}
    for name in ("persona", "model", "stats", "config"):
        assert perms[name] == MANAGE_GUILD, name
    assert perms["forget"] == MANAGE_MESSAGES
    for name in ("ask", "summarize", "image", "help", "docs"):
        assert perms[name] is None, name


async def test_message_context_menus_serialise(make_bot):
    bot = await make_bot()
    menus = bot.tree.get_commands(type=discord.AppCommandType.message)
    assert sorted(m.name for m in menus) == sorted([ASK_ABOUT_MENU, SUMMARIZE_FROM_MENU])
    for menu in menus:
        payload = menu.to_dict(bot.tree)
        assert payload["type"] == 3 and payload["dm_permission"] is False
        assert len(payload["name"]) <= 32


def test_help_embed_lists_new_features():
    embed = build_help_embed(offline=True)
    text = " ".join(f.value for f in embed.fields)
    assert "/stats" in text and ASK_ABOUT_MENU in text and SUMMARIZE_FROM_MENU in text
    assert "Offline backend" in embed.footer.text


# ------------------------------------------------------------------- /ask
async def test_ask_replies_and_stores_memory_with_guild(make_bot):
    nim = ScriptedNim(["first answer", "second answer"])
    bot = await make_bot(nim, cooldown_seconds=0)
    guild, channel = _world()
    user = FakeUser(5)
    first = FakeInteraction(guild, channel, user)
    await _cmd(bot, "ask").callback(first, prompt="hello")
    assert first.response.deferred == {"thinking": True, "ephemeral": False}
    assert first.text == "first answer"
    second = FakeInteraction(guild, channel, user)
    await _cmd(bot, "ask").callback(second, prompt="and then?")
    sent = [m["content"] for m in nim.chat_calls[1]["messages"]]
    assert sent[1:] == ["hello", "first answer", "and then?"]  # memory used
    assert await bot.db.count_history(1) == 4


async def test_precheck_rate_limit_and_allowlist(make_bot):
    bot = await make_bot(ScriptedNim(), cooldown_seconds=30, allowed_guild_ids={1})
    bot.guard._allowed = {1}
    guild, channel = _world()
    user = FakeUser(5)
    await _cmd(bot, "ask").callback(FakeInteraction(guild, channel, user), prompt="a")
    blocked = FakeInteraction(guild, channel, user)
    await _cmd(bot, "ask").callback(blocked, prompt="b")
    assert blocked.replies[0]["ephemeral"] and "Please wait" in blocked.text
    other_guild = FakeGuild(2, channels=[channel])
    denied = FakeInteraction(other_guild, channel, FakeUser(6))
    await _cmd(bot, "ask").callback(denied, prompt="c")
    assert "not enabled for this server" in denied.text


# ------------------------------------------------------------- /summarize
async def test_summarize_includes_the_newest_message_and_honest_header(make_bot):
    nim = ScriptedNim(["• summary"])
    bot = await make_bot(nim, cooldown_seconds=0)
    author, robot = FakeUser(5, "ana"), FakeUser(9, "bot", bot=True)
    messages = [FakeMessage(i, author, f"message number {i:03d} " + "x" * 60) for i in range(200)]
    messages.insert(100, FakeMessage(1000, robot, "I am a bot"))
    guild, channel = _world(messages)
    interaction = FakeInteraction(guild, channel, author)
    await _cmd(bot, "summarize").callback(interaction, count=200)
    transcript = nim.chat_calls[0]["messages"][1]["content"]
    assert "message number 199" in transcript  # regression: newest was dropped
    assert "message number 000" not in transcript
    assert "I am a bot" not in transcript
    assert len(transcript) <= bot.settings.summarize_char_budget
    header = interaction.text.splitlines()[0]
    assert header.startswith("**Summary of the last ") and " of 199 messages**" in header


# ------------------------------------------------------------- /docs
async def test_docs_ask_lists_only_cited_sources(make_bot):
    nim = ScriptedNim(["I could not find that in the server's documents."],
                      vocab=["refund", "hours", "dogs"], default_min_score=0.0)
    bot = await make_bot(nim, cooldown_seconds=0)
    await bot.rag.ingest_documents(1, [("billing.md", "refund"), ("hours.md", "hours"), ("pets.md", "dogs")])
    guild, channel = _world()
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _cmd(bot, "docs", "ask").callback(interaction, question="refund hours dogs")
    assert interaction.text == "I could not find that in the server's documents."
    assert "Sources" not in interaction.text


async def test_docs_ingest_reads_every_pin_and_pdf(make_bot):
    bot = await make_bot(cooldown_seconds=0)
    author = FakeUser(5, "ana")
    pins = [FakeMessage(i, author, f"Rule {i}: be kind to member {i}.") for i in range(60)]
    pdf = FakeAttachment(900, "policy.pdf", minimal_pdf("Refunds are accepted within 30 days."))
    pins.append(FakeMessage(61, author, "", attachments=[pdf]))
    docs_channel = FakeChannel(300, "docs", pins=pins)
    guild, channel = _world(extra_channels=[docs_channel])
    await bot.db.set_guild_field(1, "docs_channel_id", 300)
    interaction = FakeInteraction(guild, channel, author)
    await _cmd(bot, "docs", "ingest").callback(interaction)
    assert interaction.response.deferred == {"thinking": True, "ephemeral": True}
    assert "from **61** document(s)" in interaction.text  # 60 pins + the PDF, not 50
    status = FakeInteraction(guild, channel, author)
    await _cmd(bot, "docs", "status").callback(status)
    assert "offline/hashed-bow-512" in status.text and "<#300>" in status.text
    ask = FakeInteraction(guild, channel, author)
    await _cmd(bot, "docs", "ask").callback(ask, question="how many days for refunds?")
    assert "30 days" in ask.text and "pinned: policy.pdf" in ask.text


async def test_collect_documents_never_awaits_pins_and_skips_unknown_files():
    author = FakeUser(5)
    junk = FakeAttachment(1, "photo.png", b"\x89PNG....", "image/png")
    notes = FakeAttachment(2, "notes.md", b"# Notes\nhello", "text/markdown")
    channel = FakeChannel(1, pins=[FakeMessage(1, author, "pinned text", attachments=[junk, notes])])
    docs = await collect_documents(channel)
    assert docs == [("pin by user5", "pinned text"), ("pinned: notes.md", "# Notes\nhello")]


async def test_pdf_text_is_extracted_off_the_event_loop(monkeypatch):
    seen = {}
    real = rag_module.extract_pdf_text

    def spy(data):
        seen["thread"] = threading.get_ident()
        return real(data)

    monkeypatch.setattr("bot.services.extract_pdf_text", spy)
    text = await document_text("a.pdf", minimal_pdf("Hello from a PDF"))
    assert "Hello from a PDF" in text
    assert seen["thread"] != threading.get_ident()


# ------------------------------------------------------------ memory/wipe
async def test_forget_and_wipe_cover_threads(make_bot):
    bot = await make_bot(ScriptedNim(), cooldown_seconds=0)
    thread = FakeChannel(555, "a-thread")
    guild, channel = _world()
    user = FakeUser(5)
    await _cmd(bot, "ask").callback(FakeInteraction(guild, channel, user), prompt="in channel")
    await _cmd(bot, "ask").callback(FakeInteraction(guild, thread, user), prompt="secret in a thread")
    await bot.db.add_history(777, "user", "channel deleted since", guild_id=1)
    await bot.rag.ingest_documents(1, [("a.md", "some text")])

    forget = FakeInteraction(guild, channel, user)
    await _cmd(bot, "forget").callback(forget)
    assert "Cleared 2 stored message(s)" in forget.text

    no = FakeInteraction(guild, channel, user)
    await _cmd(bot, "config", "wipe").callback(no, confirm=False)
    assert "Nothing wiped" in no.text and await bot.db.count_history(1) == 3

    wipe = FakeInteraction(guild, channel, user)
    await _cmd(bot, "config", "wipe").callback(wipe, confirm=True)
    assert "Wiped: 3 memory message(s)" in wipe.text
    assert await bot.db.get_history(555, 10) == [] and await bot.db.get_history(777, 10) == []
    assert bot.rag.status(1)["chunks"] == 0


# ---------------------------------------------------------------- /stats
async def test_stats_embed_after_recorded_usage(make_bot):
    bot = await make_bot(ScriptedNim(), cooldown_seconds=0)
    guild, channel = _world()
    for uid, prompt in [(5, "a"), (5, "b"), (6, "c")]:
        await _cmd(bot, "ask").callback(FakeInteraction(guild, channel, FakeUser(uid)), prompt=prompt)
    await _cmd(bot, "help").callback(FakeInteraction(guild, channel, FakeUser(6)))
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _cmd(bot, "stats").callback(interaction)
    reply = interaction.response.sent[0]
    assert reply["ephemeral"]
    embed = reply["embed"]
    fields = {f.name: f.value for f in embed.fields}
    assert embed.description == "4 command(s) recorded"  # ask x3 + stats (help is not counted)
    assert fields["Per command"].splitlines()[0] == "`/ask` — 3"
    assert fields["Top users"].splitlines()[0] == "1. <@5> — 3"
    assert fields["Memory"] == "6 stored /ask message(s)"


# ---------------------------------------------------------------- /image
def _image_bytes(size, fmt, colour=(40, 80, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format=fmt)
    return buf.getvalue()


async def test_image_command_normalises_before_the_vision_call(make_bot):
    nim = ScriptedNim(["a blue rectangle"])
    bot = await make_bot(nim, cooldown_seconds=0)
    guild, channel = _world()
    att = FakeAttachment(1, "big.webp", _image_bytes((4000, 3000), "WEBP"), "image/webp")
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _cmd(bot, "image").callback(interaction, image=att, question="what is it?")
    call = nim.vision_calls[0]
    assert call["mime"] == "image/jpeg"
    assert Image.open(io.BytesIO(call["bytes"])).size == (1568, 1176)
    assert interaction.text.startswith("a blue rectangle\n-# Image converted WEBP to JPEG")


async def test_image_command_rejects_non_images(make_bot):
    bot = await make_bot(ScriptedNim(), cooldown_seconds=0)
    guild, channel = _world()
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    att = FakeAttachment(1, "notes.txt", b"hello", "text/plain")
    await _cmd(bot, "image").callback(interaction, image=att, question="?")
    assert "Please attach an image file" in interaction.text


# -------------------------------------------------------- context menus
async def test_ask_about_this_explains_a_text_message(make_bot):
    nim = ScriptedNim(["It means the deploy is blocked."])
    bot = await make_bot(nim, cooldown_seconds=0)
    guild, channel = _world()
    target = FakeMessage(42, FakeUser(7, "luis"), "the deploy is blocked on CI again", channel=channel)
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _menu(bot, ASK_ABOUT_MENU).callback(interaction, target)
    assert interaction.response.deferred == {"thinking": True, "ephemeral": True}
    assert interaction.followup.sent[0] == {
        "content": "It means the deploy is blocked.", "embed": None, "ephemeral": True,
    }
    prompt = nim.chat_calls[0]["messages"][1]["content"]
    assert "written by luis" in prompt and "the deploy is blocked on CI again" in prompt
    assert await bot.db.count_history(1) == 0  # does not pollute /ask memory


async def test_ask_about_this_uses_vision_for_an_image_message(make_bot):
    nim = ScriptedNim(["A red square."])
    bot = await make_bot(nim, cooldown_seconds=0)
    guild, channel = _world()
    gif = FakeAttachment(3, "anim.gif", _image_bytes((300, 200), "GIF", (200, 0, 0)), "image/gif")
    target = FakeMessage(42, FakeUser(7, "luis"), "look at this", channel=channel, attachments=[gif])
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _menu(bot, ASK_ABOUT_MENU).callback(interaction, target)
    call = nim.vision_calls[0]
    assert call["mime"] == "image/jpeg" and "by luis" in call["prompt"] and "look at this" in call["prompt"]
    assert nim.chat_calls == []
    assert interaction.text.startswith("A red square.\n-# Image converted GIF")


async def test_ask_about_this_with_nothing_to_look_at(make_bot):
    bot = await make_bot(ScriptedNim(), cooldown_seconds=0)
    guild, channel = _world()
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await _menu(bot, ASK_ABOUT_MENU).callback(interaction, FakeMessage(1, FakeUser(7), "", channel=channel))
    assert "no text or image" in interaction.text


async def test_summarize_from_here_reads_forward_from_the_selected_message(make_bot):
    nim = ScriptedNim(["• recap"])
    bot = await make_bot(nim, cooldown_seconds=0)
    author = FakeUser(5, "ana")
    messages = [FakeMessage(i, author, f"line {i}") for i in range(1, 31)]
    guild, channel = _world(messages)
    for m in messages:
        m.channel = channel
    interaction = FakeInteraction(guild, channel, author)
    await _menu(bot, SUMMARIZE_FROM_MENU).callback(interaction, messages[9])  # "line 10"
    transcript = nim.chat_calls[0]["messages"][1]["content"].splitlines()
    assert transcript[0] == "ana: line 10" and transcript[-1] == "ana: line 30"
    assert len(transcript) == 21
    assert channel.history_calls[-1]["oldest_first"] is True
    assert interaction.text.splitlines()[0] == f"**Summary of 21 messages from {messages[9].jump_url}**"


# ------------------------------------------------------------ moderation
async def test_on_message_posts_advisories_only_when_warranted(make_bot):
    bot = await make_bot(OfflineNimClient())
    mod_channel = FakeChannel(900, "mod-log")
    guild, channel = _world(extra_channels=[mod_channel])
    await bot.db.set_guild_field(1, "mod_channel_id", 900)
    await bot.db.set_guild_field(1, "moderation_on", True)
    await bot.db.add_allowlist(1, 77, "role")

    def msg(content, author):
        return FakeMessage(1, author, content, channel=channel, guild=guild)

    await bot.on_message(msg("what a paradox", FakeUser(5)))
    await bot.on_message(msg("free nitro click this link discord.gg/x", FakeUser(6, roles=[77])))
    await bot.on_message(msg("free nitro click this link discord.gg/x", FakeUser(8, bot=True)))
    assert mod_channel.sent == []
    await bot.on_message(msg("free nitro click this link discord.gg/x", FakeUser(5)))
    assert len(mod_channel.sent) == 1
    embed = mod_channel.sent[0]["embed"]
    fields = {f.name: f.value for f in embed.fields}
    assert embed.title == "Moderation advisory (suggestion only)"
    assert fields["Category"] == "scam" and fields["Severity"] == "0.85"
    assert fields["Channel"] == "<#100>" and "Go to message" in fields["Jump"]
    assert "took no action" in embed.footer.text


async def test_moderation_is_opt_in(make_bot):
    bot = await make_bot(OfflineNimClient())
    mod_channel = FakeChannel(900, "mod-log")
    guild, channel = _world(extra_channels=[mod_channel])
    await bot.db.set_guild_field(1, "mod_channel_id", 900)  # but moderation stays off
    await bot.on_message(FakeMessage(1, FakeUser(5), "I will kill you", channel=channel, guild=guild))
    assert mod_channel.sent == []


# --------------------------------------------------------- error handler
async def test_error_handler_replies_privately():
    handler = build_error_handler(None)
    guild, channel = _world()
    interaction = FakeInteraction(guild, channel, FakeUser(5))
    await handler(interaction, app_commands.MissingPermissions(["manage_guild"]))
    assert interaction.replies[0]["ephemeral"] and "permission" in interaction.text
    crashed = FakeInteraction(guild, channel, FakeUser(5))
    await crashed.response.defer()
    await handler(crashed, app_commands.AppCommandError("boom"))
    assert crashed.followup.sent[0]["ephemeral"] and "something went wrong" in crashed.text
