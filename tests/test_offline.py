"""The deterministic offline backend and the services running on top of it."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

from PIL import Image

from bot.moderation import ModerationService, parse_verdict
from bot.offline import OFFLINE_TAG, OfflineNimClient, hashed_embedding, split_sentences
from bot.prompts import MODERATION_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT
from bot.rag import NOT_FOUND_TEXT, RagService
from bot.services import BotCore

REPO = Path(__file__).resolve().parent.parent
DOCS = [
    ("billing.md", "Refund requests are accepted within 30 days of purchase. Billing runs monthly."),
    ("hours.md", "The office is open from 9 to 5 on weekdays. Support answers tickets within a day."),
    ("pets.md", "Dogs and other pets are welcome in the lobby but not in meeting rooms."),
]


def test_hashed_embedding_is_deterministic_and_normalised():
    a = hashed_embedding("Refund requests are accepted within 30 days")
    assert a == hashed_embedding("Refund requests are accepted within 30 days")
    assert len(a) == 512
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)
    assert hashed_embedding("the of and") == [0.0] * 512  # only stopwords


def test_split_sentences_joins_wrapped_prose_and_keeps_code_lines():
    text = "First sentence wraps\nonto a second line. Second one!\n\n## Heading\n```bash\nrun --this\n```"
    assert split_sentences(text) == [
        "First sentence wraps onto a second line.",
        "Second one!",
        "Heading",
        "run --this",
    ]


def test_split_sentences_detects_a_chunk_that_starts_inside_code():
    text = "pip install x\nrun it\n```\n\nBack to prose here."
    assert split_sentences(text) == ["pip install x", "run it", "Back to prose here."]


async def test_retrieval_ranks_by_overlap_and_answer_cites_the_right_source(settings):
    nim = OfflineNimClient()
    service = RagService(settings, nim)
    await service.ingest_documents(1, DOCS)
    result = await service.answer(1, "How many days do I have to request a refund?")
    assert result.found
    assert result.hits[0][0].source == "billing.md"
    assert "30 days" in result.text and "[1]" in result.text
    assert [c.source for _, c in result.sources] == ["billing.md"]
    assert result.text.startswith(OFFLINE_TAG)


async def test_offline_docs_answer_not_found(settings):
    service = RagService(settings, OfflineNimClient())
    await service.ingest_documents(1, DOCS)
    result = await service.answer(1, "what is the wifi password?")
    assert result.text == NOT_FOUND_TEXT and result.sources == []


async def test_end_to_end_on_the_repository_setup_guide(settings):
    async with BotCore(settings, nim=OfflineNimClient()) as core:
        text = (REPO / "docs" / "setup.md").read_text(encoding="utf-8")
        ingest = await core.docs.ingest(1, [("setup.md", text)])
        assert ingest.ok and "chunks" in ingest.text
        reply = await core.docs.ask(1, "what permission integer does the invite URL use?")
        assert reply.ok
        assert "84992" in reply.text
        assert "**Sources**" in reply.text and "setup.md" in reply.text


async def test_offline_summary_is_extractive_ordered_and_labelled():
    transcript = (REPO / "examples" / "standup.txt").read_text(encoding="utf-8")
    out = await OfflineNimClient().chat(
        [{"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT}, {"role": "user", "content": transcript}]
    )
    bullets = [line for line in out.splitlines() if line.startswith("• ")]
    assert 3 <= len(bullets) <= 7
    assert any(b.startswith("• Decision: **Ana**: ok, we decided to ship v2") for b in bullets)
    assert any(b.startswith("• Question:") for b in bullets)
    assert not any("lol" in b or "+1" in b for b in bullets)
    # Every bullet quotes a real line, in the original order.
    lines = transcript.splitlines()
    positions = [
        next(i for i, line in enumerate(lines) if line.split(": ", 1)[1] in b) for b in bullets
    ]
    assert positions == sorted(positions)


async def test_offline_moderation_verdicts_parse_and_classify():
    nim = OfflineNimClient()

    async def verdict(text):
        user = f'Automated prefilter noticed: x.\n\nMessage:\n"""\n{text}\n"""'
        raw = await nim.chat(
            [{"role": "system", "content": MODERATION_SYSTEM_PROMPT}, {"role": "user", "content": user}]
        )
        json.loads(raw)  # always valid JSON
        return parse_verdict(raw)

    assert (await verdict("free nitro click this link")).category == "scam"
    assert (await verdict("I will kill you")).category == "threat"
    calm = await verdict("see you at the standup")
    assert calm.flag is False and calm.severity == 0.0


async def test_moderation_service_on_offline_backend():
    service = ModerationService(None, OfflineNimClient())
    scam = await service.review_text("free nitro click this link discord.gg/x", 0, 0.6)
    assert scam.llm_called and scam.report and scam.verdict.category == "scam"
    paradox = await service.review_text("what a paradox", 0, 0.6)
    assert not paradox.llm_called and not paradox.report


async def test_offline_vision_measures_the_image():
    buf = io.BytesIO()
    Image.new("RGB", (320, 200), (220, 30, 30)).save(buf, format="PNG")
    out = await OfflineNimClient().vision("what is it?", buf.getvalue(), "image/png")
    assert out.startswith(OFFLINE_TAG)
    assert "320x200 landscape PNG" in out and "red 100%" in out
    assert "not an image" in await OfflineNimClient().vision("q", b"nope")


async def test_offline_small_talk_is_labelled_and_counts_memory(settings):
    async with BotCore(settings, nim=OfflineNimClient()) as core:
        first = await core.ask.ask(1, 10, "hello there")
        second = await core.ask.ask(1, 10, "and again")
    assert first.text.startswith(OFFLINE_TAG) and "hello there" in first.text
    assert "persona Assistant, 0 earlier" in first.text
    assert "2 earlier message(s)" in second.text
