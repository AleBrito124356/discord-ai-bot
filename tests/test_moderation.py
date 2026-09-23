"""Moderation: word-boundary prefilter, strict verdict parsing, review flow."""
from __future__ import annotations

import pytest

from bot.moderation import (
    ModerationService,
    _extract_json,
    parse_verdict,
    prefilter,
    prefilter_floor,
)
from bot.nim_client import NimError
from fakes import ScriptedNim

# ---------------------------------------------------------------- prefilter
BENIGN = [
    "What a paradox, right?",
    "I slurped my coffee",
    "Use fire retardant paint",
    "an unorthodox approach",
    "the orthodox view on paradoxes",
    "hey team, what time is the standup tomorrow?",
    "I will find the docs later",
    "this is a skyscraper",
]


@pytest.mark.parametrize("text", BENIGN)
def test_prefilter_ignores_substrings_of_ordinary_words(text):
    result = prefilter(text, mention_count=0)
    assert result.score == 0.0, (text, result.reasons)
    # ...and therefore never costs an LLM call at the default threshold.
    assert result.score < prefilter_floor(0.6)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("I will kill you", "threat"),
        ("i'll hurt you tomorrow", "threat"),
        ("we will dox him", "threat"),
        ("he got doxxed yesterday", "threat"),
        ("kys", "slur"),
        ("you are retarded", "slur"),
        ("FREE NITRO here", "scam"),
        ("click this link to claim your prize", "scam"),
    ],
)
def test_prefilter_still_catches_real_signals(text, reason):
    result = prefilter(text, mention_count=0)
    assert result.score > 0
    assert any(reason in r for r in result.reasons), result.reasons


def test_prefilter_structural_signals():
    assert "mass mention (8 pings)" in prefilter("look", 8).reasons
    assert any("link spam" in r for r in prefilter("http://a http://b https://c", 0).reasons)
    assert any("repeated" in r for r in prefilter("aaaaaaaaaaaaaa", 0).reasons)
    assert any("shouting" in r for r in prefilter("THIS IS ALL CAPS TEXT", 0).reasons)
    assert any("invite" in r for r in prefilter("join discord.gg/abc", 0).reasons)
    assert prefilter("I will kill you and dox you http://x http://y https://z", 9).score == 1.0


# ---------------------------------------------------------- verdict parsing
@pytest.mark.parametrize(
    "raw,flag,severity,category",
    [
        ('{"flag": "false", "severity": 0.9, "category": "none"}', False, 0.9, "none"),
        ('{"flag": "true", "severity": "0.8", "category": "SCAM"}', True, 0.8, "scam"),
        ('{"flag": 1, "severity": 85, "category": "threat"}', True, 0.85, "threat"),
        ('{"flag": true, "severity": 8, "category": "spam"}', True, 0.8, "spam"),
        ('{"flag": true, "severity": "70%", "category": "hate"}', True, 0.7, "hate"),
        ('{"flag": true, "severity": 1000, "category": "hate"}', True, 1.0, "hate"),
        ('{"flag": true, "severity": -3, "category": "hate"}', True, 0.0, "hate"),
        ('{"flag": true, "severity": 0.9, "category": "made-up"}', True, 0.9, "other"),
        ('{"flag": true, "severity": 0.9, "category": "none"}', True, 0.9, "other"),
        ('{"flag": false, "severity": 0.2, "category": "threat"}', False, 0.2, "none"),
    ],
)
def test_parse_verdict_coerces_strictly(raw, flag, severity, category):
    verdict = parse_verdict(raw)
    assert verdict is not None
    assert verdict.flag is flag
    assert verdict.severity == pytest.approx(severity)
    assert verdict.category == category


@pytest.mark.parametrize(
    "raw",
    [
        '{"flag": "maybe", "severity": 0.9}',
        '{"flag": true, "severity": "very high"}',
        '{"flag": 7, "severity": 0.9}',
        '{"flag": true, "severity": NaN}',
        "no json here",
    ],
)
def test_parse_verdict_rejects_unusable_replies(raw):
    assert parse_verdict(raw) is None


def test_extract_json_handles_two_brace_groups_and_prose():
    raw = 'Verdict: {"flag": true, "severity": 0.9} (schema was {"flag": bool})'
    assert _extract_json(raw) == {"flag": True, "severity": 0.9}


def test_extract_json_from_fenced_block():
    raw = '```json\n{"flag": true, "severity": 0.9, "category": "threat"}\n```'
    assert _extract_json(raw) == {"flag": True, "severity": 0.9, "category": "threat"}


def test_extract_json_skips_arrays_and_broken_objects():
    assert _extract_json('[1, 2] {"flag": false}') == {"flag": False}
    assert _extract_json('{"broken": } then {"ok": 1}') == {"ok": 1}
    assert _extract_json("not json at all") is None


# ------------------------------------------------------------- review flow
async def test_review_below_floor_never_calls_the_model():
    nim = ScriptedNim()
    service = ModerationService(None, nim)
    result = await service.review_text("What a paradox", 0, threshold=0.6)
    assert not result.llm_called and not result.report
    assert nim.chat_calls == []


async def test_review_reports_only_flagged_verdicts_over_threshold():
    nim = ScriptedNim(
        [
            '{"flag": true, "category": "threat", "severity": 0.9, "rationale": "Direct threat."}',
            '{"flag": true, "category": "threat", "severity": 0.4, "rationale": "Mild."}',
            '{"flag": "false", "category": "none", "severity": 0.95, "rationale": "Joke."}',
        ]
    )
    service = ModerationService(None, nim)
    reported = await service.review_text("I will kill you", 0, threshold=0.6)
    assert reported.report and reported.verdict.category == "threat"
    weak = await service.review_text("I will kill you", 0, threshold=0.6)
    assert weak.llm_called and not weak.report
    string_false = await service.review_text("I will kill you", 0, threshold=0.6)
    assert string_false.verdict.flag is False and not string_false.report
    # The model saw the message and the heuristic reasons.
    user_prompt = nim.chat_calls[0]["messages"][1]["content"]
    assert "I will kill you" in user_prompt and "threat" in user_prompt


async def test_review_survives_nim_errors():
    class FailingNim(ScriptedNim):
        async def chat(self, messages, **kwargs):
            raise NimError("down")

    result = await ModerationService(None, FailingNim()).review_text(
        "I will kill you", 0, threshold=0.6
    )
    assert result.llm_called and result.verdict is None and not result.report
