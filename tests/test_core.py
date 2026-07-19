"""Offline unit tests for the pure logic. No network, no Discord, no NIM.

Run with:  pytest
These cover the deterministic building blocks: chunking, message splitting, the
moderation prefilter, JSON extraction, ID parsing and the rate limiter.
"""
from __future__ import annotations

from bot.config import _parse_ids
from bot.main import split_message
from bot.middleware import RateLimiter
from bot.moderation import _extract_json, prefilter
from bot.rag import chunk_text


def test_parse_ids_handles_mixed_separators_and_junk():
    assert _parse_ids("1, 2;3 , x, 4") == {1, 2, 3, 4}
    assert _parse_ids("") == set()
    assert _parse_ids(None) == set()


def test_chunk_text_respects_size_and_is_non_empty():
    text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(50))
    chunks = chunk_text(text, size=200, overlap=20)
    assert chunks, "expected at least one chunk"
    # Overlap can push chunks slightly past `size`; allow a small margin.
    assert all(len(c) <= 200 + 20 + 40 for c in chunks)
    assert "Paragraph number 0" in chunks[0]


def test_chunk_text_hard_slices_oversized_paragraph():
    text = "x" * 1000
    chunks = chunk_text(text, size=200, overlap=0)
    assert len(chunks) >= 5
    assert "".join(chunks) == text


def test_split_message_keeps_chunks_under_limit():
    text = "\n".join("line " + str(i) for i in range(1000))
    parts = split_message(text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert "".join(p for p in parts).replace("\n", "") == text.replace("\n", "")


def test_split_message_short_text_is_single_chunk():
    assert split_message("hello") == ["hello"]


def test_prefilter_flags_threats_and_scores_high():
    result = prefilter("I will kill you and dox your family", mention_count=0)
    assert result.score >= 0.6
    assert any("threat" in r for r in result.reasons)


def test_prefilter_ignores_normal_chat():
    result = prefilter("hey team, what time is the standup tomorrow?", mention_count=0)
    assert result.score == 0.0
    assert result.reasons == []


def test_prefilter_detects_mass_mentions():
    result = prefilter("look here everyone", mention_count=8)
    assert result.score > 0.0
    assert any("mass mention" in r for r in result.reasons)


def test_extract_json_from_fenced_block():
    raw = '```json\n{"flag": true, "severity": 0.9, "category": "threat"}\n```'
    data = _extract_json(raw)
    assert data == {"flag": True, "severity": 0.9, "category": "threat"}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


def test_rate_limiter_enforces_cooldown_then_allows():
    limiter = RateLimiter(cooldown_seconds=0.0, per_minute=2)
    assert limiter.check(42)[0] is True
    assert limiter.check(42)[0] is True
    allowed, retry = limiter.check(42)
    assert allowed is False
    assert retry > 0
