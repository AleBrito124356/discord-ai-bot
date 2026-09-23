"""Fence-aware message splitting and the newest-first summarize transcript."""
from __future__ import annotations

from bot.textutil import build_transcript, split_message


def _code_block(lines: int, lang: str = "python") -> str:
    body = "\n".join(f"value_{i} = compute({i})  # line {i}" for i in range(lines))
    return f"```{lang}\n{body}\n```"


def test_split_keeps_every_part_fenced_correctly():
    # Regression: parts had fence counts [1, 0, 1] and rendered broken.
    parts = split_message(_code_block(120), 1900)
    assert len(parts) >= 2
    assert all(len(p) <= 1900 for p in parts)
    assert all(p.count("```") % 2 == 0 for p in parts)
    assert all(p.startswith("```python") for p in parts)  # language tag re-opened


def test_split_mixed_prose_and_code_stays_balanced():
    text = "Intro paragraph.\n" + _code_block(60, "js") + "\nMiddle text.\n" + _code_block(60)
    parts = split_message(text, 500)
    assert all(len(p) <= 500 for p in parts)
    assert all(p.count("```") % 2 == 0 for p in parts)
    # No code line is lost or duplicated by the split.
    joined = "\n".join(parts)
    original_lines = [l for l in text.split("\n") if not l.startswith("```")]
    split_lines = [l for l in joined.split("\n") if not l.startswith("```")]
    assert split_lines == original_lines


def test_split_hard_cuts_a_single_huge_line_inside_a_fence():
    parts = split_message("```\n" + "a" * 5000 + "\n```", 1900)
    assert all(len(p) <= 1900 and p.count("```") == 2 for p in parts)
    assert "".join(p.replace("```", "").replace("\n", "") for p in parts) == "a" * 5000


def test_split_plain_text_preserves_content():
    text = "\n".join(f"line {i}" for i in range(1000))
    parts = split_message(text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")


def test_split_short_and_empty():
    assert split_message("hello") == ["hello"]
    assert split_message("") == ["(empty response)"]


def test_transcript_keeps_the_newest_messages():
    # Regression: the transcript was cut with [:12000] and kept the OLDEST.
    lines = [f"user{i}: message number {i:03d} " + "x" * 50 for i in range(200)]
    transcript = build_transcript(lines, 12_000)
    assert lines[-1] in transcript.text
    assert lines[0] not in transcript.text
    assert len(transcript.text) <= 12_000
    assert transcript.truncated
    assert transcript.header() == (
        f"**Summary of the last {transcript.included} of 200 messages** (older ones did not fit)"
    )


def test_transcript_fits_entirely():
    transcript = build_transcript(["a: hi", "b: hello"], 100)
    assert transcript.text == "a: hi\nb: hello"
    assert not transcript.truncated
    assert transcript.header() == "**Summary of the last 2 messages**"


def test_transcript_cuts_a_single_oversized_newest_line():
    transcript = build_transcript(["old", "n" * 50], 10)
    assert transcript.text == "n" * 10 and transcript.included == 1
