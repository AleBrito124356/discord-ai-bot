"""RAG: chunking, atomic store, integrity checks, relevance floor, citations."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bot.rag import (
    NOT_FOUND_TEXT,
    Chunk,
    GuildRagStore,
    RagIndexError,
    RagService,
    chunk_text,
    parse_citations,
)
from fakes import ScriptedNim

VOCAB = ["refund", "days", "hours", "open", "dogs", "pets", "billing"]
DOCS = [
    ("billing.md", "Refund requests are accepted within 30 days. Billing runs monthly."),
    ("hours.md", "The office is open from 9 to 5. Support hours are weekdays."),
    ("pets.md", "Dogs and other pets are welcome in the lobby."),
]


# ----------------------------------------------------------------- chunking
def test_chunk_text_respects_size_and_is_non_empty():
    text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(50))
    chunks = chunk_text(text, size=200, overlap=20)
    assert chunks
    assert all(len(c) <= 200 + 20 + 40 for c in chunks)
    assert "Paragraph number 0" in chunks[0]


def test_chunk_text_hard_slices_oversized_paragraph():
    chunks = chunk_text("x" * 1000, size=200, overlap=0)
    assert len(chunks) >= 5
    assert "".join(chunks) == "x" * 1000


def test_chunk_text_empty():
    assert chunk_text("  \r\n ", 100, 10) == []


# -------------------------------------------------------------- citations
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Refunds take 30 days [1].", [1]),
        ("See [2] and [1][3].", [1, 2, 3]),
        ("Both [1, 3] agree.", [1, 3]),
        ("Covered in [1-3].", [1, 2, 3]),
        ("Out of range [9] and [0].", []),
        ("I could not find that in the server's documents.", []),
    ],
)
def test_parse_citations(answer, expected):
    assert parse_citations(answer, 3) == expected


# ------------------------------------------------------------------- store
def test_store_roundtrip_is_atomic_single_file(tmp_path):
    store = GuildRagStore(1, tmp_path)
    vectors = np.eye(3, dtype=np.float32)
    chunks = [Chunk("a", "s1"), Chunk("b", "s2"), Chunk("c", "s3")]
    store.replace(vectors, chunks, embed_model="m1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1.npz"]  # no temp leftovers

    reloaded = GuildRagStore(1, tmp_path)
    assert reloaded.size == 3 and reloaded.load_error is None
    assert (reloaded.meta.embed_model, reloaded.meta.dim, reloaded.meta.rows) == ("m1", 3, 3)
    hits = reloaded.search(np.array([0, 1, 0]), k=1)
    assert hits[0][0].text == "b" and hits[0][1] == pytest.approx(1.0)


def test_store_rejects_mismatched_rows_on_write(tmp_path):
    with pytest.raises(ValueError):
        GuildRagStore(1, tmp_path).replace(np.eye(2), [Chunk("a", "s")] * 3)


def test_legacy_store_with_mismatched_rows_is_rejected_not_mispaired(tmp_path):
    # Regression: a .npy with 2 rows next to a .json with 3 chunks loaded
    # silently and search paired vectors with the wrong text.
    np.save(tmp_path / "1.npy", np.eye(2, 3, dtype=np.float32))
    (tmp_path / "1.json").write_text(
        json.dumps([{"text": t, "source": "s"} for t in "abc"]), encoding="utf-8"
    )
    store = GuildRagStore(1, tmp_path)
    assert store.size == 0
    assert "2 vectors but 3 chunks" in store.load_error
    assert "/docs ingest" in store.load_error


def test_legacy_store_still_loads_when_consistent(tmp_path):
    np.save(tmp_path / "1.npy", np.eye(2, dtype=np.float32))
    (tmp_path / "1.json").write_text(
        json.dumps([{"text": "a", "source": "s"}, {"text": "b", "source": "t"}]),
        encoding="utf-8",
    )
    store = GuildRagStore(1, tmp_path)
    assert store.size == 2 and store.meta.format == 1 and store.meta.embed_model is None
    # The next ingest replaces the legacy pair with the single-file format.
    store.replace(np.eye(2), [Chunk("x", "s"), Chunk("y", "t")], embed_model="m")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1.npz"]


def test_corrupt_store_file_reports_a_clear_error(tmp_path):
    (tmp_path / "1.npz").write_bytes(b"definitely not a zip file")
    store = GuildRagStore(1, tmp_path)
    assert store.size == 0 and "Re-run `/docs ingest`" in store.load_error


def test_dimension_mismatch_raises_clear_error(tmp_path):
    store = GuildRagStore(1, tmp_path)
    store.replace(np.ones((2, 1024)), [Chunk("a", "s"), Chunk("b", "s")], embed_model="m")
    with pytest.raises(RagIndexError, match="1024-d.*4096-d.*re-run `/docs ingest`"):
        store.search(np.ones(4096), k=2)


def test_min_score_filters_hits(tmp_path):
    store = GuildRagStore(1, tmp_path)
    store.replace(np.eye(2), [Chunk("a", "s"), Chunk("b", "t")], embed_model="m")
    assert [c.text for c, _ in store.search(np.array([1.0, 0.1]), k=2, min_score=0.5)] == ["a"]


# ----------------------------------------------------------------- service
async def _ingested(settings, nim):
    service = RagService(settings, nim)
    report = await service.ingest_documents(1, DOCS + [("empty.md", "   ")])
    assert (report.documents, report.chunks, report.skipped) == (3, 3, ["empty.md"])
    assert report.embed_model == nim.embed_model_name
    return service


async def test_answer_lists_only_cited_sources(settings):
    nim = ScriptedNim(["Refunds are accepted within 30 days [1]."], vocab=VOCAB, default_min_score=0.1)
    service = await _ingested(settings, nim)
    result = await service.answer(1, "how many days for a refund?")
    assert result.found
    assert [(n, c.source) for n, c in result.sources] == [(1, "billing.md")]
    assert result.formatted().endswith("**Sources**\n[1] billing.md")
    context = nim.chat_calls[0]["messages"][1]["content"]
    assert "[1] Source: billing.md" in context and "pets.md" not in context  # floor


async def test_not_found_answer_has_no_sources_footer(settings):
    # Regression: every retrieved chunk was listed even when nothing was cited.
    nim = ScriptedNim(
        ["I could not find that in the server's documents."], vocab=VOCAB, default_min_score=0.0
    )
    service = await _ingested(settings, nim)
    result = await service.answer(1, "refund dogs hours")
    assert not result.found and result.sources == []
    assert "**Sources**" not in result.formatted()


async def test_nothing_relevant_skips_the_chat_call(settings):
    nim = ScriptedNim(vocab=VOCAB, default_min_score=0.3)
    service = await _ingested(settings, nim)
    result = await service.answer(1, "what is the wifi password?")
    assert result.text == NOT_FOUND_TEXT and nim.chat_calls == []


async def test_env_min_score_overrides_backend_default(settings):
    settings.rag_min_score = 0.99
    nim = ScriptedNim(vocab=VOCAB, default_min_score=0.0)
    service = await _ingested(settings, nim)
    assert service.min_score == 0.99
    assert (await service.answer(1, "refund billing hours")).text == NOT_FOUND_TEXT


async def test_index_built_with_another_embedder_is_rejected(settings):
    service = await _ingested(settings, ScriptedNim(vocab=VOCAB, embed_model_name="old/model"))
    fresh = RagService(settings, ScriptedNim(vocab=VOCAB, embed_model_name="new/model"))
    with pytest.raises(RagIndexError, match="old/model.*new/model.*re-run `/docs ingest`"):
        await fresh.answer(1, "refund?")
    assert service.status(1)["embed_model"] == "old/model"


async def test_no_index_message_and_forget(settings):
    service = RagService(settings, ScriptedNim(vocab=VOCAB))
    assert "No documents have been ingested" in (await service.answer(5, "q")).text
    await service.ingest_documents(5, DOCS)
    service.forget(5)
    assert service.status(5)["chunks"] == 0
    assert not list(settings.rag_dir.glob("5.*"))
