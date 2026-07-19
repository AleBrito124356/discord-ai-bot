"""Optional retrieval over a server's docs channel.

Ingest flow
-----------
1. An admin sets a docs channel with ``/config docs-channel``.
2. ``/docs ingest`` reads that channel's **pinned messages** and any attached
   ``.txt`` / ``.md`` / ``.pdf`` files.
3. Text is chunked, embedded through NVIDIA NIM, and stored per guild as a
   numpy matrix (``<guild>.npy``) plus a JSON sidecar with the chunk text and
   source label.

Query flow
----------
``/docs ask`` embeds the question, cosine-ranks the stored chunks, feeds the top
matches to the chat model and returns an answer with numbered citations.

The store is deliberately a flat numpy matrix: a Discord server's docs are small
(hundreds to low-thousands of chunks), so a brute-force dot product is instant and
needs no external vector database.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import Settings
from .nim_client import NimClient

log = logging.getLogger("rag")


@dataclass
class Chunk:
    text: str
    source: str


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: List[str] = field(default_factory=list)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping windows on paragraph/sentence boundaries."""
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    # Prefer to break on blank lines, then single newlines, then hard slice.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= size:
            buffer = f"{buffer}\n\n{para}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(para) <= size:
            buffer = para
        else:
            # Paragraph itself is larger than the window: hard-slice it.
            start = 0
            while start < len(para):
                chunks.append(para[start : start + size])
                start += max(1, size - overlap)
            buffer = ""
    if buffer:
        chunks.append(buffer)

    # Add a little overlap between adjacent chunks for context continuity.
    if overlap > 0 and len(chunks) > 1:
        stitched: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                stitched.append(ch)
            else:
                tail = chunks[i - 1][-overlap:]
                stitched.append(f"{tail}\n{ch}".strip())
        chunks = stitched
    return [c for c in chunks if c.strip()]


def extract_pdf_text(data: bytes) -> str:
    """Best-effort text extraction from a PDF. Returns '' if pypdf is missing."""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping PDF attachment")
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - PDFs are wild; never crash ingest
        log.warning("Failed to parse PDF: %s", exc)
        return ""


class GuildRagStore:
    """Numpy-backed vector store for one guild, persisted under ``rag_dir``."""

    def __init__(self, guild_id: int, rag_dir: Path) -> None:
        self.guild_id = guild_id
        self._vectors_path = rag_dir / f"{guild_id}.npy"
        self._meta_path = rag_dir / f"{guild_id}.json"
        self._vectors: Optional[np.ndarray] = None
        self._chunks: List[Chunk] = []
        self._load()

    def _load(self) -> None:
        if self._vectors_path.exists() and self._meta_path.exists():
            try:
                self._vectors = np.load(self._vectors_path)
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._chunks = [Chunk(**c) for c in raw]
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load RAG store for %s: %s", self.guild_id, exc)
                self._vectors = None
                self._chunks = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def replace(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """Atomically replace the store contents and persist to disk."""
        self._vectors = vectors.astype(np.float32)
        self._chunks = chunks
        np.save(self._vectors_path, self._vectors)
        self._meta_path.write_text(
            json.dumps([c.__dict__ for c in chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []
        self._vectors_path.unlink(missing_ok=True)
        self._meta_path.unlink(missing_ok=True)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if self._vectors is None or not len(self._chunks):
            return []
        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        mat = self._vectors
        mat_norms = np.linalg.norm(mat, axis=1)
        mat_norms[mat_norms == 0] = 1e-9
        sims = (mat @ q) / mat_norms
        top = np.argsort(-sims)[: max(1, k)]
        return [(self._chunks[i], float(sims[i])) for i in top]


class RagService:
    """Owns per-guild stores and orchestrates ingest + question answering."""

    def __init__(self, settings: Settings, nim: NimClient) -> None:
        self._settings = settings
        self._nim = nim
        self._stores: dict[int, GuildRagStore] = {}

    def store_for(self, guild_id: int) -> GuildRagStore:
        store = self._stores.get(guild_id)
        if store is None:
            store = GuildRagStore(guild_id, self._settings.rag_dir)
            self._stores[guild_id] = store
        return store

    async def ingest_documents(
        self, guild_id: int, documents: List[Tuple[str, str]]
    ) -> IngestReport:
        """Embed and store ``(source_label, text)`` documents for a guild."""
        report = IngestReport()
        chunks: List[Chunk] = []
        for source, text in documents:
            pieces = chunk_text(
                text,
                self._settings.rag_chunk_chars,
                self._settings.rag_chunk_overlap,
            )
            if not pieces:
                report.skipped.append(source)
                continue
            report.documents += 1
            chunks.extend(Chunk(text=p, source=source) for p in pieces)

        if not chunks:
            return report

        # Embed in batches to stay well under request-size limits.
        vectors: List[List[float]] = []
        batch = 32
        for i in range(0, len(chunks), batch):
            window = [c.text for c in chunks[i : i + batch]]
            vectors.extend(await self._nim.embed(window, input_type="passage"))

        matrix = np.array(vectors, dtype=np.float32)
        self.store_for(guild_id).replace(matrix, chunks)
        report.chunks = len(chunks)
        return report

    async def answer(
        self, guild_id: int, question: str
    ) -> Tuple[str, List[Chunk]]:
        """Return ``(answer_text, cited_chunks)`` for a question."""
        store = self.store_for(guild_id)
        if store.size == 0:
            return (
                "No documents have been ingested yet. An admin can run "
                "`/docs ingest` after setting a docs channel with "
                "`/config docs-channel`.",
                [],
            )
        query_vecs = await self._nim.embed([question], input_type="query")
        query_vec = np.array(query_vecs[0], dtype=np.float32)
        hits = store.search(query_vec, self._settings.rag_top_k)
        cited = [chunk for chunk, _ in hits]

        context_blocks = []
        for idx, chunk in enumerate(cited, start=1):
            context_blocks.append(f"[{idx}] Source: {chunk.source}\n{chunk.text}")
        context = "\n\n".join(context_blocks)

        system = (
            "You answer questions strictly from the provided context excerpts. "
            "Cite the sources you use with bracketed numbers like [1] that match "
            "the excerpt numbers. If the answer is not in the context, say you "
            "could not find it in the server's documents. Do not invent citations."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        answer = await self._nim.chat(messages, temperature=0.2)
        return answer, cited
