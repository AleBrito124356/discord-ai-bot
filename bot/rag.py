"""Optional retrieval over a server's docs channel.

Ingest flow
-----------
1. An admin sets a docs channel with ``/config docs-channel``.
2. ``/docs ingest`` reads that channel's **pinned messages** (all of them, not
   just the first page) and any attached ``.txt`` / ``.md`` / ``.pdf`` files.
3. Text is chunked, embedded, and stored per guild as ONE ``<guild>.npz`` file
   holding the vector matrix, the chunk texts/sources and a metadata record
   (embedding model, dimension, row count, build time).

Query flow
----------
``/docs ask`` embeds the question, cosine-ranks the stored chunks, drops the ones
under the relevance floor (``RAG_MIN_SCORE``), feeds the rest to the chat model
and returns an answer. Only the excerpts the answer actually cites (``[n]``) are
listed as sources; an answer that cites nothing gets no "Sources" footer.

Integrity
---------
The store is written to a temporary file and moved into place with
``os.replace``, so a crash mid-ingest leaves the previous index intact. On load
the row count, dimension and embedding model are validated; an index built with
a different embedding model is rejected with a clear "re-run /docs ingest"
message instead of silently mis-citing. Legacy ``.npy`` + ``.json`` pairs from
version 1.0 are still read (and validated) until the next ingest replaces them.

The store is deliberately a flat numpy matrix: a Discord server's docs are small
(hundreds to low-thousands of chunks), so a brute-force dot product is instant and
needs no external vector database.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import RAG_SYSTEM_PROMPT

log = logging.getLogger("rag")

STORE_FORMAT = 2
NOT_FOUND_TEXT = "I could not find anything about that in the server's documents."
NO_INDEX_TEXT = (
    "No documents have been ingested yet. An admin can run `/docs ingest` after "
    "setting a docs channel with `/config docs-channel`."
)

class RagIndexError(RuntimeError):
    """The stored index can not be used as-is. The message is user-safe."""


@dataclass
class Chunk:
    text: str
    source: str


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: List[str] = field(default_factory=list)
    embed_model: str = ""
    dim: int = 0


@dataclass
class StoreMeta:
    embed_model: Optional[str]
    dim: int
    rows: int
    built_at: float
    format: int = STORE_FORMAT


@dataclass
class RagAnswer:
    """Result of a docs question."""

    text: str
    sources: List[Tuple[int, Chunk]]
    hits: List[Tuple[Chunk, float]]
    found: bool
    indexed: bool = True  # False when the guild has no docs index at all

    def formatted(self) -> str:
        """The answer followed by a Sources footer listing only cited excerpts."""
        if not self.sources:
            return self.text
        lines = [f"[{n}] {chunk.source}" for n, chunk in self.sources]
        return f"{self.text}\n\n**Sources**\n" + "\n".join(lines)


# ------------------------------------------------------------------ helpers
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
                stitched.append(f"{_overlap_tail(chunks[i - 1], overlap)}\n{ch}".strip())
        chunks = stitched
    return [c for c in chunks if c.strip()]


def _overlap_tail(text: str, overlap: int) -> str:
    """The last ~``overlap`` characters, starting on a line or word boundary."""
    tail = text[-overlap:]
    if len(tail) < len(text):
        for sep in ("\n", " "):
            cut = tail.find(sep)
            if 0 <= cut < len(tail) - 1:
                return tail[cut + 1 :]
    return tail


def extract_pdf_text(data: bytes) -> str:
    """Best-effort text extraction from a PDF. Returns '' if pypdf is missing.

    This is CPU-bound; async callers should run it with ``asyncio.to_thread``.
    """
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


_CITATION_RE = re.compile(r"\[(\d+(?:\s*(?:[,;]|[-–])\s*\d+)*)\]")


def parse_citations(answer: str, available: int) -> List[int]:
    """Return the sorted excerpt numbers an answer cites, within 1..available.

    Understands ``[2]``, ``[1][3]``, ``[1, 3]`` and ranges like ``[1-3]``.
    """
    cited = set()
    for match in _CITATION_RE.finditer(answer or ""):
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            bounds = re.split(r"\s*[-–]\s*", part)
            try:
                numbers = [int(b) for b in bounds if b]
            except ValueError:
                continue
            if len(numbers) == 2 and numbers[0] <= numbers[1] <= available:
                cited.update(range(numbers[0], numbers[1] + 1))
            else:
                cited.update(numbers)
    return sorted(n for n in cited if 1 <= n <= available)


def _encode_blob(text: str) -> np.ndarray:
    """UTF-8 text as a uint8 array (loads without pickle, 1 byte per ASCII char)."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _decode_blob(array: np.ndarray) -> str:
    return np.asarray(array, dtype=np.uint8).tobytes().decode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# -------------------------------------------------------------------- store
class GuildRagStore:
    """Numpy-backed vector store for one guild, persisted under ``rag_dir``."""

    def __init__(self, guild_id: int, rag_dir: Path) -> None:
        self.guild_id = guild_id
        self._path = rag_dir / f"{guild_id}.npz"
        # Version 1.0 layout, read-only fallback.
        self._legacy_vectors = rag_dir / f"{guild_id}.npy"
        self._legacy_chunks = rag_dir / f"{guild_id}.json"
        self._vectors: Optional[np.ndarray] = None
        self._chunks: List[Chunk] = []
        self.meta: Optional[StoreMeta] = None
        self.load_error: Optional[str] = None
        self._load()

    # ----------------------------------------------------------- loading
    def _load(self) -> None:
        try:
            if self._path.exists():
                self._load_npz()
            elif self._legacy_vectors.exists() and self._legacy_chunks.exists():
                self._load_legacy()
        except Exception as exc:  # noqa: BLE001 - never crash on a bad file
            log.warning("Could not load RAG store for %s: %s", self.guild_id, exc)
            self._vectors, self._chunks, self.meta = None, [], None
            self.load_error = (
                "The docs index on disk is unreadable or inconsistent "
                f"({exc}). Re-run `/docs ingest` to rebuild it."
            )

    def _load_npz(self) -> None:
        with np.load(self._path, allow_pickle=False) as data:
            vectors = np.array(data["vectors"], dtype=np.float32)
            chunks_raw = json.loads(_decode_blob(data["chunks"]))
            meta_raw = json.loads(_decode_blob(data["meta"]))
        meta = StoreMeta(**meta_raw)
        chunks = [Chunk(**c) for c in chunks_raw]
        self._validate(vectors, chunks, meta)
        self._vectors, self._chunks, self.meta = vectors, chunks, meta

    def _load_legacy(self) -> None:
        vectors = np.load(self._legacy_vectors, allow_pickle=False).astype(np.float32)
        chunks = [
            Chunk(**c)
            for c in json.loads(self._legacy_chunks.read_text(encoding="utf-8"))
        ]
        meta = StoreMeta(
            embed_model=None,  # v1 did not record it
            dim=int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            rows=len(chunks),
            built_at=self._legacy_vectors.stat().st_mtime,
            format=1,
        )
        self._validate(vectors, chunks, meta)
        self._vectors, self._chunks, self.meta = vectors, chunks, meta

    @staticmethod
    def _validate(vectors: np.ndarray, chunks: List[Chunk], meta: StoreMeta) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vector matrix has {vectors.ndim} dimensions, expected 2")
        rows, dim = vectors.shape
        if rows != len(chunks):
            raise ValueError(f"{rows} vectors but {len(chunks)} chunks")
        if rows != meta.rows or dim != meta.dim:
            raise ValueError(
                f"metadata says {meta.rows}x{meta.dim}, file holds {rows}x{dim}"
            )

    # ----------------------------------------------------------- queries
    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> List[str]:
        return sorted({c.source for c in self._chunks})

    def check_model(self, embed_model: str) -> None:
        """Raise RagIndexError if this index was built by another embedder."""
        if self.meta and self.meta.embed_model and self.meta.embed_model != embed_model:
            raise RagIndexError(
                f"This server's docs index was built with the embedding model "
                f"`{self.meta.embed_model}` ({self.meta.dim}-d), but the bot now "
                f"embeds with `{embed_model}`. An admin needs to re-run "
                "`/docs ingest` to rebuild it."
            )

    def search(
        self, query_vec: np.ndarray, k: int, min_score: float = -1.0
    ) -> List[Tuple[Chunk, float]]:
        if self._vectors is None or not len(self._chunks):
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise RagIndexError(
                f"This server's docs index holds {self._vectors.shape[1]}-d vectors "
                f"but the current embedding model returns {q.shape[0]}-d vectors. "
                "An admin needs to re-run `/docs ingest` to rebuild it."
            )
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        mat = self._vectors
        mat_norms = np.linalg.norm(mat, axis=1)
        mat_norms[mat_norms == 0] = 1e-9
        sims = (mat @ q) / mat_norms
        top = np.argsort(-sims, kind="stable")[: max(1, k)]
        return [
            (self._chunks[i], float(sims[i])) for i in top if float(sims[i]) >= min_score
        ]

    # ----------------------------------------------------------- writes
    def replace(
        self, vectors: np.ndarray, chunks: List[Chunk], *, embed_model: Optional[str] = None
    ) -> None:
        """Replace the store contents and persist them atomically.

        Everything (vectors, chunks, metadata) goes into a single ``.npz`` that
        is written to a temp file and ``os.replace``-d into place, so readers see
        either the old index or the new one, never a mix.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"cannot store {vectors.shape} vectors for {len(chunks)} chunks"
            )
        meta = StoreMeta(
            embed_model=embed_model,
            dim=int(vectors.shape[1]),
            rows=len(chunks),
            built_at=time.time(),
        )
        buffer = io.BytesIO()
        np.savez(
            buffer,
            vectors=vectors,
            chunks=_encode_blob(json.dumps([asdict(c) for c in chunks], ensure_ascii=False)),
            meta=_encode_blob(json.dumps(asdict(meta))),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._path, buffer.getvalue())
        # The legacy pair is superseded; remove it so it can never be re-read.
        self._legacy_vectors.unlink(missing_ok=True)
        self._legacy_chunks.unlink(missing_ok=True)
        self._vectors, self._chunks, self.meta = vectors, list(chunks), meta
        self.load_error = None

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []
        self.meta = None
        self.load_error = None
        for path in (self._path, self._legacy_vectors, self._legacy_chunks):
            path.unlink(missing_ok=True)


# ------------------------------------------------------------------ service
class RagService:
    """Owns per-guild stores and orchestrates ingest + question answering."""

    def __init__(self, settings, nim) -> None:
        self._settings = settings
        self._nim = nim
        self._stores: Dict[int, GuildRagStore] = {}

    @property
    def embed_model(self) -> str:
        return getattr(self._nim, "embed_model_name", None) or self._settings.embed_model

    @property
    def min_score(self) -> float:
        if self._settings.rag_min_score is not None:
            return float(self._settings.rag_min_score)
        return float(getattr(self._nim, "default_min_score", 0.2))

    def store_for(self, guild_id: int) -> GuildRagStore:
        store = self._stores.get(guild_id)
        if store is None:
            store = GuildRagStore(guild_id, self._settings.rag_dir)
            self._stores[guild_id] = store
        return store

    def forget(self, guild_id: int) -> None:
        self.store_for(guild_id).clear()
        self._stores.pop(guild_id, None)

    async def ingest_documents(
        self, guild_id: int, documents: Sequence[Tuple[str, str]]
    ) -> IngestReport:
        """Embed and store ``(source_label, text)`` documents for a guild."""
        report = IngestReport(embed_model=self.embed_model)
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
        self.store_for(guild_id).replace(matrix, chunks, embed_model=self.embed_model)
        report.chunks = len(chunks)
        report.dim = int(matrix.shape[1])
        return report

    async def retrieve(self, guild_id: int, question: str) -> List[Tuple[Chunk, float]]:
        """Top-k chunks at or above the relevance floor (may be empty)."""
        store = self.store_for(guild_id)
        if store.load_error:
            raise RagIndexError(store.load_error)
        if store.size == 0:
            return []
        store.check_model(self.embed_model)
        query_vecs = await self._nim.embed([question], input_type="query")
        query_vec = np.array(query_vecs[0], dtype=np.float32)
        return store.search(query_vec, self._settings.rag_top_k, self.min_score)

    async def answer(self, guild_id: int, question: str) -> RagAnswer:
        """Answer from the guild's docs, citing only what the answer uses."""
        store = self.store_for(guild_id)
        if store.size == 0 and not store.load_error:
            return RagAnswer(
                text=NO_INDEX_TEXT, sources=[], hits=[], found=False, indexed=False
            )
        hits = await self.retrieve(guild_id, question)
        if not hits:
            # Nothing is relevant enough: skip the chat call entirely.
            return RagAnswer(text=NOT_FOUND_TEXT, sources=[], hits=[], found=False)

        context_blocks = []
        for idx, (chunk, _score) in enumerate(hits, start=1):
            context_blocks.append(f"[{idx}] Source: {chunk.source}\n{chunk.text}")
        context = "\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        text = await self._nim.chat(messages, temperature=0.2)
        cited = parse_citations(text, len(hits))
        sources = [(n, hits[n - 1][0]) for n in cited]
        return RagAnswer(text=text, sources=sources, hits=hits, found=bool(cited))

    def status(self, guild_id: int) -> Dict[str, object]:
        store = self.store_for(guild_id)
        meta = store.meta
        return {
            "chunks": store.size,
            "sources": len(store.sources),
            "embed_model": meta.embed_model if meta else None,
            "dim": meta.dim if meta else 0,
            "built_at": meta.built_at if meta else None,
            "error": store.load_error,
            "current_embed_model": self.embed_model,
        }
