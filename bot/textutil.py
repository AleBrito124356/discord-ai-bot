"""Pure text helpers shared by the Discord adapters, the services and the CLI.

Nothing here touches Discord, the network or the database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

_FENCE = "```"
_CLOSE = "\n```"


def _fence_state(text: str, opener: Optional[str] = None) -> Optional[str]:
    """Return the opening fence line still open at the end of ``text``, if any.

    ``opener`` is the fence already open when ``text`` starts.
    """
    current = opener
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(_FENCE):
            current = None if current is not None else stripped
    return current


def split_message(text: str, limit: int = 1900) -> List[str]:
    """Split text into <= ``limit`` chunks for Discord, preferring newlines.

    Code fences survive the split: when a cut lands inside a fenced block, the
    chunk is closed with a fence and the next chunk re-opens it with the same
    language tag, so every part renders as valid Discord markdown.
    """
    text = text or "(empty response)"
    if len(text) <= limit:
        return [text]
    fence_aware = limit >= 40
    reserve = len(_CLOSE) if fence_aware else 0
    chunks: List[str] = []
    remaining = text
    reopen = ""  # fence line (plus newline) to prepend to the next chunk
    while remaining:
        body = reopen + remaining
        if len(body) <= limit:
            chunks.append(body)
            break
        budget = limit - reserve
        window = body[:budget]
        cut = window.rfind("\n")
        if cut < budget // 2 or cut <= len(reopen):
            cut = budget
        piece = body[:cut].rstrip()
        open_fence = _fence_state(piece) if fence_aware else None
        if open_fence is not None and piece.strip() == open_fence and cut != budget:
            # A part holding only an opening fence line is useless: hard cut.
            cut = budget
            piece = body[:cut].rstrip()
            open_fence = _fence_state(piece)
        rest = body[cut:].lstrip("\n")
        if open_fence is not None and len(open_fence) + 1 < budget // 2:
            piece += _CLOSE
            reopen = open_fence + "\n"
        else:
            reopen = ""
        if piece.strip():
            chunks.append(piece)
        remaining = rest
    return chunks


@dataclass
class Transcript:
    """A transcript trimmed to a character budget.

    ``keep`` says which end survived: ``"newest"`` (the default, used by
    /summarize) or ``"oldest"`` (used when summarizing *from* a message).
    """

    text: str
    included: int
    total: int
    keep: str = "newest"

    @property
    def truncated(self) -> bool:
        return self.included < self.total

    def header(self, noun: str = "messages", anchor: Optional[str] = None) -> str:
        if anchor is not None:
            if self.truncated:
                return (
                    f"**Summary of the first {self.included} of {self.total} {noun} "
                    f"from {anchor}** (later ones did not fit)"
                )
            return f"**Summary of {self.total} {noun} from {anchor}**"
        if self.truncated:
            which, dropped = ("last", "older") if self.keep == "newest" else ("first", "later")
            return (
                f"**Summary of the {which} {self.included} of {self.total} {noun}** "
                f"({dropped} ones did not fit)"
            )
        return f"**Summary of the last {self.total} {noun}**"


def build_transcript(lines: Sequence[str], budget: int, keep: str = "newest") -> Transcript:
    """Join ``lines`` (given oldest-first) within ``budget`` characters.

    With ``keep="newest"`` the NEWEST lines are kept: summaries are about what
    is happening now, so when the transcript is too long the oldest lines are
    dropped, never the most recent ones. ``keep="oldest"`` keeps the start
    instead. A single line longer than the budget is cut to fit, so the line
    that must survive is never lost entirely.
    """
    if keep not in ("newest", "oldest"):
        raise ValueError("keep must be 'newest' or 'oldest'")
    ordered = list(reversed(lines)) if keep == "newest" else list(lines)
    kept: List[str] = []
    used = 0
    for line in ordered:
        cost = len(line) + (1 if kept else 0)
        if used + cost > budget:
            if not kept and budget > 0:
                kept.append(line[:budget])
            break
        kept.append(line)
        used += cost
    if keep == "newest":
        kept.reverse()
    return Transcript(text="\n".join(kept), included=len(kept), total=len(lines), keep=keep)
