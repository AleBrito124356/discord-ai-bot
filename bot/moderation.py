"""Assist-only moderation.

Design stance (important): this module NEVER bans, kicks, mutes, deletes, or edits
anything. It only *suggests*. When a message looks like it might break the rules,
the bot posts a short, sourced note to a mod-only channel so a human can decide.
Moderation is opt-in per guild and off by default.

Pipeline
--------
1. Cheap local heuristics score every message (no API cost). Messages that score
   below the floor (``max(0.25, threshold / 2)``) are dropped immediately.
2. Messages that clear the prefilter are sent to the chat model for a structured
   second opinion (returns JSON: flag, category, severity, rationale). The reply
   is parsed defensively: string booleans, 0-100 severities and unknown
   categories are normalised instead of trusted.
3. If the model agrees and severity clears the threshold, an advisory embed is
   posted to the mod channel. That is the end of the bot's involvement.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Pattern, Sequence

from .nim_client import NimError
from .prompts import MODERATION_SYSTEM_PROMPT

log = logging.getLogger("moderation")


def _compile(patterns: Sequence[str]) -> Pattern[str]:
    """Join word-level patterns into one case-insensitive, word-bounded regex."""
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


# Structural signals only. This is intentionally a small, generic list — the LLM
# pass does the nuanced judgement. Every pattern is matched on word boundaries,
# so "paradox", "slurped" or "fire retardant" never trigger. Extend these lists
# with your community's rules (they are regular expressions).
SLUR_PATTERNS: List[str] = [r"kys", r"kill\s+yourself", r"retard(?:ed|s)?"]
THREAT_PATTERNS: List[str] = [
    r"i\s*(?:will|'ll|’ll|am\s+going\s+to|'m\s+gonna|’m\s+gonna)\s+(?:kill|hurt)",
    r"dox(?:x)?(?:ed|ing)?",
    r"swat(?:ting)?\s+you",
    r"hurt\s+you",
]
SCAM_PATTERNS: List[str] = [
    r"free\s+nitro",
    r"steam\s+gift",
    r"click\s+(?:this|the|my)\s+link",
    r"claim\s+your\s+(?:prize|reward|gift|nitro)",
]

_SLUR_RE = _compile(SLUR_PATTERNS)
_THREAT_RE = _compile(THREAT_PATTERNS)
_SCAM_RE = _compile(SCAM_PATTERNS)
_INVITE_RE = re.compile(r"(discord\.gg/|discord\.com/invite/)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://", re.IGNORECASE)

CATEGORIES = (
    "harassment",
    "hate",
    "threat",
    "nsfw",
    "scam",
    "spam",
    "doxxing",
    "other",
    "none",
)

SYSTEM_PROMPT = MODERATION_SYSTEM_PROMPT


@dataclass
class HeuristicResult:
    score: float
    reasons: List[str]


@dataclass
class Verdict:
    flag: bool
    category: str
    severity: float
    rationale: str


@dataclass
class ReviewResult:
    """Everything the pipeline decided about one message (for logs, CLI, tests)."""

    heuristic: HeuristicResult
    floor: float
    threshold: float
    llm_called: bool = False
    verdict: Optional[Verdict] = None
    notes: List[str] = field(default_factory=list)

    @property
    def report(self) -> bool:
        """True when an advisory should be posted to the mod channel."""
        return (
            self.verdict is not None
            and self.verdict.flag
            and self.verdict.severity >= self.threshold
        )


def prefilter_floor(threshold: float) -> float:
    """Heuristic score a message needs before it costs an LLM call."""
    return max(0.25, threshold * 0.5)


def prefilter(message_content: str, mention_count: int) -> HeuristicResult:
    """Cheap local scoring. Returns a 0..1 score and human-readable reasons."""
    text = message_content or ""
    score = 0.0
    reasons: List[str] = []

    def bump(amount: float, reason: str) -> None:
        nonlocal score
        score += amount
        reasons.append(reason)

    if _SLUR_RE.search(text):
        bump(0.6, "possible slur or self-harm phrase")
    if _THREAT_RE.search(text):
        bump(0.7, "possible threat language")
    if _SCAM_RE.search(text):
        bump(0.5, "possible scam or phishing bait")
    if _INVITE_RE.search(text):
        bump(0.3, "contains a Discord invite link")

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.7:
            bump(0.2, "shouting (mostly capitals)")

    if mention_count >= 5:
        bump(0.4, f"mass mention ({mention_count} pings)")

    urls = len(_URL_RE.findall(text))
    if urls >= 3:
        bump(0.3, f"link spam ({urls} links)")

    if re.search(r"(.)\1{9,}", text):
        bump(0.2, "spammy repeated characters")

    return HeuristicResult(score=min(score, 1.0), reasons=reasons)


# ----------------------------------------------------------- verdict parsing
_TRUE_STRINGS = {"true", "yes", "y", "1", "flag", "flagged"}
_FALSE_STRINGS = {"false", "no", "n", "0", "none", "null", ""}


def _coerce_flag(value: Any) -> Optional[bool]:
    """Strict boolean: ``bool("false")`` is True in Python, so never use it."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    if value is None:
        return False
    return None


def _coerce_severity(value: Any) -> Optional[float]:
    """Severity as 0..1.

    Accepts 0.8, "0.8", "80%", 8 (a 0-10 scale) and 85 (a 0-100 scale); anything
    else numeric is clamped into 0..1. Non-numeric values make the verdict void.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        raw = value.strip()
        percent = raw.endswith("%")
        try:
            number = float(raw.rstrip("%").strip())
        except ValueError:
            return None
        if percent:
            number /= 100.0
    elif isinstance(value, (int, float)):
        number = float(value)
    elif value is None:
        return 0.0
    else:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if 1.0 < number <= 10.0:
        number /= 10.0  # the model answered on a 0-10 scale
    elif 10.0 < number <= 100.0:
        number /= 100.0  # the model answered on a 0-100 scale
    return min(1.0, max(0.0, number))


def _coerce_category(value: Any, flag: bool) -> str:
    category = str(value or "").strip().lower()
    if category not in CATEGORIES:
        category = "other"
    if flag and category == "none":
        category = "other"
    if not flag:
        category = "none"
    return category


def parse_verdict(raw: str) -> Optional[Verdict]:
    """Turn a model reply into a validated Verdict, or None when unusable."""
    data = _extract_json(raw or "")
    if data is None:
        return None
    flag = _coerce_flag(data.get("flag", False))
    severity = _coerce_severity(data.get("severity", 0.0))
    if flag is None or severity is None:
        return None
    return Verdict(
        flag=flag,
        category=_coerce_category(data.get("category"), flag),
        severity=severity,
        rationale=str(data.get("rationale", "") or "").strip()[:300],
    )


def _extract_json(raw: str) -> Optional[dict]:
    """Return the first JSON *object* found anywhere in a model response.

    Scans with ``json.JSONDecoder.raw_decode`` from every ``{`` so prose around
    the object, code fences or a second brace group later in the reply do not
    break parsing (a greedy ``\\{.*\\}`` regex fails on all three).
    """
    text = (raw or "").strip()
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(value, dict):
            return value
        index = text.find("{", index + 1)
    return None


def build_user_prompt(content: str, heuristic_reasons: Sequence[str]) -> str:
    hint = ", ".join(heuristic_reasons) or "none"
    return (
        f"Automated prefilter noticed: {hint}.\n\n"
        f"Message:\n\"\"\"\n{content[:1500]}\n\"\"\""
    )


class ModerationService:
    """Runs the prefilter + optional LLM check and builds advisories."""

    def __init__(self, settings, nim) -> None:
        self._settings = settings
        self._nim = nim

    async def review_text(
        self, content: str, mention_count: int, threshold: float
    ) -> ReviewResult:
        """Discord-free review of one message. Never takes action."""
        heur = prefilter(content, mention_count)
        result = ReviewResult(
            heuristic=heur, floor=prefilter_floor(threshold), threshold=threshold
        )
        if heur.score < result.floor:
            result.notes.append("below prefilter floor; no API call")
            return result
        result.llm_called = True
        result.verdict = await self._llm_check(content, heur.reasons)
        if result.verdict is None:
            result.notes.append("model reply was unusable or the call failed")
        return result

    async def review_message(self, message, threshold: float) -> Optional[Verdict]:
        """Return a Verdict worth reporting for a discord.Message, or None."""
        result = await self.review_text(
            message.content, len(message.mentions), threshold
        )
        return result.verdict if result.report else None

    async def _llm_check(
        self, content: str, heuristic_reasons: List[str]
    ) -> Optional[Verdict]:
        try:
            raw = await self._nim.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(content, heuristic_reasons)},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except NimError as exc:
            log.warning("Moderation LLM check failed: %s", exc)
            return None
        verdict = parse_verdict(raw)
        if verdict is None:
            log.warning("Unparseable moderation verdict: %r", (raw or "")[:200])
        return verdict

    def build_advisory_embed(self, message, verdict: Verdict):
        """Build the advisory embed. Suggestion only — no action buttons."""
        import discord

        colour = discord.Color.orange()
        if verdict.severity >= 0.85:
            colour = discord.Color.red()
        embed = discord.Embed(
            title="Moderation advisory (suggestion only)",
            description=verdict.rationale or "Flagged for human review.",
            colour=colour,
        )
        author = message.author
        embed.add_field(name="Author", value=f"{author.mention} ({author})", inline=True)
        embed.add_field(name="Category", value=verdict.category, inline=True)
        embed.add_field(name="Severity", value=f"{verdict.severity:.2f}", inline=True)
        channel_ref = getattr(message.channel, "mention", "#channel")
        embed.add_field(name="Channel", value=channel_ref, inline=True)
        if message.jump_url:
            embed.add_field(
                name="Jump", value=f"[Go to message]({message.jump_url})", inline=True
            )
        excerpt = (message.content or "").strip()
        if excerpt:
            embed.add_field(
                name="Message",
                value=excerpt if len(excerpt) <= 1000 else excerpt[:997] + "...",
                inline=False,
            )
        embed.set_footer(
            text="The bot took no action. A human moderator decides what to do."
        )
        return embed
