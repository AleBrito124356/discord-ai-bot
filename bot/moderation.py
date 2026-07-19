"""Assist-only moderation.

Design stance (important): this module NEVER bans, kicks, mutes, deletes, or edits
anything. It only *suggests*. When a message looks like it might break the rules,
the bot posts a short, sourced note to a mod-only channel so a human can decide.
Moderation is opt-in per guild and off by default.

Pipeline
--------
1. Cheap local heuristics score every message (no API cost). Messages that score
   below the guild threshold are dropped immediately.
2. Messages that clear the prefilter are sent to the chat model for a structured
   second opinion (returns JSON: flag, category, severity, rationale).
3. If the model agrees and severity clears the threshold, an advisory embed is
   posted to the mod channel. That is the end of the bot's involvement.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import discord

from .config import Settings
from .nim_client import NimClient, NimError

log = logging.getLogger("moderation")

# Structural signals only. This is intentionally a small, generic list — the LLM
# pass does the nuanced judgement. Extend for your community's rules.
_SLUR_HINTS = ("slur", "kys", "kill yourself", "retard")
_THREAT_HINTS = ("i will kill", "i'll kill", "dox", "swat you", "hurt you")
_SCAM_HINTS = ("free nitro", "steam gift", "click this link", "claim your prize")
_INVITE_RE = re.compile(r"(discord\.gg/|discord\.com/invite/)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://", re.IGNORECASE)


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


def prefilter(message_content: str, mention_count: int) -> HeuristicResult:
    """Cheap local scoring. Returns a 0..1 score and human-readable reasons."""
    text = message_content or ""
    lowered = text.lower()
    score = 0.0
    reasons: List[str] = []

    def bump(amount: float, reason: str) -> None:
        nonlocal score
        score += amount
        reasons.append(reason)

    if any(h in lowered for h in _SLUR_HINTS):
        bump(0.6, "possible slur or self-harm phrase")
    if any(h in lowered for h in _THREAT_HINTS):
        bump(0.7, "possible threat language")
    if any(h in lowered for h in _SCAM_HINTS):
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


class ModerationService:
    """Runs the prefilter + optional LLM check and posts advisories."""

    def __init__(self, settings: Settings, nim: NimClient) -> None:
        self._settings = settings
        self._nim = nim

    async def review_message(
        self, message: discord.Message, threshold: float
    ) -> Optional[Verdict]:
        """Return a Verdict worth reporting, or None. Never takes action."""
        heur = prefilter(message.content, len(message.mentions))
        if heur.score < max(0.25, threshold * 0.5):
            return None  # clearly fine; skip the API call entirely

        verdict = await self._llm_check(message.content, heur.reasons)
        if verdict is None:
            return None
        if not verdict.flag or verdict.severity < threshold:
            return None
        return verdict

    async def _llm_check(
        self, content: str, heuristic_reasons: List[str]
    ) -> Optional[Verdict]:
        system = (
            "You are a moderation assistant. You do not take any action; you only "
            "advise human moderators. Judge whether a single chat message likely "
            "breaks common community rules (harassment, hate, threats, sexual "
            "content involving minors, scams/phishing, spam, or doxxing). "
            "Respond with ONLY a compact JSON object and nothing else, of the form: "
            '{"flag": true|false, "category": "harassment|hate|threat|nsfw|scam|'
            'spam|doxxing|other|none", "severity": 0.0-1.0, "rationale": "one short '
            'sentence"}. Be conservative: normal disagreement, profanity used '
            "casually, or edgy jokes are usually not violations."
        )
        hint = ", ".join(heuristic_reasons) or "none"
        user = (
            f"Automated prefilter noticed: {hint}.\n\n"
            f"Message:\n\"\"\"\n{content[:1500]}\n\"\"\""
        )
        try:
            raw = await self._nim.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except NimError as exc:
            log.warning("Moderation LLM check failed: %s", exc)
            return None

        data = _extract_json(raw)
        if data is None:
            return None
        try:
            return Verdict(
                flag=bool(data.get("flag", False)),
                category=str(data.get("category", "other")),
                severity=float(data.get("severity", 0.0)),
                rationale=str(data.get("rationale", "")).strip()[:300],
            )
        except (TypeError, ValueError):
            return None

    def build_advisory_embed(
        self, message: discord.Message, verdict: Verdict
    ) -> discord.Embed:
        """Build the advisory embed. Suggestion only — no action buttons."""
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
        embed.add_field(
            name="Severity", value=f"{verdict.severity:.2f}", inline=True
        )
        channel_ref = getattr(message.channel, "mention", "#channel")
        embed.add_field(name="Channel", value=channel_ref, inline=True)
        if message.jump_url:
            embed.add_field(name="Jump", value=f"[Go to message]({message.jump_url})", inline=True)
        excerpt = (message.content or "").strip()
        if excerpt:
            embed.add_field(
                name="Message",
                value=excerpt[:1000] if len(excerpt) <= 1000 else excerpt[:997] + "...",
                inline=False,
            )
        embed.set_footer(
            text="The bot took no action. A human moderator decides what to do."
        )
        return embed


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
