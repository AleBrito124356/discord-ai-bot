"""Deterministic offline backend with the same interface as ``NimClient``.

Selected with ``BOT_OFFLINE=1`` or ``NIM_BASE_URL=offline`` (the terminal
simulator ``python -m bot.cli`` uses it by default). No network, no keys, no
model weights — and identical input always gives identical output, which makes
it useful for demos, CI and tests.

What it really does
-------------------
* **Embeddings** — feature hashing of stemmed content words plus word bigrams
  into a signed 512-d vector, L2-normalised. Retrieval therefore ranks by real
  lexical overlap (not semantics).
* **Chat** recognises the prompt shapes the bot sends and answers them
  *extractively* (it never invents text):

  - docs questions → the best-matching sentences from the numbered excerpts,
    cited as ``[n]``; "could not find" when nothing overlaps;
  - summaries → the most informative lines of the transcript as bullets, in
    order, labelled Decision / Question / Action where the wording says so;
  - moderation → a rule-based JSON verdict built on the same prefilter
    patterns the bot uses;
  - anything else (``/ask``) → an explicitly labelled offline reply.
* **Vision** → facts Pillow can measure: format, size, frames, orientation,
  brightness and dominant colours. It can not see what is *in* the picture and
  says so.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .moderation import _SCAM_RE, _SLUR_RE, _THREAT_RE, prefilter
from .prompts import MODERATION_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT

OFFLINE_EMBED_MODEL = "offline/hashed-bow-512"
OFFLINE_CHAT_MODEL = "offline/extractive"
OFFLINE_TAG = "[offline mode]"

_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at be
    because been before being below between both but by can cannot could did do
    does doing done down during each few for from further get got had has have
    having he her here hers herself him himself his how i if in into is isn it
    its itself just let me more most my myself no nor not now of off on once only
    or other ought our ours ourselves out over own same shall she should so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up us very was we were what when where which
    while who whom why will with would you your yours yourself yourselves please
    tell know use used using
    """.split()
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(word: str) -> str:
    """Tiny suffix stripper: good enough to match plural/verb forms."""
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    return word


def content_words(text: str) -> List[str]:
    words = [w.split("'")[0] for w in _WORD_RE.findall((text or "").lower())]
    return [_stem(w) for w in words if w and w not in _STOPWORDS]


def features(text: str) -> Dict[str, float]:
    """Unigrams (weight 1) and adjacent-word bigrams (weight 0.5), as presence."""
    words = content_words(text)
    feats: Dict[str, float] = {w: 1.0 for w in words}
    for a, b in zip(words, words[1:]):
        feats[f"{a} {b}"] = 0.5
    return feats


def _bucket(feature: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


def hashed_embedding(text: str, dim: int = 512) -> List[float]:
    vec = [0.0] * dim
    for feat, weight in features(text).items():
        index, sign = _bucket(feat, dim)
        vec[index] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


# --------------------------------------------------------------- sentences
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*\"'(\[])")


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#{1,6}\s*", "", line)          # markdown headings
    line = re.sub(r"^(?:[-*•>]|\d+[.)])\s+", "", line)  # list markers / quotes
    if line.startswith("|") and line.endswith("|"):  # markdown table rows
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            return ""
        line = " — ".join(c for c in cells if c)
    return line.strip()


_BLOCK_LINE = re.compile(r"^\s*(?:#{1,6}\s|[-*•>]\s|\d+[.)]\s|\|)")


def split_sentences(text: str) -> List[str]:
    """Sentences of a markdown-ish text.

    Soft-wrapped prose lines are joined into paragraphs first, so a sentence
    that spans two lines stays whole. Headings, list items, table rows and
    lines inside code fences are kept as units of their own.
    """
    sentences: List[str] = []
    paragraph: List[str] = []
    lines = (text or "").split("\n")
    fences = [l.strip() for l in lines if l.strip().startswith("```")]
    # A chunk can start in the middle of a code block (chunk overlap). Closing
    # fences are bare, openers usually carry a language tag: use that to guess.
    in_code = bool(fences) and fences[0] == "```" and (
        len(fences) % 2 == 1 or (len(fences) > 1 and fences[1] != "```")
    )

    def flush() -> None:
        if paragraph:
            joined = " ".join(paragraph)
            sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(joined) if s.strip())
            paragraph.clear()

    for raw in lines:
        if raw.strip().startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            if raw.strip():
                sentences.append(raw.strip())
            continue
        if not raw.strip():
            flush()
            continue
        if _BLOCK_LINE.match(raw):
            flush()
            line = _clean_line(raw)
            if line:
                sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
            continue
        paragraph.append(_clean_line(raw))
    flush()
    return sentences


def _overlap(
    query: Dict[str, float], sentence: str, idf: Optional[Dict[str, float]] = None
) -> Tuple[float, int]:
    """(score, number of shared features) of a sentence against a query.

    Shared features are weighted by ``idf`` (rarer words in the retrieved
    excerpts count more) and bigrams count double.
    """
    feats = features(sentence)
    shared = [f for f in query if f in feats]
    if not shared:
        return 0.0, 0
    weight = sum(
        min(query[f], feats[f]) * (2.0 if " " in f else 1.0) * (idf.get(f, 1.0) if idf else 1.0)
        for f in shared
    )
    # Prefer short, dense sentences over long ones that merely mention a word,
    # but do not let 1-2 word fragments (table headers) win on density alone.
    return weight / math.sqrt(max(len(feats), 4)), len(shared)


class OfflineNimClient:
    """Drop-in stand-in for :class:`bot.nim_client.NimClient`."""

    embed_model_name = OFFLINE_EMBED_MODEL
    # Hashed bag-of-words cosines are small: a relevant chunk usually scores
    # 0.08-0.4 and an unrelated one close to 0.
    default_min_score = 0.05
    offline = True

    def __init__(self, settings=None, *, dim: int = 512) -> None:
        self._settings = settings
        self.dim = dim

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------ embeddings
    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str = "passage",
        model: Optional[str] = None,
    ) -> List[List[float]]:
        return [hashed_embedding(t, self.dim) for t in texts]

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: Sequence[Dict[str, object]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        system = next(
            (str(m.get("content", "")) for m in messages if m.get("role") == "system"), ""
        )
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if system == RAG_SYSTEM_PROMPT:
            return self._answer_from_context(last_user)
        if system == SUMMARIZE_SYSTEM_PROMPT:
            return self._summarize(last_user)
        if system == MODERATION_SYSTEM_PROMPT:
            return self._moderate(last_user)
        return self._small_talk(messages, system, last_user)

    # docs ------------------------------------------------------------------
    @staticmethod
    def _parse_context(user: str) -> Tuple[List[Tuple[int, str, str]], str]:
        question = ""
        match = re.search(r"\n\nQuestion:\s*(.*)\Z", user, re.DOTALL)
        body = user
        if match:
            question = match.group(1).strip()
            body = user[: match.start()]
        body = re.sub(r"\AContext:\n", "", body)
        blocks: List[Tuple[int, str, str]] = []
        parts = re.split(r"(?m)^\[(\d+)\] Source: (.*)$", body)
        # parts = [prefix, n, source, text, n, source, text, ...]
        for i in range(1, len(parts) - 2, 3):
            blocks.append((int(parts[i]), parts[i + 1].strip(), parts[i + 2].strip()))
        return blocks, question

    def _answer_from_context(self, user: str) -> str:
        blocks, question = self._parse_context(user)
        query = features(question)
        candidates: List[Tuple[int, int, str]] = []
        seen = set()
        for number, _source, text in blocks:
            for position, sentence in enumerate(split_sentences(text)):
                if sentence in seen:  # chunk overlap repeats sentences
                    continue
                seen.add(sentence)
                candidates.append((number, position, sentence))
        doc_freq: Counter = Counter()
        for _n, _p, sentence in candidates:
            doc_freq.update(features(sentence).keys())
        total = max(1, len(candidates))
        idf = {f: math.log(1.0 + total / doc_freq[f]) for f in query if doc_freq[f]}
        scored: List[Tuple[float, int, int, str, int]] = []
        for number, position, sentence in candidates:
            score, shared = _overlap(query, sentence, idf)
            if score > 0:
                scored.append((score, number, position, sentence, shared))
        if not scored:
            return (
                f"{OFFLINE_TAG} I could not find that in the server's documents."
            )
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        best = scored[0]
        picked = [best]
        for candidate in scored[1:]:
            if len(picked) == 2:
                break
            # A second sentence must be nearly as good, share 2+ query terms
            # and not repeat the first (chunk overlap can cut a copy of it).
            if (
                candidate[0] >= 0.6 * best[0]
                and candidate[4] >= 2
                and candidate[3] not in best[3]
                and best[3] not in candidate[3]
            ):
                picked.append(candidate)
        sentences = " ".join(f"{s[3].rstrip()} [{s[1]}]" for s in picked)
        return f"{OFFLINE_TAG} Best-matching passage(s) from your docs: {sentences}"

    # summaries ---------------------------------------------------------------
    _DECISION = re.compile(r"\b(decid\w*|agreed?|approved?|we(?:'ll| will) go with|final(?:ly)?|ship\w*)\b", re.I)
    _ACTION = re.compile(r"\b(todo|to-do|action|will|i'll|assign\w*|owns?|deadline|by (?:mon|tue|wed|thu|fri|sat|sun)\w*|tomorrow|need to|should)\b", re.I)

    def _summarize(self, transcript: str) -> str:
        entries: List[Tuple[str, str]] = []
        for line in transcript.splitlines():
            line = line.strip()
            if not line:
                continue
            name, sep, text = line.partition(": ")
            entries.append((name, text) if sep else ("", line))
        if not entries:
            return f"{OFFLINE_TAG} Nothing to summarize."
        doc_freq = Counter(w for _, text in entries for w in set(content_words(text)))
        scored = []
        for index, (name, text) in enumerate(entries):
            words = set(content_words(text))
            centrality = sum(doc_freq[w] - 1 for w in words) / math.sqrt(len(words) + 1)
            bonus = 0.0
            if self._DECISION.search(text):
                bonus += 3.0
            if self._ACTION.search(text):
                bonus += 2.0
            if text.rstrip().endswith("?"):
                bonus += 1.5
            if len(words) < 3:
                bonus -= 2.0  # "ok", "lol", "+1"
            scored.append((centrality + bonus, index))
        keep = min(7, max(3, len(entries) // 3 + 1), len(entries))
        chosen = sorted(i for _, i in sorted(scored, key=lambda s: (-s[0], s[1]))[:keep])
        bullets = []
        for index in chosen:
            name, text = entries[index]
            if self._DECISION.search(text):
                label = "Decision: "
            elif text.rstrip().endswith("?"):
                label = "Question: "
            elif self._ACTION.search(text):
                label = "Action: "
            else:
                label = ""
            short = text if len(text) <= 200 else text[:197] + "..."
            who = f"**{name}**: " if name else ""
            bullets.append(f"• {label}{who}{short}")
        return (
            f"{OFFLINE_TAG} Extractive summary ({len(chosen)} of {len(entries)} lines):\n"
            + "\n".join(bullets)
        )

    # moderation -------------------------------------------------------------
    def _moderate(self, user: str) -> str:
        match = re.search(r'"""\n(.*)\n"""', user, re.DOTALL)
        text = match.group(1) if match else user
        heur = prefilter(text, 0)
        mentions = re.search(r"mass mention \((\d+) pings\)", user)
        verdict = {"flag": False, "category": "none", "severity": 0.0,
                   "rationale": "No rule-breaking pattern found (offline rules)."}
        if _THREAT_RE.search(text):
            verdict = {"flag": True, "category": "threat", "severity": 0.9,
                       "rationale": "Contains threat or doxxing language (offline rules)."}
        elif _SLUR_RE.search(text):
            verdict = {"flag": True, "category": "harassment", "severity": 0.8,
                       "rationale": "Contains a slur or self-harm taunt (offline rules)."}
        elif _SCAM_RE.search(text):
            verdict = {"flag": True, "category": "scam", "severity": 0.85,
                       "rationale": "Looks like scam or phishing bait (offline rules)."}
        elif mentions or any("link spam" in r or "repeated" in r for r in heur.reasons):
            verdict = {"flag": True, "category": "spam", "severity": 0.65,
                       "rationale": "Mass mentions, link spam or repeated characters (offline rules)."}
        elif heur.reasons:
            verdict = {"flag": False, "category": "none", "severity": 0.3,
                       "rationale": "Only weak signals: " + ", ".join(heur.reasons) + "."}
        return json.dumps(verdict)

    # /ask -------------------------------------------------------------------
    @staticmethod
    def _small_talk(messages: Sequence[Dict[str, object]], system: str, user: str) -> str:
        from .personas import PERSONAS  # local import keeps module import light

        persona = next((p["name"] for p in PERSONAS.values() if p["system"] == system), None)
        earlier = max(0, sum(1 for m in messages if m.get("role") in ("user", "assistant")) - 1)
        prompt = " ".join(user.split())
        if len(prompt) > 300:
            prompt = prompt[:297] + "..."
        lines = [
            f"{OFFLINE_TAG} No language model is connected, so this is not a real answer.",
            f"You said: \"{prompt}\"",
            f"Context I would have sent: persona {persona or 'custom'}, "
            f"{earlier} earlier message(s) from this channel's memory.",
            "Set NVIDIA_API_KEY (and unset BOT_OFFLINE) to get model answers; "
            "/docs, /summarize, moderation and /image still do real offline work.",
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------------- vision
    async def vision(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        from .imaging import ImageError, describe_pixels

        try:
            facts = describe_pixels(image_bytes)
        except ImageError as exc:
            return f"{OFFLINE_TAG} {exc}"
        colours = ", ".join(f"{name} {int(share * 100)}%" for name, share in facts["dominant_colours"])
        frames = f", {facts['frames']} frames" if facts["frames"] > 1 else ""
        return (
            f"{OFFLINE_TAG} I can not see image content without a vision model, "
            f"but I measured it: a {facts['width']}x{facts['height']} {facts['orientation']} "
            f"{facts['format']} ({mime_type}{frames}), brightness {int(facts['brightness'] * 100)}%, "
            f"dominant colours: {colours}. Your question was: \"{prompt}\""
        )

