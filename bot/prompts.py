"""System prompts shared by the services and the offline backend.

Keeping them in one module means the offline backend (``bot.offline``) can
recognise which task a chat request is for without importing the services.
"""
from __future__ import annotations

RAG_SYSTEM_PROMPT = (
    "You answer questions strictly from the provided context excerpts. "
    "Cite the sources you use with bracketed numbers like [1] that match "
    "the excerpt numbers. If the answer is not in the context, say you "
    "could not find it in the server's documents. Do not invent citations."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following Discord conversation into 3-7 concise "
    "bullet points capturing decisions, questions and action items. "
    "Use Discord markdown bullets. Do not invent details."
)

MODERATION_SYSTEM_PROMPT = (
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

EXPLAIN_MESSAGE_TEMPLATE = (
    "A server member asked you to look at this Discord message written by "
    "{author}:\n\"\"\"\n{content}\n\"\"\"\n"
    "If it is a question, answer it. Otherwise explain what it means or what "
    "it is about, briefly."
)

IMAGE_IN_MESSAGE_TEMPLATE = (
    "This image was posted in a Discord message by {author}{caption}. "
    "Describe it and explain anything in it that needs explaining."
)
