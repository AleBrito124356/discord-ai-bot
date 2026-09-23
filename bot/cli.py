"""Terminal simulator: run every bot feature without Discord (and without keys).

    python -m bot.cli docs ingest docs/
    python -m bot.cli docs ask "what permission integer does the invite URL use?"
    python -m bot.cli moderate "free nitro, click this link discord.gg/x"
    python -m bot.cli summarize chat.txt
    python -m bot.cli image photo.webp --question "what is this?"
    python -m bot.cli ask "hello"      # plus: docs status, stats, forget

It uses the same services, SQLite schema and RAG store as the bot, against a
separate data directory (``--data-dir``, default ``<DATA_DIR>/sim``). By default
it runs on the deterministic offline backend; ``--live`` uses NVIDIA NIM with
the ``NVIDIA_API_KEY`` from the environment or the repository ``.env``.
Installed with pip, the same program is available as ``discord-ai-bot-sim``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import BASE_DIR, SIGNUP_HINT, load_env_file, load_settings
from .services import DOC_EXTS, BotCore, Reply, document_text, stats_lines

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _default_data_dir() -> Path:
    base = Path(os.getenv("DATA_DIR") or (BASE_DIR / "data")).expanduser()
    return base / "sim"


def _common_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Options accepted before OR after the subcommand."""
    default = (lambda value: argparse.SUPPRESS) if suppress else (lambda value: value)
    parser.add_argument(
        "--data-dir", type=Path, default=default(None),
        help="where the simulator keeps its SQLite DB and docs index "
             "(default: <DATA_DIR>/sim)",
    )
    parser.add_argument(
        "--live", action="store_true", default=default(False),
        help="use NVIDIA NIM (needs NVIDIA_API_KEY) instead of the offline backend",
    )
    parser.add_argument("--guild", type=int, default=default(1), help="simulated server id")
    parser.add_argument("--channel", type=int, default=default(1), help="simulated channel id")
    parser.add_argument("--user", type=int, default=default(1), help="simulated user id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discord-ai-bot-sim",
        description="Run the Discord AI bot's features from a terminal. "
        "Offline and deterministic unless --live is given.",
    )
    _common_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        _common_options(p, suppress=True)
        return p

    p = add("ask", "chat with per-channel memory, like /ask")
    p.add_argument("prompt", nargs="+")

    p = add("summarize", "summarize a transcript file of 'Name: message' lines, like /summarize")
    p.add_argument("file", help="text file, one message per line ('-' reads stdin)")
    p.add_argument("--count", type=int, default=200,
                   help="use only the last N lines (default 200, the /summarize maximum)")
    p.add_argument("--from-line", type=int, default=None, metavar="N",
                   help="summarize from line N onward, like the 'Summarize from here' menu")

    p = add("docs", "ingest documents and ask questions about them, like /docs")
    docs_sub = p.add_subparsers(dest="docs_command", metavar="ACTION", required=True)
    d = docs_sub.add_parser("ingest", help="index files or directories (.md .txt .pdf ...)")
    _common_options(d, suppress=True)
    d.add_argument("paths", nargs="+", type=Path)
    d = docs_sub.add_parser("ask", help="answer a question from the indexed docs")
    _common_options(d, suppress=True)
    d.add_argument("question", nargs="+")
    d.add_argument("--explain", action="store_true",
                   help="also print every retrieved excerpt with its similarity score")
    d = docs_sub.add_parser("status", help="show what is indexed")
    _common_options(d, suppress=True)

    p = add("moderate", "run one message through the assist-only moderation pipeline")
    p.add_argument("text", nargs="+")
    p.add_argument("--mentions", type=int, default=0, help="how many users it pings")
    p.add_argument("--threshold", type=float, default=0.6, help="guild threshold (default 0.6)")

    p = add("image", "ask about an image file, like /image")
    p.add_argument("path", type=Path)
    p.add_argument("--question", default="Describe this image in detail.")

    add("stats", "usage counters recorded by the simulator, like /stats")
    add("forget", "wipe the simulated channel's /ask memory, like /forget")
    return parser


def _read_documents(paths: Sequence[Path]) -> Tuple[List[Tuple[Path, bytes]], List[str]]:
    files: List[Tuple[Path, bytes]] = []
    problems: List[str] = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in DOC_EXTS
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
            )
            if not found:
                problems.append(f"{path}: no {'/'.join(DOC_EXTS)} files inside")
            files.extend((p, p.read_bytes()) for p in found)
        elif path.is_file():
            if path.suffix.lower() in DOC_EXTS:
                files.append((path, path.read_bytes()))
            else:
                problems.append(f"{path}: unsupported extension")
        else:
            problems.append(f"{path}: not found")
    return files, problems


def _print(text: str = "") -> None:
    print(text.replace("**", ""))


def _print_reply(reply: Reply) -> int:
    _print(reply.text)
    if reply.note:
        _print(f"({reply.note})")
    return EXIT_OK if reply.ok else EXIT_FAILED


async def run(args: argparse.Namespace) -> int:
    if args.live:
        load_env_file()
    settings = load_settings(create_dirs=False)
    settings.data_dir = args.data_dir or _default_data_dir()
    if args.live:
        if not settings.has_nim_key():
            print("--live needs NVIDIA_API_KEY.\n" + SIGNUP_HINT, file=sys.stderr)
            return EXIT_USAGE
        settings.offline = False
    else:
        settings.offline = True

    async with BotCore(settings) as core:
        guild, channel, user = args.guild, args.channel, args.user
        command = args.command if args.command != "docs" else f"docs_{args.docs_command}"
        await core.db.increment_usage(guild, user, command)
        backend = "NVIDIA NIM" if not core.offline else "offline backend"
        print(f"# {command.replace('_', ' ')} | {backend} | data: {settings.data_dir}", file=sys.stderr)

        if args.command == "ask":
            return _print_reply(await core.ask.ask(guild, channel, " ".join(args.prompt)))

        if args.command == "summarize":
            if args.file == "-":
                raw = sys.stdin.read()
            else:
                raw = Path(args.file).read_text(encoding="utf-8", errors="replace")
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if args.from_line is not None:
                start = max(1, args.from_line)
                window = lines[start - 1 : start - 1 + max(1, args.count)]
                reply = await core.summarizer.summarize(
                    window, keep="oldest", anchor=f"line {start}"
                )
            else:
                reply = await core.summarizer.summarize(lines[-max(1, args.count):])
            return _print_reply(reply)

        if args.command == "docs":
            if args.docs_command == "ingest":
                files, problems = _read_documents(args.paths)
                for problem in problems:
                    print(f"skipped {problem}", file=sys.stderr)
                documents = []
                for path, data in files:
                    text = await document_text(path.name, data)
                    if text.strip():
                        documents.append((path.name, text))
                where = ", ".join(str(p) for p in args.paths)
                reply = await core.docs.ingest(guild, documents, where=where)
                code = _print_reply(reply)
                report = reply.data.get("report")
                if report is not None and report.chunks:
                    _print(f"embedding model: {report.embed_model} ({report.dim}-d)")
                return code
            if args.docs_command == "ask":
                question = " ".join(args.question)
                reply = await core.docs.ask(guild, question)
                code = _print_reply(reply)
                answer = reply.data.get("answer")
                if args.explain and answer is not None:
                    _print("\nretrieved (cosine similarity, floor "
                           f"{core.rag.min_score:.2f}):")
                    for n, (chunk, score) in enumerate(answer.hits, 1):
                        preview = " ".join(chunk.text.split())[:90]
                        _print(f"  [{n}] {score:.3f}  {chunk.source}: {preview}...")
                return code
            _print("\n".join(core.docs.status_lines(guild)))
            return EXIT_OK

        if args.command == "moderate":
            text = " ".join(args.text)
            result = await core.moderation.review_text(text, args.mentions, args.threshold)
            heur = result.heuristic
            _print(f"prefilter score: {heur.score:.2f} (floor {result.floor:.2f})"
                   f"  reasons: {', '.join(heur.reasons) or 'none'}")
            if not result.llm_called:
                _print("model call: skipped (below the prefilter floor, no API cost)")
            elif result.verdict is None:
                _print("model call: made, but the verdict was unusable")
            else:
                v = result.verdict
                _print("model verdict: " + json.dumps(
                    {"flag": v.flag, "category": v.category,
                     "severity": round(v.severity, 2), "rationale": v.rationale}))
            _print(f"advisory posted to the mod channel: {'YES' if result.report else 'no'}"
                   f" (threshold {args.threshold:.2f}; the bot never acts on its own)")
            return EXIT_OK

        if args.command == "image":
            if not args.path.is_file():
                print(f"{args.path}: not found", file=sys.stderr)
                return EXIT_USAGE
            data = args.path.read_bytes()
            reply = await core.vision.describe(data, None, args.question)
            image = reply.data.get("image")
            if image is not None:
                sent = f"{image.width}x{image.height} {image.mime}, {len(image.data) // 1024} KB, " \
                       f"{image.b64_len} base64 chars (budget {settings.image_max_b64})"
                _print(f"input: {image.original_width}x{image.original_height} "
                       f"{image.original_format}, {image.original_bytes // 1024} KB")
                _print(f"sent to vision model: {sent}"
                       + ("" if image.changed else " (unchanged)"))
            return _print_reply(reply)

        if args.command == "stats":
            blocks = stats_lines(await core.db.usage_stats(guild), label_user=lambda uid: f"user {uid}")
            _print(blocks["total"] + " · " + blocks["memory"])
            _print("\nper command:\n" + blocks["commands"].replace("`", ""))
            _print("\ntop users:\n" + blocks["users"])
            return EXIT_OK

        if args.command == "forget":
            removed = await core.db.clear_history(channel)
            _print(f"Cleared {removed} stored message(s) of context for channel {channel}.")
            return EXIT_OK
    return EXIT_USAGE  # pragma: no cover - argparse enforces a command


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            # Pipes/files get UTF-8 (a Windows code page can not encode the
            # bullets); a real console already handles Unicode.
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
