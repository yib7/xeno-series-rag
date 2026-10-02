"""Ask the Xeno wiki RAG bot from the terminal."""

import argparse
import sys

from .config import load_config
from .errors import SetupError
from .parse_wikitext import _BASE_GAMES
from .router import TIERS


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _game_code(text: str) -> str:
    """A known base game code (case-insensitive). Free text would silently disable the filter and be
    copied into the model prompt, so an unknown code is rejected here the way the web layer does."""
    code = text.strip().upper()
    if code not in _BASE_GAMES:
        raise argparse.ArgumentTypeError(
            f"unknown game code {text!r}; choose one of {', '.join(sorted(_BASE_GAMES))}")
    return code


def main(argv=None, answer_fn=None) -> None:
    parser = argparse.ArgumentParser(description="Ask the Xeno Series Wiki RAG chatbot.")
    parser.add_argument("--question", "-q", required=True, help="the question to ask")
    parser.add_argument("--game", type=_game_code, default=None,
                        help="restrict to one game code (e.g. XC3, XC2, XG); omit for all")
    parser.add_argument("--k", type=_positive_int, default=None, help="number of chunks to retrieve")
    parser.add_argument("--tier", choices=TIERS, default=None,
                        help="force an answer tier instead of auto-routing (fast | thinking | scholar)")
    args = parser.parse_args(argv)

    if answer_fn is None:
        from . import rag
        answer_fn = rag.answer

    try:
        cfg = load_config()
        result = answer_fn(args.question, cfg=cfg, game_filter=args.game, k=args.k, tier=args.tier)
    except SetupError as exc:
        # An expected first-run problem (no config, no API key, no vector store): the message already
        # says what to do. `sys.exit(str)` prints just it and exits 1, matching scripts/setup.py.
        sys.exit(f"error: {exc}")
    except Exception as exc:  # noqa: BLE001 (CLI boundary: never a raw traceback to the console)
        # answer_fn (retrieval + GeminiClient) is unwrapped, unlike the web app's /ask (which turns
        # every failure into an SSE `error` event). Without this, any unexpected failure would
        # propagate as a raw Python traceback, including internal file paths.
        sys.exit(f"error: {type(exc).__name__}: {exc}")

    # Redirected output on Windows uses the ANSI codepage, which cannot encode some wiki titles
    # (e.g. "Alpha (∞)"). Replace those characters rather than crash after a paid answer.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    print(result["answer"])
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            if isinstance(s, dict):
                title = s.get("title") or s.get("url")
                print(f"  - {title}: {s.get('url')}")
            else:
                print(f"  - {s}")
    if result.get("tier"):
        print(f"[{result['tier']} mode]", file=sys.stderr)


if __name__ == "__main__":
    main()
