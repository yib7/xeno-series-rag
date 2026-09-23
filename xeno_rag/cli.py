"""Ask the Xeno wiki RAG bot from the terminal."""

import argparse
import sys

from .config import load_config
from .router import TIERS


def main(argv=None, answer_fn=None) -> None:
    parser = argparse.ArgumentParser(description="Ask the Xeno Series Wiki RAG chatbot.")
    parser.add_argument("--question", "-q", required=True, help="the question to ask")
    parser.add_argument("--game", default=None,
                        help="restrict to one game code (e.g. XC3, XC2, XG); omit for all")
    parser.add_argument("--k", type=int, default=None, help="number of chunks to retrieve")
    parser.add_argument("--tier", choices=TIERS, default=None,
                        help="force an answer tier instead of auto-routing (fast | thinking | scholar)")
    args = parser.parse_args(argv)

    if answer_fn is None:
        from . import rag
        answer_fn = rag.answer

    cfg = load_config()
    try:
        result = answer_fn(args.question, cfg=cfg, game_filter=args.game, k=args.k, tier=args.tier)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: never a raw traceback to the console
        # answer_fn (retrieval + GeminiClient) is unwrapped, unlike the web app's /ask (which turns
        # every failure into a generic SSE `error` event). Without this, an expected first-run bad
        # path (no GOOGLE_API_KEY / GEMINI_API_KEY set) would propagate as a raw Python traceback
        # printed to stderr, including internal file paths. `sys.exit(str)` prints just the message
        # and exits 1, matching scripts/setup.py's existing convention for user-facing CLI failures.
        sys.exit(f"error: {exc}")

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
