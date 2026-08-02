"""Ask the Xeno wiki RAG bot from the terminal."""

import argparse
import sys

from .config import load_config


def main(argv=None, answer_fn=None) -> None:
    parser = argparse.ArgumentParser(description="Ask the Xeno Series Wiki RAG chatbot.")
    parser.add_argument("--question", "-q", required=True, help="the question to ask")
    parser.add_argument("--game", default=None,
                        help="restrict to one game code (e.g. XC3, XC2, XG); omit for all")
    parser.add_argument("--k", type=int, default=None, help="number of chunks to retrieve")
    parser.add_argument("--model", default=None,
                        help="override the Gemini model for this question (e.g. gemini-3.5-flash)")
    args = parser.parse_args(argv)

    if answer_fn is None:
        from . import rag
        answer_fn = rag.answer

    cfg = load_config()
    if args.model:
        if args.model not in cfg.get("answer_styles", {}):
            # Advisory only: still proceed with the override. answer_styles is keyed by model id
            # in config.yaml; an unlisted/typo'd model silently skips the retrieval-depth pairing
            # (falls back to base depth) and would otherwise only surface as a raw SDK error deep
            # in the model call.
            print(
                f"warning: model '{args.model}' has no answer_styles entry in config; "
                "using base retrieval depth",
                file=sys.stderr,
            )
        cfg["gemini_model"] = args.model
    try:
        result = answer_fn(args.question, cfg=cfg, game_filter=args.game, k=args.k)
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


if __name__ == "__main__":
    main()
