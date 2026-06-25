"""Ask the Xeno wiki RAG bot from the terminal."""

import argparse

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
        cfg["gemini_model"] = args.model
    result = answer_fn(args.question, cfg=cfg, game_filter=args.game, k=args.k)

    print(result["answer"])
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            if isinstance(s, dict):
                title = s.get("title") or s.get("url")
                print(f"  - {title} — {s.get('url')}")
            else:
                print(f"  - {s}")


if __name__ == "__main__":
    main()
