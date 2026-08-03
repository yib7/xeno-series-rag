"""One-off evaluation harness: 5 questions x 8 games (40 total) with the per-game filter ON.

For each question we run the *real* production retrieval + prompt + Gemini generation, and record
the answer, the deduped source URLs, AND the game tag + title of every retrieved chunk. The chunk
game tags are the objective signal for cross-game leakage: with a filter on game G, every retrieved
chunk should be tagged G or "series"; anything else (or a "series" chunk whose content is really a
different game) is a leak to inspect.

Writes incrementally to eval/results.jsonl (one line per question, crash-safe) and a final
eval/results.json. Authorized by the user to spend API credits on these specific calls.
"""

import json
import time
from pathlib import Path

from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.rag import GeminiClient, _dedupe_sources, build_prompt
from xeno_rag.retrieve import retrieve

# 5 questions per game spanning story / characters / combat / items / world. Chosen to be
# answerable from the wiki and verifiable, with a few deliberate cross-game stressors
# (e.g. "Zohar" appears in both Xenogears and Xenosaga).
QUESTIONS = {
    "XG": [
        "Who is the main protagonist of Xenogears?",
        "What are Deathblows and how do you perform them?",
        "What are Gears in Xenogears?",
        "What is the Zohar?",
        "Who is Citan Uzuki?",
    ],
    "XS1": [
        "Who is KOS-MOS?",
        "Who is Shion Uzuki?",
        "What are the Gnosis?",
        "How does the Boost system work in battle?",
        "What is the Zohar in Xenosaga?",
    ],
    "XS2": [
        "How does the battle system work in Xenosaga Episode II?",
        "Who is Gaignun Kukai Jr.?",
        "Who is Albedo?",
        "What are E.S. units?",
        "Who is Jin Uzuki?",
    ],
    "XS3": [
        "Who is T-elos?",
        "Who is Wilhelm?",
        "What is the battle system in Xenosaga Episode III?",
        "What is the role of the Zohar in Xenosaga Episode III?",
        "Who is chaos?",
    ],
    "XC1": [
        "Who is Shulk?",
        "What is the Monado?",
        "How do Arts work in Xenoblade Chronicles?",
        "Who is Metal Face?",
        "What is a Chain Attack?",
    ],
    "XC2": [
        "Who is Rex?",
        "What are Blades and Drivers?",
        "How do Driver Combos work?",
        "Who is Pyra?",
        "What is Elysium?",
    ],
    "XC3": [
        "Who is Noah?",
        "What is Interlinking and Ouroboros?",
        "What are Keves and Agnus?",
        "What is a flame clock?",
        "Who is N?",
    ],
    "XCX": [
        "What is New Los Angeles?",
        "What are Skells?",
        "What are mimeosomes?",
        "How do Soul Voices work in combat?",
        "Who is Elma?",
    ],
}

BASE_GAMES = {"XG", "XS1", "XS2", "XS3", "XC1", "XC2", "XC3", "XCX"}


def generate_with_retry(llm, system, user, tries=4):
    """Generate, retrying transient API errors with backoff. Returns (text, error_str_or_None)."""
    last = None
    for attempt in range(tries):
        try:
            return llm.generate(system, user), None
        except Exception as exc:  # noqa: BLE001 - record and back off
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return None, last


def main():
    cfg = load_config()
    out_dir = Path("eval")
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    json_path = out_dir / "results.json"

    print(f"model: {cfg.get('gemini_model')}")
    embedder = Embedder(cfg)
    llm = GeminiClient(cfg)

    all_q = [(g, q) for g, qs in QUESTIONS.items() for q in qs]
    total = len(all_q)
    results = []

    # fresh run: truncate the jsonl
    jsonl_path.write_text("", encoding="utf-8")

    for i, (game, q) in enumerate(all_q, 1):
        t0 = time.time()
        chunks = retrieve(q, cfg, game_filter=game, embedder=embedder)
        system, user = build_prompt(q, chunks)
        ans, err = generate_with_retry(llm, system, user)
        elapsed = round(time.time() - t0, 1)

        chunk_info = [
            {
                "title": c.get("title"),
                "game": c.get("game"),
                "url": c.get("url"),
                "heading": c.get("heading"),
                "distance": round(c["distance"], 4) if c.get("distance") is not None else None,
            }
            for c in chunks
        ]
        # leakage signal: chunk tags that are a base game other than the filter
        tag_counts = {}
        for c in chunk_info:
            tag_counts[c["game"]] = tag_counts.get(c["game"], 0) + 1
        foreign_tags = [t for t in tag_counts if t in BASE_GAMES and t != game]

        rec = {
            "idx": i,
            "game_filter": game,
            "question": q,
            "answer": ans,
            "error": err,
            "sources": _dedupe_sources(chunks),
            "n_chunks": len(chunks),
            "tag_counts": tag_counts,
            "foreign_base_tags": foreign_tags,
            "chunks": chunk_info,
            "elapsed_s": elapsed,
        }
        results.append(rec)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        status = "ERR" if err else "ok"
        flag = " LEAK-TAG" if foreign_tags else ""
        print(f"[{i:2d}/{total}] [{game:3s}] {status} {elapsed:4.1f}s{flag}  {q}")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    errs = sum(1 for r in results if r["error"])
    leaks = sum(1 for r in results if r["foreign_base_tags"])
    print(f"\nDONE: {total} questions, {errs} errors, {leaks} with foreign base-game tags.")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
