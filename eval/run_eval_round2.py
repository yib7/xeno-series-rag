"""Round-2 evaluation harness: 5 *harder* questions x 8 games (40 total), per-game filter ON.

Round 1 (run_eval.py) was deliberately mainstream — "Who is <lead>?", "What is <core mechanic>?".
Round 2 stress-tests the corpus + retrieval with NICHE and COMPLEX questions:
  - minor characters / NPCs / bosses (Emeralda, Canaan, Dr. Sellers, Voyager, Dunban, Tatsu),
  - deep multi-entity lore (Solaris caste society, the Testaments, the Miltian Conflict, the
    High Entia secret, Fei/Id/Grahf identity),
  - aggregation / numeric answers the grounded-reasoning prompt must reason over (how many Zohar
    Emulators; how many / what are the BLADE divisions),
  - niche mechanics (Ether, Gem Crafting, field skills + Affinity Chart, Overdrive, class-change),
  - a deliberate cross-game leakage stressor: XC2's **Jin** (Flesh Eater, leader of Torna) shares a
    first name with Xenosaga's **Jin Uzuki** — with the XC2 filter on, no Xenosaga Jin chunk may leak.

Same machinery as run_eval.py (real production retrieve + prompt + Gemini). Writes to
eval/results_round2.jsonl (crash-safe) and eval/results_round2.json. User authorized API spend for
these specific calls (gemini-3.1-flash-lite).
"""

import json
import time
from pathlib import Path

from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve
from xeno_rag.rag import build_prompt, _dedupe_sources, GeminiClient

QUESTIONS = {
    "XG": [
        "Who is Emeralda and how was she created?",
        "What is Solaris and how is its caste-based society organized?",
        "What is the relationship between Fei Fong Wong, Id, and Grahf?",
        "How does the Ether system work in Xenogears?",
        "What is the Yggdrasil and how does it change over the course of the game?",
    ],
    "XS1": [
        "What is the U.M.N. (Unus Mundus Network)?",
        "Who is Margulis and what is his role in U-TIC?",
        "What is the Encephalon and how is it used in the story?",
        "What is the Woglinde and what happens to it?",
        "How do characters learn new Tech Attacks and Ether skills in Xenosaga Episode I?",
    ],
    "XS2": [
        "What was the Miltian Conflict and what caused it?",
        "Who is Canaan?",
        "What is a Zohar Emulator and how many Zohar Emulators are there?",
        "Who is Dr. Sellers?",
        "How does the Stock and Break system work in Xenosaga Episode II's combat?",
    ],
    "XS3": [
        "What are the Testaments and who are they?",
        "Who is Voyager?",
        "What is Zarathustra?",
        "Who is Abel and what is Abel's Ark?",
        "What is U-DO and how does it relate to chaos and Wilhelm's plan?",
    ],
    "XC1": [
        "Who is Dunban?",
        "What is the secret of the High Entia race?",
        "Who is Egil and what is his goal?",
        "How does Gem Crafting work in Xenoblade Chronicles?",
        "What is the relationship between the Bionis and the Mechonis?",
    ],
    "XC2": [
        "What is a Titan and how do people live on them in Alrest?",
        "Who is Jin and what is his goal as a leader of Torna?",
        "What are Core Crystals and how is a Blade awakened?",
        "What happened to the kingdom of Torna?",
        "How do field skills and the Affinity Chart work for Blades?",
    ],
    "XC3": [
        "Who are Moebius and the Consuls?",
        "What is the City in Xenoblade Chronicles 3 and who founded it?",
        "Who is Mio and what is her role among the Ouroboros?",
        "What is Origin?",
        "How does the class and class-change system work in Xenoblade Chronicles 3?",
    ],
    "XCX": [
        "What is BLADE and what are its divisions?",
        "Who are the Ganglion?",
        "What is the Lifehold Core?",
        "How does the Overdrive system work in combat?",
        "Who is Tatsu?",
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
    jsonl_path = out_dir / "results_round2.jsonl"
    json_path = out_dir / "results_round2.json"

    print(f"model: {cfg.get('gemini_model')}")
    embedder = Embedder(cfg)
    llm = GeminiClient(cfg)

    all_q = [(g, q) for g, qs in QUESTIONS.items() for q in qs]
    total = len(all_q)
    results = []

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
