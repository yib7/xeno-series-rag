"""Evaluate retrieval against the gold question set (eval/gold_questions.json).

For each of the 200 gold questions, this runs the *real* production hybrid retrieval with the
question's game filter and checks whether the gold **source page** (the wiki page that actually
contains the answer) appears among the retrieved chunks. That "source-page hit rate" is a free,
LLM-less metric and is exactly the signal an embedding-model swap moves: it measures whether the
embedder surfaces the right page. Run it once per model (e.g. bge-base, then Qwen3-Embedding-0.6B)
and compare the hit rates - higher is better.

With --generate it additionally runs the grounded prompt + Gemini generation and records each answer
next to its gold answer for manual/LLM grading. That spends API credits, so it is OFF by default.

Usage:
  python -m eval.run_gold_eval                  # retrieval-only, free; writes eval/gold_results.jsonl
  python -m eval.run_gold_eval --top-k 20       # override retrieval depth
  python -m eval.run_gold_eval --limit 24       # quick smoke over the first N questions
  python -m eval.run_gold_eval --generate       # also generate answers (COSTS API credits)

To A/B two embedding models, point config.yaml at model A, run this, save the printed summary;
repeat with model B against its vectorstore. The vectorstore must have been built with the same
model the config currently names, or query/doc vectors won't match.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve

GOLD = Path("eval") / "gold_questions.json"


def _norm(u: str) -> str:
    """Normalize a wiki URL for comparison (trim, drop trailing slash)."""
    return (u or "").strip().rstrip("/")


def gold_hit(chunks, gold_url: str, gold_title: str):
    """Return (hit, rank) where rank is the 1-based position of the first retrieved chunk on the
    gold source page (matched by URL, with a title fallback), or (False, None) if not retrieved."""
    gu = _norm(gold_url)
    for i, c in enumerate(chunks, 1):
        if _norm(c.get("url")) == gu or (c.get("title") and c.get("title") == gold_title):
            return True, i
    return False, None


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval against the gold question set.")
    ap.add_argument("--top-k", type=int, default=None, help="override cfg top_k retrieval depth")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--generate", action="store_true", help="also generate answers (COSTS credits)")
    ap.add_argument("--out", default=str(Path("eval") / "gold_results.jsonl"))
    # A/B overrides: evaluate a model without editing config.yaml. The vectorstore on disk must have
    # been built with whichever model you name here, or query/doc vectors won't match.
    ap.add_argument("--embed-model", default=None, help="override cfg embed_model (e.g. for an A/B run)")
    ap.add_argument("--embed-device", default=None, help="override cfg embed_device (cpu | auto | directml)")
    ap.add_argument("--query-instruction", default=None, help="override the query instruction prefix")
    args = ap.parse_args()

    cfg = load_config()
    if args.top_k:
        cfg["top_k"] = args.top_k
    if args.embed_model:
        cfg["embed_model"] = args.embed_model
    if args.embed_device:
        cfg["embed_device"] = args.embed_device
    if args.query_instruction is not None:
        cfg["query_instruction"] = args.query_instruction
    gold = json.loads(GOLD.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        gold = gold[: args.limit]

    print(f"embed_model: {cfg.get('embed_model')}  | top_k: {cfg.get('top_k')}  | questions: {len(gold)}")
    embedder = Embedder(cfg)

    llm = None
    if args.generate:
        from xeno_rag.rag import GeminiClient, build_prompt  # noqa: F401 (build_prompt used below)
        llm = GeminiClient(cfg)

    out_path = Path(args.out)
    out_path.write_text("", encoding="utf-8")

    hits = 0
    per_game = defaultdict(lambda: [0, 0])      # game -> [hit, total]
    per_cat = defaultdict(lambda: [0, 0])       # category -> [hit, total]
    ranks = []

    for i, q in enumerate(gold, 1):
        t0 = time.time()
        chunks = retrieve(q["question"], cfg, game_filter=q["game"], embedder=embedder)
        hit, rank = gold_hit(chunks, q["source_url"], q["source_title"])
        hits += hit
        per_game[q["game"]][0] += hit
        per_game[q["game"]][1] += 1
        per_cat[q["category"]][0] += hit
        per_cat[q["category"]][1] += 1
        if hit:
            ranks.append(rank)

        rec = {
            "id": q["id"], "game": q["game"], "category": q["category"],
            "question": q["question"], "gold_answer": q["answer"],
            "gold_source": q["source_url"], "hit": bool(hit), "gold_rank": rank,
            "retrieved": [{"title": c.get("title"), "url": c.get("url")} for c in chunks],
            "elapsed_s": round(time.time() - t0, 2),
        }
        if llm is not None:
            from xeno_rag.rag import build_prompt
            system, user = build_prompt(q["question"], chunks)
            try:
                rec["generated_answer"] = llm.generate(system, user)
            except Exception as exc:  # noqa: BLE001 - record and continue
                rec["generated_answer"] = None
                rec["gen_error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        flag = "HIT " if hit else "miss"
        print(f"[{i:3d}/{len(gold)}] [{q['game']:3s}] {flag} rank={rank}  {q['question'][:60]}")

    n = len(gold)
    print("\n================ RETRIEVAL: gold source-page hit rate ================")
    print(f"OVERALL: {hits}/{n} = {hits / n:.1%}"
          + (f"   | mean rank when hit: {sum(ranks) / len(ranks):.1f}" if ranks else ""))
    print("\nby game:")
    for g in sorted(per_game):
        h, t = per_game[g]
        print(f"  {g:3s}  {h:2d}/{t:2d}  {h / t:.0%}")
    print("\nby category:")
    for c in sorted(per_cat):
        h, t = per_cat[c]
        print(f"  {c:14s}  {h:2d}/{t:2d}  {h / t:.0%}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
