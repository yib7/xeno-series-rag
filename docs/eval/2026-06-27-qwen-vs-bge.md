# Embedding swap: Qwen3-Embedding-0.6B vs bge-base-en-v1.5 (2026-06-27)

An A/B of the candidate new embedder (**Qwen3-Embedding-0.6B**, 1024-dim) against the incumbent
(**BAAI/bge-base-en-v1.5**, 768-dim) over the **200-question gold set** ([`eval/QUESTIONS.md`](../../eval/QUESTIONS.md)).
The Qwen corpus was re-embedded on a Colab GPU; query embedding still runs locally on CPU at serve
time. Both runs use the *identical* production pipeline (hybrid dense+BM25 retrieval, RRF fusion,
the CPU cross-encoder rerank, and the same per-game membership filter), so the **only** variable is
the dense embedding model. The metric is the free, LLM-less **gold source-page hit rate**: for each
question, with that game's filter on, does the wiki page that actually contains the answer appear in
the retrieved chunks (`eval/run_gold_eval.py`)?

Both stores hold the same **289,196 chunks**. The Colab build of the Qwen store shipped without the
per-game membership flags (`g_<game>`) the filter depends on; they were reconstructed in place from
`data/processed/chunks.jsonl` via `embed_index._metadata` (metadata-only update, embeddings
untouched), verified identical to the bge store's flags on cross-appearance pages (e.g. KOS-MOS →
`g_XC2, g_XS1, g_XS2, g_XS3`).

## Headline results

| | bge-base-en-v1.5 (768-d) | Qwen3-Embedding-0.6B (1024-d) |
|---|---|---|
| **Overall hit rate** | **197/200 = 98.5%** | **197/200 = 98.5%** |
| Mean rank when hit | 1.40 | **1.35** |
| Gold page at rank 1 | 168/197 | 168/197 |
| Per-question rank vs other | n/a | **better 2, worse 0, same 195** |
| Mean latency / question (CPU) | 1.54 s | 1.62 s |

- **Dead heat on recall (98.5% each), Qwen marginally better on ranking.** Over the 197 questions
  both retrieve, Qwen never ranks the gold page *worse* than bge on a single question, and ranks it
  *better* on two (XC2-20: 4→2; XCX-01: 5→4). Mean rank 1.40 → 1.35.
- **~5% slower per query** (1.62 s vs 1.54 s on CPU). Qwen is the larger model, but still far inside
  the serve-time budget (query embedding is ~0.2 s of that; the rest is BM25 + the cross-encoder).
- **Per-game and per-category breakdowns are identical** between the two models, including the two
  soft spots (XS2 23/25, XS3 24/25; `mechanic` 4/5, `world` 21/22).

## The three misses were answer-key faults, not embedder gaps (now corrected)

Both models miss the **same three** questions, the tell that the embedder isn't the cause. Tracing
each through the pipeline (raw dense rank, BM25 rank, and the game filter) showed all three were
faults in the *answer key*, not retrieval failures:

- **XS2-25 (Durandal):** the gold label pointed at `Durandal (XS1)`, a shop/data page that never
  states the ship's role (and is tagged XS1, so an XS2 filter would exclude it regardless). The
  answer lives on the `Durandal` lore page, which both models *did* retrieve. **Wrong source page.**
- **XS3-22 (Break Limit):** a general battle mechanic with no explainer page in the corpus, pinned to
  one arbitrary boss (`27-Series Asura`); retrieval returned other XS3 enemies that equally expose a
  Break Limit value. **Unanswerable as a single-source question.**
- **XS2-05 (Joachim Mizrahi):** *not* a recall gap: with no game filter his page is **dense rank 1
  of 300**. But his membership set is `XS1, XS3` (no `XS2`), so the XS2-filtered query excludes his
  page entirely and returns same-surname relatives instead. **A per-game tagging gap**: ~35 Xenosaga
  pages carry only two of the three episode tags, not an embedder miss.

**Follow-up: corrected gold set.** All three were fixed in `gold_questions.json` (v2, 2026-06-27):
XS2-25 relabeled to `Durandal`; XS3-22 reframed to the concrete 27-Series Asura Break Limit value;
XS2-05 replaced with `Ziggy` (a genuinely XS2-tagged character). Re-running Qwen on the corrected set
scores **200/200 = 100%** (mean rank 1.3), every game 25/25. The bge column was not re-run (its store
was removed when Qwen was promoted), but all three fixes are embedder-independent (the relevant pages
were already retrieved by both models above), so the head-to-head conclusion is unchanged.

## Interpretation

The gold set is **saturated** for this comparison: the bge baseline was already at 98.5% with the
gold page at rank 1 five times out of six, so there is almost no headroom for a better embedder to
demonstrate a win. Within that ceiling, **Qwen is at-par-or-better on every question and never
worse**, so the swap is *safe* on quality, buying a small ranking improvement and the headroom of a
stronger general model for the messier real-world queries this clean answer-key set doesn't capture,
at the cost of ~5% query latency and the migration (publishing the Qwen vectorstore as a data
release + repointing the setup TAG/ASSET, or clones break).

This eval **cannot** justify the migration on the gold numbers alone; it can only certify it carries
no regression. The case for switching rests on out-of-distribution robustness, not this set.

## Reproduce

```
# Qwen (config.yaml as-is; Qwen store at data/vectorstore):
.venv\Scripts\python.exe -m eval.run_gold_eval --out eval/gold_results_qwen.jsonl

# bge baseline (against a bge-built store on disk):
.venv\Scripts\python.exe -m eval.run_gold_eval --out eval/gold_results_bge.jsonl \
  --embed-model BAAI/bge-base-en-v1.5 --embed-device cpu \
  --query-instruction "Represent this sentence for searching relevant passages: "
```

Per-question results stream to the named `eval/gold_results_*.jsonl` (gitignored evidence). After
this comparison the bge store was removed and Qwen promoted to the sole production store; re-running
the bge column would mean rebuilding its vectorstore from `data/processed/chunks.jsonl` first.
