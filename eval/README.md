# Evaluation harness

How the retrieval quality of this RAG system was measured and tuned.

## What it does

`run_eval.py` runs a fixed set of 40 questions (5 per game across all 8 Xeno titles) through the
production pipeline: hybrid retrieval, reranking, the grounded prompt, and live Gemini
generation. For every question it records the answer, the deduped source links, and the game tag and
title of every retrieved chunk.

The chunk game tags are the objective signal. With the filter set to game G, every retrieved chunk
should be tagged G or `series`; anything tagged a different base game is cross-game leakage worth
inspecting. The questions deliberately include cross-game stressors (for example, "the Zohar" appears
in both Xenogears and Xenosaga) to surface tagging bugs.

## Running it

Generation calls hit the live Gemini API and spend credits, so this is a deliberate, manual run:

```
.venv\Scripts\python.exe -m eval.run_eval      # writes eval/results.jsonl (incremental) + results.json
.venv\Scripts\python.exe -m eval.analyze       # summarize tags, retrieval gaps, and answers
```

`results.json` in this directory is a captured run, kept as evidence behind the written reports.

## Gold question set (answer key)

`gold_questions.json` is a 200-question answer key (25 per game across all 8 titles) with a
documented correct answer for each, grounded in the indexed corpus and linked to its source wiki
page. `QUESTIONS.md` is the human-readable view. Categories span characters, enemy/boss stats,
art/attack values, collectible locations, quests, world/lore, items, and mechanics.

`run_gold_eval.py` scores retrieval against this set: for each question it runs the production
hybrid retrieval with the question's game filter and checks whether the gold source page appears
among the retrieved chunks (the "source-page hit rate"). This is free (no LLM) and is the
signal an embedding-model swap moves. A higher hit rate means the embedder surfaces the page that
actually contains the answer more often.

```
.venv\Scripts\python.exe -m eval.run_gold_eval                 # retrieval-only, free
.venv\Scripts\python.exe -m eval.run_gold_eval --generate      # also generate answers (spends credits)
```

The production embedder is `Qwen/Qwen3-Embedding-0.6B`, so the default run uses `config.yaml` as-is.
To A/B a different embedding model without editing config, pass overrides. The vectorstore on disk
must have been built with the model you name, or query and document vectors won't match:

```
# default: the production Qwen store
... -m eval.run_gold_eval
# a candidate model, against a store built with it (left padding/instruction set to match):
... -m eval.run_gold_eval --embed-model <hf-model-id> --embed-device cpu \
      --query-instruction "<that model's query instruction>"
```

This is how the move to Qwen was vetted against the previous `bge-base-en-v1.5` baseline. Both hit
98.5% on the gold set, with Qwen never ranking the answer page worse. Tracing the three shared misses
showed all three were answer-key faults (a wrong source page, an unanswerable mechanic question, and
a per-game tagging gap), not retrieval failures; correcting them (set v2) takes Qwen to 200/200.
See [`docs/eval/2026-06-27-qwen-vs-bge.md`](../docs/eval/2026-06-27-qwen-vs-bge.md).

Per-question results (retrieved pages + the gold page's rank) stream to `eval/gold_results.jsonl`.

## Jev gate eval (paid)

`run_jev_gates_eval.py` measures the two Jev gates, the off-topic check and the answerability check,
against the gold set plus `jev_gates_cases.json` (30 hand-written off-topic questions, 20 questions
the wiki does not cover, and 10 follow-ups). It runs the production `route()` and `rag._ground()` paths,
records the raw Jev confidences, and replays the decision at 0.7, 0.8, and 0.9 to pick the shipped
thresholds.

```
.venv\Scripts\python.exe -m eval.run_jev_gates_eval --limit 20 --max-calls 50   # smoke run
.venv\Scripts\python.exe -m eval.run_jev_gates_eval                             # full run, about 570 Jev calls
```

This script makes live paid Jev calls on purpose and never calls Gemini. A call budget (default
1,000) aborts the run early; `--max-calls` tightens it. Results stream to
`eval/jev_gates_results.jsonl` (gitignored).

## The findings

The narrative reports (the before/after comparison with the highest-impact bug found and fixed, and
the embedding-model evaluation) live in [`docs/eval/`](../docs/eval/). The headline result was a
cross-subseries tagging fix: cameo characters such as KOS-MOS were being tagged by a cameo
appearance and hidden from their home game filter, which the evaluation caught and the multi-tag membership schema resolved.
