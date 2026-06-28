# Evaluation harness

How the retrieval quality of this RAG system was measured and tuned.

## What it does

`run_eval.py` runs a fixed set of 40 questions (5 per game across all 8 Xeno titles) through the
**real** production pipeline: hybrid retrieval, reranking, the grounded prompt, and live Gemini
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

`gold_questions.json` is a 200-question answer key — 25 per game across all 8 titles — with a
documented correct answer for each, grounded in the indexed corpus and linked to its source wiki
page. `QUESTIONS.md` is the human-readable view. Categories span characters, enemy/boss stats,
art/attack values, collectible locations, quests, world/lore, items, and mechanics.

`run_gold_eval.py` scores **retrieval** against this set: for each question it runs the production
hybrid retrieval with the question's game filter and checks whether the gold source page appears
among the retrieved chunks (the "source-page hit rate"). This is free (no LLM) and is exactly the
signal an embedding-model swap moves — a higher hit rate means the embedder surfaces the page that
actually contains the answer more often.

```
.venv\Scripts\python.exe -m eval.run_gold_eval                 # retrieval-only, free
.venv\Scripts\python.exe -m eval.run_gold_eval --generate      # also generate answers (spends credits)
```

To A/B two embedding models (for example `bge-base-en-v1.5` vs `Qwen3-Embedding-0.6B`) without
editing `config.yaml`, pass overrides. The vectorstore on disk must have been built with the model
you name, or query and document vectors won't match:

```
# baseline against the current bge vectorstore
... -m eval.run_gold_eval --embed-model BAAI/bge-base-en-v1.5 --embed-device cpu \
      --query-instruction "Represent this sentence for searching relevant passages: "
# the new model (after building its vectorstore on Colab and unzipping it in place)
... -m eval.run_gold_eval        # uses config.yaml as-is
```

Per-question results (retrieved pages + the gold page's rank) stream to `eval/gold_results.jsonl`.

## The findings

The narrative reports (before/after comparisons, the highest-impact bug found and fixed, and a second
evaluation round) live in [`docs/eval/`](../docs/eval/). The headline result was a cross-subseries
tagging fix: cameo characters such as KOS-MOS were being tagged by a cameo appearance and hidden from
their home game filter, which the evaluation caught and the multi-tag membership schema resolved.
