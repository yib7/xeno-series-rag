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

## The findings

The narrative reports (before/after comparisons, the highest-impact bug found and fixed, and a second
evaluation round) live in [`docs/eval/`](../docs/eval/). The headline result was a cross-subseries
tagging fix: cameo characters such as KOS-MOS were being tagged by a cameo appearance and hidden from
their home game filter, which the evaluation caught and the multi-tag membership schema resolved.
