# Design — Xeno RAG, Cycle 3: UX polish

Date: 2026-06-25. Status: approved (user "ready", 2026-06-25). Branch: `autopilot/cycle3-ux-polish`.

Four user-requested UX improvements, **app-layer only** — the ChromaDB collection, BM25 index, and
all corpus/embeddings are untouched (no re-embed, no re-tag). The live LLM is never called in tests
(MockLLM); no emojis; every answer keeps its CC-BY-SA source links.

## Goals (frozen scope)

1. **Relevance-scored, size-tiered source bubbles.** Sources are shown ordered by how correlated they
   are to the question, with bubble *size* reflecting that correlation (most-correlated = topmost +
   biggest), packed so variable-size bubbles don't leave large gaps. **No inline `[n]` section
   tagging** — the user explicitly cut that; the answer prose stays clean and grounded as today.
2. **Original cosmic SVG background** for the all-games "Xeno Series" view (the current flat CSS
   gradient/starfield reads as boring). Hand-authored SVG so it ships in the public repo.
3. **Answer-style depth tuning** — "Faster" (flash-lite) pulls slightly fewer chunks; "Thinking"
   (flash-3.5) stays the heavyweight that covers questions needing more leverage.
4. **Faster/Thinking distinction copy** — make the UI clearly say Faster suffices for most questions
   and Thinking reads more of the wiki at once (larger aggregated context) for complex, multi-topic
   questions.

Out of scope (→ BACKLOG): inline citation markers, any corpus re-embed/re-tag, server-side scoring
beyond the existing reranker.

## SP1 — Relevance-scored, size-tiered source bubbles

**Problem.** `_dedupe_sources` already emits sources in relevance order, but every bubble is the same
size and the page never sees *how* relevant each is. The cross-encoder reranker computes a precise
per-chunk relevance score (`rerank.py`) and then throws it away after sorting.

**Backend data flow.**
- `rerank.Reranker.rerank()` — attach the score onto each returned item
  (`it["_score"] = float(scores[i])`) while preserving the existing best-first order and tie behavior.
  This is the only change to scoring; no new model call.
- `retrieve.retrieve()` / `embed_index.cap_per_page()` — carry the dict through unchanged (the score
  rides along on each chunk).
- `rag._dedupe_sources(chunks)` — for each cited page, read the best (first-seen) chunk's `_score` and
  derive:
  - `relevance`: a 0–1 value, **min-max normalized across the returned source set** (top source → 1.0,
    bottom → 0.0; if all equal, all 1.0).
  - `tier`: a discrete size bucket from `relevance` thresholds — `"high"` (≥ ~0.66), `"med"`
    (≥ ~0.33), else `"low"`. The #1 source is always `high`.
  - **Fallback** when the reranker is off (`use_reranker: false`) or no `_score` is present: derive
    `relevance`/`tier` from **rank position** (index in the ordered list) so the feature degrades
    gracefully instead of disappearing.
- The SSE `sources` event dicts gain `relevance` (float) and `tier` (string). These are *additive*
  keys — legacy consumers and the existing string-url tolerance are unaffected.

**Frontend (`render.js` `sourcesHtml` + `index.html` CSS).**
- Each bubble gets a `tier-high|med|low` class driving its size (font-size / padding / min-width and,
  for high tier, a 2-line snippet vs. clamped for lower tiers).
- Render order is unchanged (already sorted best-first) → most-correlated bubble is first and biggest.
- Replace the uniform `flex-wrap` chip row with a **CSS multi-column / masonry-style** packed layout so
  the differently-sized bubbles fill space without big ragged gaps. Keep the collapsed
  `<details><summary>Sources (N)</summary>` wrapper exactly as today.

**Tests (TDD).**
- PY: `rerank` attaches `_score` and preserves order; `_dedupe_sources` carries normalized
  `relevance` + correct `tier`; the rank-based fallback path (no `_score`).
- JS: `sourcesHtml` emits the right `tier-*` class per source, preserves order, still HTML-escapes
  name/snippet/url, and tolerates legacy bare-string sources.

**Checkpoint.** `pytest tests/test_rag.py tests/test_rerank.py tests/test_retrieve.py -q` green +
`node --test tests/js/` green, with the new assertions; a live preview shows bubbles of differing size
in correlation order with no large gaps.

## SP2 — Original cosmic SVG background (all-games)

**Approach.** Behind the `html[data-game="all"]` state, render a richer **original inline SVG** scene
as a fixed, full-bleed wash (replacing / layering over the current `body::before` CSS starfield):
deep-space vertical gradient, a large faint Zohar/monolith silhouette echoing the header brand mark,
soft nebula clouds (radial gradients), a distant planet / Conduit glow, and layered parallax stars.
Faded and bottom-masked (like `.art-wash`) so body prose stays legible — the answer card sits on its
own solid `--surface`, so its text is unaffected regardless. The per-game `.art-wash` raster mechanism
is untouched and still overrides for a selected game. All original SVG (no copyrighted assets) → it
**commits to the repo**, unlike the gitignored game key-art.

**Checkpoint.** Live preview at desktop and mobile widths: the cosmic scene renders behind the
all-games view, text remains legible, and switching to a game still shows that game's key-art wash.

## SP3 — Answer-style depth + Faster/Thinking copy

- `config.yaml` `answer_styles`: `gemini-3.1-flash-lite` (Faster) `top_k 20 / max_chunks_per_page 5`
  → **`14 / 4`**. `gemini-3.5-flash` (Thinking) unchanged (`40 / 6`, wider candidate pools).
- `index.html`: enrich the "Answer style" `<select>` option labels and add a one-line helper under the
  control — Faster = best for most questions (quick, focused lookups); Thinking = reads more of the
  wiki at once, for complex multi-topic questions.
- The existing `answer_styles` tests in `test_rag.py` must stay green (Thinking is still strictly
  deeper than Faster: 40 > 14 top_k, 6 > 4 cap).

**Checkpoint.** `pytest tests/test_rag.py -q` green (style tests still pass); config shows the new
Faster depth; the UI shows the new labels + helper.

## SP4 — Final verification + write-up

Full suite (`.venv\Scripts\python.exe -m pytest -q` + `node --test tests/js/`) green; live preview
smoke of all three visible changes (cosmic bg, sized source bubbles in order without gaps, selector
copy). Write `docs/eval/2026-06-25-ux-polish.md`; append Cycle 3 to `MILESTONES.md` + refresh the
current-state snapshot; run cycle teardown (sweep `.superpowers/sdd/*` if any). Then
`finishing-a-development-branch` → present merge options (merge to `main` is the human gate).

## Risks / mitigations

- **Score normalization edge cases** (single source, all-equal scores, reranker disabled) → explicit
  fallback to rank-based tiers; unit-tested.
- **Cosmic SVG hurting legibility** → bottom mask + low opacity + the answer card's own solid surface;
  verified on the live preview, not by eye alone.
- **Reversibility** — every change is small and app-layer; how-to-undo logged per phase in
  `DECISIONS.md`. Isolated on a branch; nothing touches `main` until the human merge.
