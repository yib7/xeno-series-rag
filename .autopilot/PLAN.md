# PLAN — Xeno RAG, Cycle 3: UX polish

> On autopilot. Resume point = the first unchecked box below. **Isolated on branch
> `autopilot/cycle3-ux-polish`** (never `main`; the final merge is the single human gate).
> Autonomy contract: `.autopilot/AUTONOMY.md` (hard-stops: secrets, real money). New ideas →
> `.autopilot/BACKLOG.md`. Shipped history → `.autopilot/MILESTONES.md`.
> Design: `docs/superpowers/specs/2026-06-25-ux-polish-design.md`.

## Scope (frozen)

Four user-requested UX improvements, **app-layer only** (corpus/index untouched, no re-embed/re-tag):
(1) **relevance-scored, size-tiered source bubbles** (most-correlated = topmost + biggest, packed; no
inline `[n]` tagging); (2) **original cosmic SVG background** for the all-games "Xeno Series" view;
(3) **answer-style depth tuning** — Faster (flash-lite) leaner, Thinking (flash-3.5) unchanged;
(4) **Faster/Thinking distinction copy** in the UI.

## Global constraints (every phase inherits)
- **Python 3.12** venv; run via `.venv\Scripts\python.exe`. `PYTHONIOENCODING=utf-8` on Windows.
- **No live LLM calls in tests** — MockLLM only. Corpus/ChromaDB/BM25 untouched (app-layer only).
- TDD: failing test first → watch fail → minimal impl → watch pass. Keep the full suite green per phase.
- JS tests run via `node --test tests/js/` (Node v24), wrapped by `tests/test_frontend_js.py`.
- No emojis in UI or output. Every answer keeps its CC-BY-SA source links.
- Commit each phase on the branch (local only; never push, never touch `main`).

---

## SP1 — Relevance-scored, size-tiered source bubbles

**Checkpoint:** `.venv\Scripts\python.exe -m pytest tests/test_rag.py tests/test_rerank.py tests/test_retrieve.py -q`
green + `node --test tests/js/` green, incl. new tests: `rerank` attaches `_score` (order preserved);
`_dedupe_sources` carries normalized `relevance` (0–1, min-max) + `tier` (high≥0.66 / med≥0.33 / low),
#1 source always `high`, rank-based fallback when no score; JS `sourcesHtml` emits `tier-*` classes in
order, still escapes, tolerates legacy string urls. Live preview: differently-sized bubbles in
correlation order, no large gaps.

- [x] PY tests: `rerank` attaches `_score` + preserves order; `_dedupe_sources` adds normalized
  `relevance`/`tier`; rank-based fallback; single-source→high.
- [x] JS tests: `sourcesHtml` adds the correct `tier-high|med|low` class per source, preserves order;
  legacy no-tier stays a plain chip; escaping intact.
- [x] Implemented: `_score` attach in `rerank.rerank`; `_score_relevance` (min-max norm + tier +
  rank fallback) in `_dedupe_sources`; `tier`/`relevance` ride the SSE `sources` payload; `tier-*`
  sizing + `columns: 2 232px` masonry layout in `render.js`/`index.html` (kept `<details>`).
- [x] Verified: test_rag/test_rerank/test_retrieve green; JS 20/20; full suite **173 passed**; live
  preview (preview_eval, screenshots time out): 5 chips in correlation order, classes
  high/med/med/low/low, computed font 17.6px(high) vs 14.6px(low), 2-col packed (no gaps).

## SP2 — Original cosmic SVG background (all-games "Xeno Series")

**Checkpoint:** live preview at desktop + mobile widths — a richer cosmic SVG scene renders behind the
`data-game="all"` view (deep-space gradient, faint Zohar/monolith silhouette, nebula, distant
planet/Conduit glow, parallax stars), body text stays legible, and selecting a game still shows that
game's key-art wash. `node --check` on the inline script clean; full suite still green (frontend-only).

- [x] Implemented the original inline `<svg class="cosmic">` scene for `html[data-game="all"]` (fixed,
  full-bleed, bottom-masked): gold+turquoise+violet nebula, distant planet (lower-left), faint Zohar
  monolith with turquoise core glow (right), 48 parallax stars (a few gently twinkle,
  `prefers-reduced-motion` respected). Replaced the flat CSS gradient/starfield; body now solid so the
  SVG is the sole cosmic layer. Per-game `.art-wash` raster mechanism untouched.
- [x] Verified (preview_eval, screenshots time out): all-games → cosmic `display:block`, full-bleed
  (1265×720 desktop / 375×812 mobile), z-0 behind `.app` z-1, 48 stars + planet/monolith, mask applied;
  switch to XC2 → cosmic `display:none` + `.art-wash` on; back to all → cosmic returns; sub text visible,
  0 console errors. Inline script unchanged + executed cleanly in-page (game-switch handlers ran).

## SP3 — Answer-style depth tuning + Faster/Thinking copy

**Checkpoint:** `.venv\Scripts\python.exe -m pytest tests/test_rag.py -q` green (the `answer_styles`
tests still pass — Thinking stays strictly deeper); `config.yaml` shows Faster at `top_k 14 / cap 4`;
the UI shows the new selector labels + helper line.

- [x] `config.yaml`: flash-lite `top_k 20 / cap 5` → `14 / 4`; flash-3.5 unchanged (40/6). Confirmed
  via `_apply_answer_style(load_config())`: Faster 14/4, Thinking 40/6.
- [x] `index.html`: labels → "Faster — best for most questions" / "Thinking — deeper, multi-topic" +
  a dynamic `#model-hint` (Faster: "Quick, focused lookups — enough for most questions."; Thinking:
  "Reads more of the wiki at once; for complex, multi-topic questions.").
- [x] Verified: `test_rag.py` 32 passed (answer_styles tests still green — Thinking stays deeper);
  live preview confirms labels + helper switch on model change.

## SP4 — Final verification + write-up

**Checkpoint:** full `.venv\Scripts\python.exe -m pytest -q` green AND `node --test tests/js/` green;
live preview smoke of all three visible changes; `docs/eval/2026-06-25-ux-polish.md` written;
MILESTONES appended + snapshot refreshed; DECISIONS current; teardown done.

- [x] Full suites green: **173 Python + 20 JS**. Live preview smoke (preview_eval): cosmic backdrop
  present (40+ stars, z-0), tiered bubbles high/med/low in correlation order (font 17.6 vs 14.6px,
  2-col packed), model hint live — all three in one pass.
- [x] Wrote `docs/eval/2026-06-25-ux-polish.md`.
- [x] Appended Cycle 3 to MILESTONES + refreshed snapshot (173 PY / 20 JS); teardown: no
  `.superpowers/sdd/*` scratch (inline run). `finishing-a-development-branch` → merge options presented.

## Blocked (filled in during the run)

-
