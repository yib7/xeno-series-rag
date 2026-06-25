# Backlog — parked ideas

New ideas raised mid-run land here so the frozen scope keeps converging. Brainstorm these as the
*next* autopilot cycle, not this one.

## Inbox (unsorted ideas)
- **Denser stat-page chunking (needs re-embed).** Root cause of "stat pages feel messy in retrieval":
  each Lua factblock (every drop sub-table, each resistance/AI/identity block) is its own ~100-char
  chunk, so a single enemy is shattered into ~19 fragments (Rotbart) — up to 140 on the biggest pages.
  The per-page cap then only surfaces 3-4 of them, and one enemy's facets compete across many tiny
  chunks. Fix: at chunk-time, merge a stat page's factblocks into a few denser, self-contained chunks
  (e.g. one "combat/stats" chunk, one "drops" chunk, keep the infobox separate) so each retrieved chunk
  carries a fuller picture and the cap isn't fighting fragmentation. Changes chunk *text* -> requires a
  full re-embed (~2h CPU or a Colab GPU run, `chromadb==1.5.9`). **Option A shipped instead** (2026-06-24
  answer-time auto-merging, `merge_fragmented_pages`, no re-embed — see DECISIONS) and largely fixes the
  *generation* side (full profile reaches the LLM). This re-chunk (Option B) is the complementary part:
  it also fixes the *retrieval* side — denser chunks embed far better than 30-tok scraps (the weak
  embedding is why the Rotbart location ranked only 5th). Medium priority; do if retrieval ranking of
  stat facts still disappoints after auto-merging.
- **Extend HTML rebuild to the remaining ~28k pages.** Current rebuild fetches rendered HTML only for
  the ~7,593 stat/data-table pages (where wikitext loses the decoded values); the rest keep their
  wikitext-parsed prose. Fetching HTML for all 36k would also resolve the minor `{{gp}}`/`{{XCn}}`
  prose link gaps on table-less pages — marginal benefit for ~40h more fetching at 5s. Do only if the
  prose polish proves worth it.
- **HTML table parser edge cases:** the Affinity-Chart grid renders verbose concatenated lines; some
  2-column header tables (EXP/AP) mislabel the row name; multi-tier headers flatten imperfectly. The
  important facts (element, stats, weapon, resistances, prices) extract cleanly — these are polish.
- ~~Stat-block field filtering~~ — **superseded** by the HTML rebuild: stats now come from the wiki's
  rendered, curated tables (decoded + labelled) instead of the raw `{{… data}}` code dumps.
- ~~**Sharper game tagging**~~ / ~~**multi-tag schema**~~ — **DONE** (2026-06-22 content-aware
  `derive_game`; 2026-06-23 cross-game cameo fix, `XS` umbrella, and the **multi-tag membership
  schema**, see Done). Cross-appearance pages now filter to exactly their games (KOS-MOS →
  XS1/XS2/XS3/XC2). **Remaining nuance — auxiliary Xeno-*media* tagging:** anime / spinoff / album
  pages use sub-franchise category codes outside the 8 mainline games (`Music (XSTA)` = The Animation,
  `Music (XSF)` = Freaks, `Xeno-pittan levels`, album codes like `AMY`), so `derive_games` finds no
  mainline game → they default to **ubiquitous** (shown under every filter; same as the old `series`,
  not a regression). A small alias map (clearly-Xenosaga media → `{XS1,XS2,XS3}`, etc.) would scope
  them — but album/spinoff codes need research to map correctly, so it's its own task. Low priority
  (these pages only surface noise on questions about non-mainline media subjects).
- **Thin corpus coverage for a few aggregation-style topics** (surfaced by round-2 eval, not bugs —
  the grounded prompt handled them honestly): no consolidated **Solaris caste/class** page (tier facts
  scattered across `Gebler Special Forces`/`Timothy`); no **XS2/XS3 combat-overview** page; no
  **`Ether (XG)`** concept page. Would need targeted re-harvest or hand-authored summary chunks, not
  code. Low priority.
- **Real token streaming:** stream Gemini output via `generate_content_stream` instead of generating the
  full answer then fake-streaming slices — lower time-to-first-token for the "Thinking" model.
- ~~**Hybrid retrieval: dense + BM25**~~ and ~~**cross-encoder reranker**~~ — **DONE (2026-06-23)**,
  see Done below. (Lexical BM25 via SQLite FTS5 + RRF fusion + `ms-marco-MiniLM-L-6-v2` reranker.)
- Periodic refresh: re-pull only pages whose revision timestamp changed (`rvprop=timestamp`).
- Share Phase-4 structured infobox output with the planned multi-game build generator.
- ONNX + DirectML embedding backend to use the AMD GPU instead of CPU.
- Thin web UI (FastAPI + SSE stream + game selector).

## Next cycle (promoted, ready to brainstorm)
-

## Done / shipped
- **Hybrid retrieval (dense + BM25) + cross-encoder reranker** (2026-06-23): SQLite-FTS5 BM25 index
  (`bm25_index.py`, built from the live collection) fused with dense via Reciprocal Rank Fusion
  (`retrieve.py`), then reordered by `cross-encoder/ms-marco-MiniLM-L-6-v2` (`rerank.py`). Toggleable
  (`use_bm25`/`use_reranker`), new `bm25` pipeline step. Fixes exact proper-noun / concept recall —
  "mimeosomes" now returns the canonical page #1 (was a flat refusal). +15 tests, built over 168,847
  chunks, full-40 re-run shows 0 regressions.
- **40-question evaluation set + cross-game tagging fix** (2026-06-23): 5 Q × 8 games run with filters
  on (`eval/`, report `docs/eval/2026-06-23-rag-evaluation.md`). Found + fixed the highest-impact bug —
  cross-subseries cameo characters (KOS-MOS/T-elos/Shion, rare XC2 Blades) were tagged the *cameo* game
  and hidden from their home filter. `derive_game` now uses the first home category for entity pages
  (pan-Xenosaga leads → `series`); re-tagged 357 chunks in place, no re-embed. Supersedes the old
  "Evaluation set" and most of "Sharper game tagging" inbox items.
- **Full game branding** (2026-06-21): all 8 games now ship a real logo + key-art wash + a game-matched
  display font (`scripts/optimize_art.py`, per-game `--font-display`). Supersedes "Remaining game logos".
