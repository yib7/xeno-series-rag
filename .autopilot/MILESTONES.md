# Milestones — Xeno Series Wiki RAG Chatbot

The project's durable accomplishment log. **Append-only:** unlike `PLAN.md` (which is reset or
archived at the end of each cycle), this file is never reset. It's how the project remembers what
it has shipped across every autopilot cycle — and what lets a new cycle safely reset `PLAN.md`
without losing history. Update it at the end of each cycle, right after
`finishing-a-development-branch`.

## Current state

A working local-first RAG chatbot over the full Xeno Series Wiki. The corpus is a **hybrid** build —
rendered HTML for the ~7,586 Lua-decoded stat pages + wikitext for the rest: **34,032 articles →
168,847 chunks** embedded with `bge-base-en-v1.5` into a persistent ChromaDB (cosine; embedded on a
Colab GPU, `chromadb==1.5.9`). Game tagging is **content-aware** (`derive_game` reads infobox/data
templates, category home-game, and cross-game cameo signals — not just title suffixes). Questions are
answered grounded + source-cited, with a **grounded-reasoning** prompt (count/total/infer, best-effort
partial answers) and **hybrid retrieval** (dense BGE + SQLite-FTS5 BM25, RRF-fused, cross-encoder
reranked, page-diversified), via a provider-agnostic LLM (live **Gemini `gemini-3.1-flash-lite`**, key
in gitignored `.env`; mock for tests) through a CLI and a FastAPI+SSE web UI with per-game theming
(real logos, key-art wash, game fonts) and a game filter. Retrieval filtering uses a
**multi-tag membership set** per page (`derive_games`): a page declares every base game it appears in
(KOS-MOS → {XS1,XS2,XS3,XC2}, Pyra → {XC2}; no signal → ubiquitous), stored as per-game `g_<game>`
metadata flags, so cross-appearance pages surface under exactly their games. The single `game` field
remains a display/breadcrumb label (base game / Xenosaga-umbrella `XS` / `series`); the per-game filter
is centralized in `parse_wikitext.filter_membership`. The web UI **streams tokens in real time**
(`answer_stream` -> text/sources/error SSE events), renders **safe Markdown including tables** (renderer
extracted to `static/render.js`, unit-tested under Node), and adds a **conversation thread with
multi-turn follow-ups** (stateless `history` + retrieval expansion), **clickable example questions**, a
**copy-answer** button, **aria-live** streaming, **rich source cards** (title/game/snippet), and
graceful in-pane errors. Retrieval keeps a stat page's **infobox** within the per-page cap and
**auto-merges** a fragmented stat page's factblocks into one full profile at answer time
(parent-document pattern, no re-embed). Python 3.12 venv; **165 Python tests + 17 JS tests** green.
Stack: requests, mwparserfromhell, beautifulsoup4/lxml, sentence-transformers, chromadb, google-genai,
fastapi; Node `--test` for the frontend renderer.

## Cycles (newest first)

### Cycle 2 — Polish / Finalize — 2026-06-23 (inline on `main`; not a git repo here)

App-layer finalization pass: bug-fix + test-coverage + interactive features. Design
`docs/superpowers/specs/2026-06-23-polish-finalize-design.md`; report
`docs/eval/2026-06-23-polish-finalize.md`. Corpus/index untouched (no re-embed/re-tag).

- **SP1** Fixed the **severe Markdown "numbers render as undefined" bug** (the stash-restore regex
  matched prose digits) and added the **missing frontend test harness** — renderer extracted to
  `static/render.js`, `tests/js/*.test.mjs` (`node --test`) wrapped by `tests/test_frontend_js.py`.
  Added GFM **table** rendering. This test gap is why the bug shipped; it's now closed.
- **SP2** Backend robustness: empty-question guard, `_extract_text` (safe LLM-response extraction, no
  more 500 on a blocked/thought-only response), non-stale default model.
- **SP3** **Real token streaming** — `generate_stream`/`answer_stream` emit `text`/`sources`/`error`
  events; `/ask` streams as the model produces; errors stream gracefully (no bare 500).
- **SP4** Prompt tuning — concise-Markdown + tables nudge; game-scope line on a filtered prompt.
- **SP5** Rich **source snippet previews** — `{url,title,game,snippet}` payload + chip-to-card UI.
- **SP6** **Conversation thread** UI — per-ask turn blocks, clickable **example questions**,
  **copy-answer**, **aria-live**, in-pane errors. CSS `#answer` -> `.answer`.
- **SP7** **Multi-turn follow-ups** — stateless `history` through `answer`/`answer_stream`/`build_prompt`
  + retrieval expansion; `AskRequest.history`; frontend history array + "New chat" reset.
- **SP8** Verified: **157 Python + 17 JS tests green**; inline app script `node --check` clean; full
  interactive browser smoke via `preview_eval` driving the real `ask()` with a stubbed `fetch`
  (numbers render, table renders, streaming across turns, sources+copy, follow-up sends history,
  New-chat resets) — **no billable LLM call** (live spend is an autonomy hard-stop, not re-authorized).
- Deferred / notes: a stale prior-session server still holds port 8000 with old code (restart on
  current code for the new features; `.claude/launch.json` preview now targets 8765). Live end-to-end
  with Gemini left for the user. Inline execution (no subagents), per the project's established pattern.

### Cycle 1 — Full Xeno-wiki RAG build — 2026-06-21 — branch `autopilot/xeno-rag-cycle1` → `main` @ 97fed74

- SP1 Scaffold — package `xeno_rag`, `config.yaml`, 3.12 venv, deps (3 tests).
- SP2 API client — MediaWiki etiquette: UA, `maxlag`, 429/`Retry-After`, backoff, serial+delay (6).
- SP3 Harvest — `allpages` ns=0 pagination → titles.jsonl (5).
- SP4 Fetch — batched, **resumable** content pull w/ checkpoint (4).
- SP5 Parse — wikitext → prose sections + structured infoboxes; game tag; url; drops (11).
- SP6 Chunk — section-aware prose + infobox-as-sentence, breadcrumbs, overlap (8).
- SP7 Embed/index — `bge-base` embedder (DirectML→CPU fallback) + ChromaDB; **resumable** (6).
- SP8 RAG — retrieve + game filter + grounded prompt + Gemini adapter (google-genai) + mock (5).
- SP9 Interface — CLI + FastAPI/SSE web UI with game selector (6).
- SP10 Full data run — harvested/fetched/parsed/chunked/embedded the entire wiki; verified live
  grounded Gemini answers end-to-end.
- SP11 — README (CC-BY-SA attribution, API-etiquette note, setup/run), pipeline driver.
- Deferred (→ BACKLOG): BM25/hybrid retrieval, cross-encoder reranker, incremental refresh, eval
  set, build-generator data sharing, GPU embedding (DirectML blocked by onnxruntime conflict; CPU used).
- cycle1 scratch swept — none to sweep (inline execution, no `.superpowers/sdd/` artifacts).

### Post-cycle-1 quality stream — 2026-06-21 → 2026-06-23 (on `main`, see DECISIONS for each)

Iterative quality work after the cycle-1 build, driven by live testing:
- **Rendered-HTML stat rebuild** — the wiki's Lua-decoded stat tables exist only in rendered HTML;
  added `fetch_html`/`parse_html` + `run_hybrid` merge. Corpus → 34,032 articles / 168,847 chunks.
- **Content-aware game tagging** — `derive_game` reads page content (infobox/data templates,
  categories), dropping the catch-all `series` share from ~93% → low single digits; the per-game
  filter is now precise.
- **Colab GPU embedding** — offloaded the ~6 h CPU embed to a Colab GPU (`chromadb==1.5.9` pinned for
  cross-machine portability); see [[colab-gpu-embedding-workflow]].
- **Full per-game branding + grounded-reasoning prompt + page-diversified retrieval.**
- **40-question evaluation + cross-game cameo tagging fix** (2026-06-23) — ran 5 Q × 8 games with
  filters on (`eval/`, report `docs/eval/2026-06-23-rag-evaluation.md`). 0 errors, no hard filter
  leaks. Found + fixed the highest-impact bug: characters who *cameo* in another subseries (KOS-MOS,
  T-elos, Shion — rare XC2 Blades) were tagged the cameo game and hidden from their home filter.
  `derive_game` now takes the home (first) category for entity pages, with pan-Xenosaga leads → series;
  re-tagged **357 chunks in place, no re-embed** (reversible snapshot).
- **Hybrid retrieval + cross-encoder reranker** (2026-06-23) — fixed the eval's remaining recall miss
  ("mimeosomes" returned only weapon SKUs). Lexical **BM25** (SQLite FTS5, `bm25_index.py`, built from
  the collection) fused with dense via **RRF** (`retrieve.py`), then a `ms-marco-MiniLM-L-6-v2`
  **reranker** (`rerank.py`); `rag.answer` routes through `retrieve.retrieve`; toggleable
  (`use_bm25`/`use_reranker`); new `bm25` pipeline step. +15 tests (122 green); BM25 built over all
  168,847 chunks; mimeosomes now answered correctly, full-40 hybrid re-run shows 0 regressions.
- **Round-2 niche/complex evaluation + Xenosaga-umbrella `XS` tag** (2026-06-23) — a *harder* 40
  (minor characters, deep lore, aggregation, niche mechanics; report
  `docs/eval/2026-06-23-rag-evaluation-round2.md`). **0 errors, 0 hard leaks, 36/40 ✅, 4 honest
  coverage-gap partials, 0 wrong/hallucinated;** the cross-game **Jin** name-collision stressor passed
  and aggregation worked (13 Zohar Emulators; 8 BLADE divisions). Found + fixed one real defect:
  Xenosaga-wide pages (`Ether (XS)`, `Ether Amp (XS1&2)`) tagged the catch-all `series` leaked into
  Xenogears/Xenoblade filters. Split `series` into a Xenosaga-umbrella **`XS`** (`derive_game` +
  shared `filter_tags`); re-tagged **1,032 chunks / 193 pages** in place (reversible), rebuilt BM25.
  +8 tests (130 green); verified via retrieval + live `/ask` (XG "Ether" now 10/10 XG sources), zero
  regressions. The other 3 partials are inherent corpus-coverage gaps.
- **Multi-tag membership schema** (2026-06-23) — finished the cross-appearance case the `XS` umbrella
  couldn't (a page in 2+ games AND another franchise had to be the all-franchises `series`, shown
  everywhere). `derive_games` now gives each page the SET of base games it appears in (from its
  categories/templates/suffix); filtering matches per-game `g_<game>` membership flags
  (`filter_membership`, shared by dense `_where` + BM25). `game` stays a display label (no re-embed).
  +5 net tests (135 green); re-tagged all 168,847 chunks in place (metadata-only, reversible), BM25
  rebuilt. **Verified:** KOS-MOS/T-elos/Elma/Shulk/Shion each retrievable under EXACTLY their games
  (KOS-MOS → XS1/XS2/XS3/XC2, absent from XG/XC1/XC3). Residual (→ backlog): auxiliary Xeno-*media*
  pages (anime/spinoff/album) lack a mainline-game category → ubiquitous (pre-existing, not a regression).

<!-- prepend each new cycle above this line -->
