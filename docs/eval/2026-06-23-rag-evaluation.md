# Xeno RAG — 40-Question Evaluation & Fixes (2026-06-23)

A full-coverage evaluation of the chatbot: **5 questions × 8 games = 40**, each run with that
game's per-game filter **ON**, against the live index (168,847 chunks) using
`gemini-3.1-flash-lite`. Goal: find where answers are wrong, where one game's data leaks into
another under the filter, and where coverage is thin — then fix the highest-impact issues.

Harness + raw data are reproducible in `eval/`:
`run_eval.py` (the 40 Q&A sweep → `results.json`), `analyze.py` (compact dump + tag/title-mismatch
detector → `analysis.txt`), `diag.py` / `inspect_tags.py` (root-cause evidence), `retag.py` (the
metadata re-tag with reversibility snapshot), `verify.py` (post-fix spot checks). `results_before.json`
is the pre-fix snapshot.

## Headline results

- **40/40 ran, 0 API errors.**
- **No hard filter leaks:** with a filter on game *G*, **zero** retrieved chunks were tagged a
  *different base game*. The `where = {"game": {"$in": [G, "series"]}}` filter is mechanically sound.
- **Answer quality before fixes:** ~30/40 strong & correct, ~6 partial (mostly honest "limited
  info" on Xenosaga II/III mechanics), **3 poor/failed** (KOS-MOS, T-elos, mimeosomes).
- **The 3 failures had one shared root cause for 2 of them** — a **tagging bug** that hid the
  canonical page from its own game's filter. That bug is now **fixed and verified**. The third
  (mimeosomes) is a distinct **retrieval-recall** bug, documented for a follow-up.
- **After the fix** (full 40 re-run, `eval/compare.py`): **14/40 questions changed, all improvements
  or neutral, zero regressions.** KOS-MOS / T-elos / Shion / Zohar(XS) now retrieve their canonical
  page and answer correctly; the Xenosaga leads that a first (over-broad) fix attempt had regressed
  (Jr., Albedo, Jin, Wilhelm, chaos) are fully restored; the 25 Xenoblade/Xenogears answers are
  unchanged. 107 tests green. *(Fixing this took two iterations — the first re-tag pinned recurring
  Xenosaga leads to their debut episode and hid them from the other two episodes' filters; the
  re-run's before/after diff caught it, and the refined rule below — pan-Xenosaga → `series` — fixed
  it. This is why the full re-run mattered.)*

## Per-question grades (pre-fix)

Grade key: ✅ correct/strong · ◑ partial (honest about gaps) · ✗ poor/failed · → fixed this session.

| # | Game | Question | Grade | Note |
|---|------|----------|-------|------|
| 1 | XG | Main protagonist | ✅ | Fei Fong Wong; Kim preincarnation noted |
| 2 | XG | Deathblows | ✅ | mechanics + learning correct |
| 3 | XG | Gears | ✅ | thorough |
| 4 | XG | The Zohar | ✅ | excellent cross-series; *(latent: `Zohar (XS)` chunk was mis-tagged XG — fixed)* |
| 5 | XG | Citan Uzuki | ✅ | identity + Hyuga + Xenosaga links |
| 6 | XS1 | KOS-MOS | ✗ → ✅ | **was:** music/episode scraps, no character info. **now:** full "anti-Gnosis Vector android" |
| 7 | XS1 | Shion Uzuki | ✅ | correct |
| 8 | XS1 | Gnosis | ✅ | excellent |
| 9 | XS1 | Boost system | ◑ | retrieval pulled "Boost Quiz" minigame; core Boost-gauge mechanic not on a clean page |
| 10 | XS1 | Zohar in Xenosaga | ◑ → ✅ | **was:** missed canonical page, Emulators only. **now:** `Zohar (XS)` retrieved, "Original Zohar" defined |
| 11 | XS2 | Battle system (XS2) | ◑ | honest "limited info"; no overview page, scattered across boss/music |
| 12 | XS2 | Gaignun Kukai Jr. | ✅ | excellent |
| 13 | XS2 | Albedo | ✅ | excellent |
| 14 | XS2 | E.S. units | ✅ | reasoned "at least 12 units" |
| 15 | XS2 | Jin Uzuki | ✅ | excellent |
| 16 | XS3 | T-elos | ✗ → ✅ | **was:** boss/music data only, "no biography". **now:** "built by Roth Mantel to replace KOS-MOS" |
| 17 | XS3 | Wilhelm | ✅ | excellent (incl. Z parallel) |
| 18 | XS3 | Battle system (XS3) | ◑ | honest; no overview page |
| 19 | XS3 | Zohar role in XS3 | ◑ | honest; aggregation-specific question, no single source |
| 20 | XS3 | chaos | ✅ | correct |
| 21 | XC1 | Shulk | ✅ | excellent |
| 22 | XC1 | Monado | ✅ | excellent |
| 23 | XC1 | Arts | ✅ | excellent |
| 24 | XC1 | Metal Face | ✅ | identity (Mumkhar) + encounters |
| 25 | XC1 | Chain Attack | ✅ | correct |
| 26 | XC2 | Rex | ✅ | excellent |
| 27 | XC2 | Blades & Drivers | ✅ | excellent |
| 28 | XC2 | Driver Combos | ✅ | Break→Topple→Launch→Smash + Fusion |
| 29 | XC2 | Pyra | ✅ | excellent |
| 30 | XC2 | Elysium | ✅ | dream vs real, correct |
| 31 | XC3 | Noah | ✅ | excellent |
| 32 | XC3 | Interlinking/Ouroboros | ✅ | excellent; correctly *separated* a Xenosaga "Interlink" cross-ref into its own section |
| 33 | XC3 | Keves & Agnus | ✅ | excellent |
| 34 | XC3 | Flame clock | ✅ | excellent |
| 35 | XC3 | N | ✅ | correct despite single-letter query noise |
| 36 | XCX | New Los Angeles | ✅ | excellent |
| 37 | XCX | Skells | ◑ | correct but shallow; enemy-Skell instances crowded the concept page |
| 38 | XCX | mimeosomes | ✗ | **retrieval-recall bug** — all 10 chunks were Skell weapon SKUs; canonical page never retrieved |
| 39 | XCX | Soul Voices | ✅ | excellent |
| 40 | XCX | Elma | ✅ | excellent |

## Issues found, by root cause

### Issue 1 — Cross-game pages mis-tagged → hidden from their own filter  *(FIXED)*

**The big one.** When a character/concept originates in game A but *cameos* in game B, `derive_game`
could tag the whole page **B**. Because the per-game filter is `{"game": {"$in": [G, "series"]}}`, a
page mis-tagged to the *wrong base game* is **silently excluded** from its correct game's filter —
removing exactly the page the question needs.

Hard evidence (index lookups, `diag.py`):

| Page | Was tagged | Correct | Effect |
|------|-----------|---------|--------|
| `KOS-MOS` | **XC2** (rare XC2 Blade cameo) | XS1 | hidden from XS1 → Q6 got only music/episode pages |
| `T-elos` | **XC2** (cameo) | XS3 | hidden from XS3 → Q16 got only boss/music pages |
| `Zohar (XS)` | **XG** (lone `{{XG}}` cross-ref) | series | hidden from all XS filters → Q10 missed it |
| `Shion` | **XC2** (cameo) | XS1 | (same class) |

**Mechanism (`parse_wikitext.derive_game`):** these character pages use a game-*agnostic*
`{{Infobox character}}`, so the "own structured template" step found no game. The code then fell to
"any lone game-prefixed template," where the only *recognized* code was the foreign cameo's
`{{XC2}}`/`{{XG}}` link shortcut — Xenosaga's own `{{XS}}` shortcut isn't a base-game code, so it was
invisible. Category order (which lists the **home game first**) was never consulted.

**Fix (`derive_game`):** for entity (infobox) pages, take the **first game-bearing category as the
home game**; keep infobox-less multi-game *lore* pages as `series`; recognize the generic `{{XS}}` /
`ArticleIcon/XS…` marker (`_is_xeno_generic`) so a lone foreign cross-ref can't hijack a Xenosaga
page. One **subseries-aware** refinement: the Xenosaga trilogy shares one continuous cast, so a
character in **2+ Xenosaga episodes → `series`** (visible under XS1/XS2/XS3); Xenoblade games have
distinct casts, so a Xenoblade character keeps her single home game (no leakage into the others). TDD:
5 new/updated tests, all 107 green. Re-tagged the live index in place (**357 chunks across 77 pages,
metadata-only — no re-embed**, reversible snapshot in `eval/retag_snapshot.jsonl`). Verified by a full
40-question re-run: KOS-MOS/T-elos/Zohar(XS)/Shion fixed, recurring Xenosaga leads retained,
Shulk/Rex/Elma + the 25 Xenoblade answers unchanged — **zero regressions**.

### Issue 2 — Retrieval recall: canonical page ranks outside the fetch window  *(FIXED)*

`mimeosomes` (Q38) is tagged **XCX correctly**, yet retrieval returned only Skell weapon model
pages (`SKM-M230ME Claymore`, …) and never the 3-chunk `Mimeosome` page. The many near-duplicate
weapon-SKU chunks out-rank the concept page within the over-fetch window. This is a pure recall
problem (not tagging) and the re-tag — as predicted — did not change it. Same family as the slightly
shallow Skells (Q37) and the "Boost Quiz vs Boost mechanic" mix-up (Q9): **exact proper-noun /
concept queries lose to title-matching ancillary pages.**

**Fix (shipped):** **hybrid retrieval** — a lexical **BM25** index (SQLite FTS5, built from the
collection so its game tags match) fused with the dense BGE candidates via **Reciprocal Rank Fusion**,
then a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) reorders the fused set; the per-page cap
still applies last. `rag.answer` now routes through `retrieve.retrieve`; toggle with
`use_bm25`/`use_reranker`. +15 tests, BM25 index built over all 168,847 chunks. **Result:** mimeosomes
now returns the `Mimeosome` page **#1** — *"the mechanical bodies used by the humans of Xenoblade
Chronicles X… 'mims'"* (was a flat refusal); the full-40 hybrid re-run removed the refusal from 7
questions with **zero regressions**, and the live `/ask` endpoint was confirmed end-to-end. Retrieval
latency ~1.7 s/query after a one-time reranker load. BM25 nails exact terms like "mimeosome"; the
reranker promotes the true subject page above ancillary music/SKU/episode look-alikes.

### Issue 3 — Coverage gaps for Xenosaga II/III mechanics  *(inherent; handled well)*

"Battle system in XS2/XS3" (Q11, Q18) and "Zohar's role in XS3" (Q19) have no single overview page;
the facts are scattered across boss/music/tutorial pages. The grounded-reasoning prompt handled these
**correctly** — it gave the best-supported partial answer and stated what was missing, rather than
hallucinating. Not a bug; a corpus-coverage limit.

### Issue 4 — Minor noise *(low priority)*

Single-letter / short proper-noun queries ("N", Q35) pull incidental title matches ("Nintendo
Network", "Neon"); the model still answered correctly. Cosmetic.

## Residual / backlog (not fixed this session)

- ~~Hybrid BM25 + dense retrieval / cross-encoder reranker~~ — **DONE** (see Issue 2 fix above).
- **Pan-installment Xenosaga characters tagged by debut game.** KOS-MOS/Shion are now `XS1` (their
  debut/home), so they surface under XS1 but **not** XS2/XS3 filters (they were hidden from *all*
  Xenosaga filters before, so this is strictly better — but not perfect). A multi-tag schema, or
  tagging trilogy-spanning characters `series`, would let them appear under every Xenosaga filter.
- **No combat-system overview pages for XS2/XS3** — corpus gap, not a code issue.
