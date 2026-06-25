# Xeno RAG — Round 2: Niche / Complex Stress-Test Evaluation & Fix (2026-06-23)

A second full-coverage evaluation, deliberately **harder** than round 1
([2026-06-23-rag-evaluation.md](2026-06-23-rag-evaluation.md)). Round 1 asked mainstream "Who is
<lead>? / What is <core mechanic>?" questions. Round 2 stress-tests the corpus and the hybrid
retriever with **5 niche/complex questions × 8 games = 40**, each run with that game's per-game
filter **ON**, against the live index (168,847 chunks) using `gemini-3.1-flash-lite`.

The round-2 set targets exactly the places a RAG pipeline tends to break:
- **minor characters / NPCs / bosses** — Emeralda, Canaan, Dr. Sellers, Voyager, Dunban, Tatsu;
- **deep multi-entity lore** — Solaris caste society, the Testaments, the Miltian Conflict, the
  High Entia secret, the Fei/Id/Grahf identity, U-DO/chaos/Wilhelm;
- **aggregation / numeric reasoning** — *how many* Zohar Emulators; the BLADE *divisions*;
- **niche mechanics** — Ether, Gem Crafting, field skills + the Affinity Chart, Overdrive, class-change;
- **a deliberate cross-game leakage stressor** — XC2's **Jin** (Flesh Eater, leader of Torna) shares a
  first name with Xenosaga's **Jin Uzuki**; with the XC2 filter on, no Xenosaga Jin chunk may leak.

Harness + raw data in `eval/`: `run_eval_round2.py` (the 40-question sweep → `results_round2.json`),
`analyze_round2.py` (graded dump → `analysis_round2.txt`), `diag_round2.py` (root-cause evidence),
`inspect_round2_retag.py` (re-tag blast-radius listing), `verify_round2.py` (post-fix retrieval
checks), `rerun_round2_spot.py` (post-fix live re-run).

## Headline results

- **40/40 ran, 0 API errors.**
- **0 hard filter leaks:** with a filter on game *G*, **zero** retrieved chunks were tagged a
  *different base game*. The **Jin stressor (#27) passed perfectly** — all 10 chunks `Jin (XC2)`,
  zero `Jin Uzuki` bleed.
- **Aggregation works:** #13 reasoned to **"at least 13" Zohar Emulators** (12 Mizrahi + Sellers'
  13th, with the distinguishing details); #36 enumerated **all 8 BLADE divisions**.
- **Answer quality: 36/40 strong & correct (✅), 4 honest partials (◑), 0 wrong, 0 hallucinations.**
- **One real defect found + fixed this session:** Xenosaga pages the wiki author explicitly labelled
  Xenosaga-wide (`Ether (XS)`, `Ether Amp (XS1&2)`, …) were tagged the all-franchises `series`, so
  they surfaced under **Xenogears and Xenoblade** filters. Introduced a Xenosaga-umbrella **`XS`**
  tag; re-tagged **1,032 chunks across 193 pages** in place (metadata-only, reversible). The leak is
  closed and verified; the three other partials are inherent **corpus-coverage** gaps, not bugs.

## Per-question grades

Grade key: ✅ correct/strong · ◑ honest partial (coverage gap, no hallucination) · ✗ poor/failed.

| # | Game | Question | Grade | Note |
|---|------|----------|-------|------|
| 1 | XG | Emeralda (creation) | ✅ | nanomachine colony; Kim/Zeboim; sealed ~4000 yrs; Ethos Dig Site |
| 2 | XG | Solaris caste society | ◑ | totalitarian structure + Soylent correct; **caste tiers not on any page** (scattered) |
| 3 | XG | Fei / Id / Grahf | ✅ | Id = alt personality; Grahf possessed Khan; excellent multi-entity synthesis |
| 4 | XG | Ether system | ◑ | **precision fixed** (Xenosaga ether no longer leaks); XG ether has no concept page → thin |
| 5 | XG | Yggdrasil evolution | ✅ | sub → desert → naval → airship; Yggdrasil IV distinguished |
| 6 | XS1 | U.M.N. | ✅ | FTL net on Collective Unconscious; Scientia/Veritas; Voyager hack |
| 7 | XS1 | Margulis / U-TIC | ✅ | Chief Inquisitor of Ormus; E.S. Levi; Pellegri |
| 8 | XS1 | Encephalon | ✅ | VR via U.M.N.; Old Miltia dive; injury feedback |
| 9 | XS1 | Woglinde fate | ✅ | Federation cruiser; Gnosis attack; Plan 31 frame-up |
| 10 | XS1 | Learn Tech/Ether | ◑ | Ether tree correct; exact Tech-Attack learning under-specified (honest) |
| 11 | XS2 | Miltian Conflict | ✅ | Zoar Incident → Hierocracy→Republic → U-TIC coup; T.C. 4753 |
| 12 | XS2 | Canaan | ✅ | Realian; Program Canaan; E.S. Asher w/ chaos; saved Jr./Gaignun |
| 13 | XS2 | Zohar Emulators (count) | ✅ | **aggregation: "at least 13" = 12 Mizrahi + Sellers' silver unit** |
| 14 | XS2 | Dr. Sellers | ✅ | crippled by Mizrahi; Hyams; Dr. Strangelove / Peter Sellers homage |
| 15 | XS2 | Stock & Break system | ◑ | honest refusal — **no XS2 combat-overview page** (round-1 Issue 3; term mismatch) |
| 16 | XS3 | Testaments | ✅ | Wilhelm's revived agents; Voyager/Virgil/Kevin/Albedo; Consul parallel |
| 17 | XS3 | Voyager | ✅ | Erich Weber, Black Testament, Net Preacher; Bugs; Revelation 16 |
| 18 | XS3 | Zarathustra | ✅ | Eternal Recurrence device; Michtam; final boss 70k HP, 6-gauge Boost |
| 19 | XS3 | Abel & Abel's Ark | ✅ | Ω Res Novae pilot; star-system-sized U-DO observation terminal |
| 20 | XS3 | U-DO / chaos / Wilhelm | ✅ | higher-dim "God"; seal U-DO's eyes for Eternal Recurrence; honest hedge |
| 21 | XC1 | Dunban | ✅ | Fiora's brother; Sword Valley hero; arm burn; dodge tank |
| 22 | XC1 | High Entia secret | ✅ | Telethia origin/transformation; Homs-crossing to dilute |
| 23 | XC1 | Egil | ✅ | Machina; Yaldabaoth; destroy Bionis → opposes Zanza |
| 24 | XC1 | Gem Crafting | ✅ | Shooter/Engineer; flame settings; cylinders; Mobile Furnace |
| 25 | XC1 | Bionis / Mechonis | ✅ | two dead titans; Sword Valley; Fallen Arm; Zanza/Meyneth |
| 26 | XC2 | Titans (life on them) | ✅ | Cloud Sea; habitat/transport/war; Core Crystal lifecycle; dying off |
| 27 | XC2 | **Jin (leakage stressor)** | ✅ | **all chunks `Jin (XC2)`; zero Jin Uzuki leak**; destroy Alrest / kill Architect |
| 28 | XC2 | Core Crystals / awakening | ✅ | resonance; Driver eligibility; revert-on-death cycle |
| 29 | XC2 | Kingdom of Torna fate | ✅ | Malos core explosion; sank to Morytha; king went down |
| 30 | XC2 | Field skills / Affinity Chart | ✅ | very detailed + accurate (node unlocks, Merc-Mission timing) |
| 31 | XC3 | Moebius / Consuls | ✅ | same entity; "endless now"; Flame Clock feeding; cores |
| 32 | XC3 | The City (founders) | ✅ | home of Lost Numbers; Founders = first seven Ouroboros |
| 33 | XC3 | Mio / Ouroboros | ✅ | Agnus off-seer; Defender form; interlinks with Noah |
| 34 | XC3 | Origin | ✅ | Bionis+Alrest soul repository; Ontos core; hijacked by Moebius |
| 35 | XC3 | Class / class-change | ✅ | Heroes; CP vs Class Succession Points; mastery |
| 36 | XCX | BLADE divisions | ✅ | **aggregation: all 8 divisions named** (Pathfinders…Mediators) |
| 37 | XCX | Ganglion | ✅ | antagonist coalition; Luxaar/Void; Puges/Xerns; Prone/Marnuck |
| 38 | XCX | Lifehold Core | ✅ | powers mimeosomes; Ch.12 reveal (Database Chamber flooded) |
| 39 | XCX | Overdrive system | ✅ | very deep mechanics (TP cost, count, duration, Skell Cockpit Time) |
| 40 | XCX | Tatsu | ✅ | Nopon; Dodonga Caravan; Koko/siblings; Tora rival |

## Issue found + fixed — Xenosaga-wide pages leaked into non-Xenosaga filters *(FIXED)*

**Symptom (#4):** "How does the Ether system work in **Xenogears**?" retrieved two **Xenosaga**
pages, `Ether (XS)` and `Ether Amp (XS1&2)`.

**Root cause (`diag_round2.py`):** both pages are tagged **`series`**, and there is **no `Ether (XG)`
page at all** (Xenogears ether is not a documented concept page). Because the per-game filter is
`{"game": {"$in": [G, "series"]}}`, the `series` bucket is a **catch-all visible to *every* game's
filter** — so a page the wiki author explicitly suffixed `(XS)` / `(XS1&2)` (unambiguously Xenosaga)
was admitted under Xenogears and all three Xenoblade filters. This is the same root limitation noted
in round-1's residual ("pan-Xenosaga pages tagged `series` leak mildly into XG/XC").

**Fix — a Xenosaga-umbrella `XS` tag.** `series` conflated two different things: *genuinely
cross-franchise* pages (thematic parallels, the shared Zohar overview) and *Xenosaga-wide* pages
(recurring leads, cross-episode concept/enemy/attack pages). We split the latter out:

1. **`derive_game` → `"XS"`** (instead of `series`) for a page that is **Xenosaga-only**: an explicit
   Xenosaga-wide title suffix (`(XS)`, `(XS1&2)`, …), or whose game categories are a **subset of the
   Xenosaga trilogy** (2+ episodes, no other franchise). A page that *also* carries a non-Xenosaga
   cameo category stays `series` — which is why `KOS-MOS` (a real XC2 Blade) correctly **stayed
   `series`** and is still visible under the XC2 filter.
2. **`filter_tags(game_filter)`** (new, in `parse_wikitext`, shared by the dense `_where` and the
   BM25 `search`): a **Xenosaga** episode admits `[game, "XS", "series"]`; a **non-Xenosaga** game
   admits only `[game, "series"]`. So `XS` pages appear under every Xenosaga filter but never under
   Xenogears/Xenoblade.
3. **Re-tagged the live index in place:** 1,032 chunks across 193 pages `series→XS` (+3 `XS1→XS` for a
   `(XS)`-suffixed page), **metadata-only, no re-embed**, reversible snapshot in
   `eval/retag_snapshot_round2.jsonl`. BM25 rebuilt from the re-tagged collection.

**TDD:** 8 new/updated tests (5 `derive_game` `XS` cases + 3 `filter_tags`/`_where` cases); full
suite **130 green**.

**Verified (`verify_round2.py`, `rerun_round2_spot.py`, live `/ask`):**
- XG "Ether" now returns **only Xenogears pages** — `Ether (XS)`/`Ether Amp (XS1&2)` gone; freed
  slots filled with on-topic XG pages (`Contact`, `Xenogears (Gear)`). Live `/ask` sources: 10/10 XG.
- XS1 "Ether" **still** returns `Ether (XS)` + `Ether Amp (XS1&2)` (now `XS`, admitted by the filter).
- **No `XS` leak** into XG / XC1 / XC2 / XCX across probe queries.
- **No regressions:** #13 (Zohar Emulators "at least 13"), #16 (Testaments), and round-1 #6 (KOS-MOS,
  who notes her XC2 cameo) all answer as well as before.

## Other partials — inherent coverage gaps (not bugs)

- **#2 Solaris caste society.** The wiki has **no consolidated Solaris-caste page**; the tier facts
  are thinly scattered (`Gebler Special Forces`: *"Solaris third class citizens"*; `Timothy`: *"third
  class worker bee"*) and *are* correctly tagged XG. The grounded answer described the totalitarian
  structure + Soylent system and was honest that it could not find explicit caste tiers. Same family
  as round-1 Issue 3 (scattered facts, no overview page). Fixing needs corpus work, not code.
- **#15 XS2 Stock & Break system.** As in round 1 (Q11/Q18), Xenosaga II/III have **no combat-overview
  page**; the model gave the best-supported partial and stated what was missing rather than
  hallucinating. (Question wording also used non-wiki terminology.)
- **#10 XS1 Tech/Ether learning.** Ether-tree progression correct; exact Tech-Attack acquisition
  under-specified — honest partial.

## Follow-up — multi-tag membership schema (completes the cross-appearance case)

The `XS` umbrella fixed Xenosaga-wide pages leaking into XG/Xenoblade, but one class remained: a page
in **2+ games across franchises** (e.g. **KOS-MOS** — a Xenosaga lead *and* an XC2 Blade) still had to
collapse to the single all-franchises `series` tag, so it showed under *every* filter. A single `game`
string can't say "this page belongs to XS1, XS2, XS3 **and** XC2."

**Fix — multi-tag membership.** Each page now declares the SET of base games it appears in
(`derive_games`, from the union of its categories/templates/suffix — evidence in
`eval/inspect_membership.py` confirmed categories are the true membership: KOS-MOS {XS1,XS2,XS3,XC2},
Elma {XCX,XC2}, Shulk {XC1,XC2,XC3}, Pyra {XC2}). Stored as per-game `g_<game>` boolean flags (ChromaDB
metadata can't hold a list) + a BM25 `games` column; the filter (`filter_membership`) matches a page
that belongs to the selected game. The single `game` field stays a display/breadcrumb label (untouched
— it's in the embedded text). **+5 net tests (135 green); re-tagged all 168,847 chunks in place
(metadata-only, reversible); BM25 rebuilt.** Verified (`eval/verify_membership.py` + live `/ask`): each
cross-appearance page is retrievable under **exactly** its games and excluded elsewhere — e.g. the
KOS-MOS bio is now present under XS1/XS2/XS3/XC2 and **absent from XG/XC1/XC3**.

*Residual (→ backlog):* auxiliary Xeno-*media* pages (anime `Music (XSTA)`, spinoffs `Xeno-pittan` /
`Music (XSF)`, album codes) carry no mainline-game category, so they default to ubiquitous (shown under
every filter) — **pre-existing** (they were `series` before; not a regression). A media sub-franchise
code map would scope them.

## Conclusion

On a deliberately harder, niche set the chatbot is **robust**: 0 wrong answers, 0 hallucinations, 0
hard leaks, correct aggregation, and the cross-game name-collision stressor (Jin) cleanly isolated.
The one true defect — Xenosaga-wide pages leaking into Xenogears/Xenoblade via the `series` catch-all
— is fixed with the `XS` umbrella tag and verified end-to-end. Remaining soft spots are corpus
coverage gaps, handled honestly by the grounded-reasoning prompt.
