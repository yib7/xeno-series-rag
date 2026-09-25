# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html): the code version tracks the application,
and the corpus/vector-store release assets are tagged separately (`data-v1`, `data-v2`).

## [Unreleased]

No data changes: retrieval depths per tier are unchanged from the old answer styles, so the shipped
`data-v2` vector store and retrieval behavior are unaffected.

### Added
- Jev (TypeSafe AI) auto-routing: each question is sent through `xeno_rag/router.py`, which asks
  Jev's "System One" decision model to pick an answer tier (fast, thinking, or scholar) instead of
  showing a selector. Routing plus the coverage check together cost about $0.0002 per on-topic
  question (measured by the gate eval), and routing falls back to a fixed tier (default `thinking`)
  with no network call when `TYPESAFE_API_KEY` is unset, `router.provider` is `fixed`, or the call
  fails or returns a low-confidence choice.
- CLI: `--tier {fast,thinking,scholar}` forces a tier directly. The chosen tier prints to stderr.
- Web UI: a small caption under each answer ("Fast mode" / "Thinking mode" / "Scholar mode"), driven
  by a new SSE `event: tier` sent before the answer starts streaming.
- `.env.example`: a commented `TYPESAFE_API_KEY` block explaining it is optional.
- `eval/run_gold_eval.py`: `--tier` applies a tier's retrieval depth to the free, retrieval-only gold
  eval.
- Off-topic gate: the routing call now also asks Jev whether a question is about the Xeno series at
  all; a confidently off-topic question gets a canned reply immediately, with no query embedding wait,
  retrieval, rerank, or Gemini call. Config `router.off_topic_gate` / `off_topic_confidence`. Live gate
  eval (`eval/run_jev_gates_eval.py`): 0/200 gold questions wrongly blocked and 30/30 hand-written
  off-topic prompts caught, at every threshold swept (0.7/0.8/0.9) — shipped at 0.7, the lowest meeting
  the ship rule.
- Concurrent routing: the query embedding now starts on a background thread before the Jev routing
  call returns, instead of after it, taking Jev's round-trip off the critical path.
- Answerability check (`xeno_rag/answerability.py`): after rerank, Jev judges whether the retrieved
  chunks actually cover the question. A `not_covered` verdict below Scholar depth escalates retrieval
  once to Scholar and checks again (a second SSE `event: tier` carries `source: "escalated"` so the
  web UI replaces, not appends, the caption); a `not_covered` verdict that survives escalation (or
  starts at Scholar depth) declines with "the wiki doesn't seem to cover it," still showing the
  closest sources, with no Gemini call. Config `router.answerability_check` / `decline_confidence` /
  `answerability_passages`. Live gate eval: gold false-decline rate 2.5% / 2.0% / 1.0% at confidence
  0.7 / 0.8 / 0.9 — shipped at 0.9. At that threshold the 20 hand-written not-covered cases decline
  75% of the time (15/20).
- Format hint: the same routing call also picks table / list / prose, added as one line in the prompt
  before the question. Over the 250-question live-eval set: prose 202, list 35, table 13.
- `eval/run_jev_gates_eval.py` and `eval/jev_gates_cases.json`: a live-eval harness for the two gates
  above (paid Jev calls, budget-guarded) plus 30 hand-written off-topic, 20 hand-written not-covered,
  and 10 hand-written follow-up cases (answerable Xeno follow-ups whose antecedent is only in the
  previous question, e.g. "Who is Nia in Xenoblade Chronicles 2?" → "What species is she?"), sweeping
  thresholds 0.7/0.8/0.9 from recorded confidences with no extra calls. The summary reports a
  follow_up false-block and false-decline rate next to gold's (neither gate should ever fire on a
  follow_up case); the ship rule itself stays gold-based. Re-run with
  `python -m eval.run_jev_gates_eval` (about 570 Jev calls for a full run).
- Answerability check + follow-up context: `answerability.check()` now judges the SAME
  `merge_fragmented_pages`-merged text the Gemini prompt is built from (not the raw fragmented
  retrieval chunks), fixing false `not_covered` declines on stat pages whose retrieved chunks were
  one-line fragments (the merged profile block covers the question; the fragments alone read as
  unrelated one-liners). The passage trim raised from 600 to 1500 chars to fit a merged block. The
  check also receives the previous question as `state["previous_question"]` on a follow-up, the same
  way routing already does, so a terse follow-up ("and what is her element?") is judged with its
  antecedent instead of coverage blind.

### Changed
- Gemini models: fast now uses `gemini-3.5-flash-lite`; thinking and scholar both use
  `gemini-3.8-flash`, with scholar additionally set to Gemini's high `thinking_level`.
- `config.yaml`: `answer_styles` (keyed by model id) replaced by `answer_tiers` (keyed by tier name,
  since thinking and scholar now share one model id) plus a new `router` block.
- A forced `--tier` (CLI, evals) no longer skips Jev entirely: it still makes one Jev call for
  `topic`/`format` when `TYPESAFE_API_KEY` is set, so the off-topic gate and format hint still apply;
  only the tier choice itself is ignored. Without a key it still makes no call.

### Removed
- The Fast/Thinking/Scholar selector from the web UI; every question is now auto-routed.
- CLI `--model` (replaced by `--tier`).
- `gemini-3.1-pro-preview` and the older `gemini-3.1-flash-lite` / `gemini-3.5-flash` model ids.

## [1.3.3] - 2026-08-02

A dependency-security, robustness, and documentation release. No data changes: the shipped `data-v2`
vector store, the embedding vectors, and retrieval behavior are all unchanged, so an existing install
keeps working and no store re-publish is needed.

### Fixed
- CLI: catch a failure from the answer path at the CLI boundary instead of letting it propagate as a
  raw Python traceback with internal file paths. The most common first-run bad path, a missing Gemini
  API key, now prints a clean one-line error and exits, matching the existing convention used by
  `scripts/setup.py`.

### Added
- README: a short "Limitations" section naming what the project doesn't do (no hosted demo, not a
  general-purpose RAG framework, not a wiki mirror) and a few genuine residual gaps, so the README
  reads as self-aware about its own edges rather than silent about them.

### Changed
- README: split the single screenshot and demo GIF, previously placed back to back with no headings,
  into a dedicated stills-only "Screenshots" grid (four distinct application states) and a separately
  placed "Demo" section. The three new stills are frames extracted from the existing demo GIF; no new
  UI capture was needed since the interface is unchanged.

### Security
- Bump `pyasn1` 0.6.3 to 0.6.4 (PYSEC-2026-3455, PYSEC-2026-3456, PYSEC-2026-3457), three
  algorithmic-complexity denial-of-service issues in the ASN.1 BER/CER/DER decoder (quadratic-time
  OID, tag-id, and REAL parsing), pulled in transitively via `pyasn1_modules`. A fresh `pip-audit`
  after the bump reports only the pre-existing, documented, unreachable ChromaDB server-mode advisory.

## [1.3.2] - 2026-07-20

A code-quality, robustness, and dependency-security release that closes an internal code audit: two P1
build-time data-integrity fixes, nine smaller robustness, input-validation, and tooling fixes, and
patches for two dependency security advisories. No data changes:
the shipped `data-v2` vector store, the embedding vectors, and the runtime answer behavior are all
unchanged, so an existing install keeps working and no store re-publish is needed. Every fix ships with a
regression test (the Python suite grows from 284 to 303; the 34 frontend tests are unchanged).

### Fixed
- Corpus build: merge HTML-parsed and wikitext-parsed articles by page id instead of by title, so a
  redirect- or normalization-resolved title mismatch can no longer double-write a page and silently drop
  its Lua-decoded stat tables. A scan of the current corpus found zero live occurrences, so this is a
  preventive fix for future redirects and re-fetches.
- Fetch recovery: the timeout-retry pass writes recovered pages into a reserved batch-index block, so a
  later resumed main fetch can no longer overwrite them.
- Wiki API client: retry a `200 OK` carrying a non-JSON body (a proxy interstitial or challenge page)
  instead of raising and aborting a long pull.
- BM25 lexical index: assert the built row count matches the vector collection and reject duplicate chunk
  ids, so an unstable paginated read cannot silently skip or duplicate chunks.
- Prebuilt-store setup: reject any archive member that would extract outside the target directory (fail
  closed), including a cross-drive member, hardening the `--skip-verify` path.
- Title harvest: write the title list atomically (temp file plus replace) so a mid-harvest failure cannot
  leave a truncated list.
- Web `/ask`: validate the `game` filter against the eight base game codes and return HTTP 422 on an
  unknown value, instead of silently disabling filtering and reflecting the raw string into the prompt.

### Changed
- Consolidate the manual reindex script onto the shared indexing primitive, inheriting its
  missing-page-id guard so one malformed chunk no longer aborts a rebuild, and removing a duplicated
  batch and flush loop.
- `eval/analyze.py` runs under a `main()` guard (now importable and testable) and prints a clear message
  when its input file is absent, instead of raising at import.
- The CLI warns when `--model` names a model with no retrieval-depth entry in the config, instead of
  silently falling back to the base depth.

### Security
- Bump `torch` 2.12.1 to 2.13.0 (PYSEC-2025-194, a `torch.jit.script` memory-corruption issue the app
  never exercises, since torch is used only for embedding and reranking) and `setuptools` 81.0.0 to
  83.0.0 (PYSEC-2026-3447). torch 2.13.0 requires only `setuptools >= 77.0.3`, so both fixes install
  together; `pip-audit` now reports only the unreachable ChromaDB server-mode advisory, which the
  embedded local store never exposes.

## [1.3.1] - 2026-07-17

A retrieval-correctness and data-hygiene release. The backend, corpus, and embedding vectors are
unchanged; the fixes touch how the per-game filter behaves and correct stale metadata in the local
vector store, so an existing `data-v2` store keeps working without a rebuild.

### Fixed
- Relax a per-game retrieval filter when it starves a query, so a page tagged for only some games of a
  subseries is no longer hidden outright. A shared-cast character (for example Joachim Mizrahi, tagged
  for Xenosaga Episode I and III) now surfaces under an Episode II filter when it is the closest match,
  with the reranker re-sorting the merged results; well-populated filters stay untouched.
- Reconstruct the per-game `g_<game>` flags in the local vector store in place, fixing about 130
  cross-appearance pages whose flags were stale. Under-tagged characters (Elma, Fiora, Melia, Nia, and
  Scott among them) were hidden from a valid game filter, and some pages were over-tagged into a game
  they do not appear in. This is a metadata-only repair with no re-embedding. The `data-v2` release
  asset was refreshed from the corrected store, so a fresh `python -m scripts.setup` installs the fixed
  flags; its checksum changed with it, and `scripts/setup.py` pins the new value. A v1.3.0 checkout pins
  the previous checksum and will report a mismatch against the refreshed asset, so use v1.3.1.

### Security
- Bumped httplib2 to 0.32.0 (PYSEC-2026-3444). It is a transitive dependency of an unused Google API
  chain, so exposure was already nil, but a fixable advisory should not ship in the lock.

### Changed
- Documentation and accessibility polish: corrected the vector-store disk-size figure in the README,
  noted the shared-cast filter fallback in the architecture doc, normalized the credits attribution
  separators, and labelled the follow-up question field for screen readers.

## [1.3.0] - 2026-07-14

Frontend redesign of the web UI plus a release-hardening pass. Backend, retrieval, corpus, and the
vector store are unchanged, so an existing `data-v2` store keeps working without a rebuild.

### Changed
- Rebuilt the single-page web UI on a design-token system (surface, radius, elevation, and motion
  tokens driving the per-game `--accent` theming). The all-games view gets a layered cosmic backdrop
  (theme-tinted nebulae, a seeded starfield, a Zohar motif); per-game views keep the game logo and
  key-art wash.
- Reworked the landing into a hero composer (game and answer-style selectors, a RETRIEVE / RERANK /
  CITED ANSWER strip, and example questions), and each answer into a conversation card that streams
  behind a search-phase indicator and a blinking caret.
- Sources now render as a collapsible "grounded in N wiki pages" block with a top-source card, mid
  cards, and compact low rows, each showing a percent-match and a relevance bar. Inline `[n]` markers
  scroll to and highlight the matching card.

### Security
- Bumped Pillow to 12.3.0, clearing five advisories (PYSEC-2026-2253 through 2257). Pillow is a
  transitive dependency used only by the local art-fetch scripts.

### Added
- This changelog, a `CREDITS.md` attribution file, and an `.nvmrc` pinning the Node version used by the
  frontend test runner and CI.

## [1.2.0] - 2026-07-11

### Added
- Inline citation markers: `[n]` in an answer links to the numbered source card it came from.
- A Stop control that aborts a running answer and keeps the partial text; the server also halts its
  model stream when the client disconnects.
- A `/health` endpoint reporting store, BM25, and version status, and an opt-in `XENO_WARM=1` startup
  hook that loads the retrieval singletons at boot instead of on the first question.

### Changed
- Self-host the Cinzel and Spectral fonts as local woff2 files and drop the Google Fonts CDN, so the
  app runs fully offline.

### Fixed
- Escape URLs in attribute context in the Markdown link renderer (XSS hardening).
- Cap question and history lengths in `/ask` to bound prompt cost; block a second concurrent request.
- BM25 index rebuild is now atomic with serving-cache revalidation; single-character tokens are kept.
- Skip chunks with a missing page id during the embed build instead of crashing.

## [1.1.3] - 2026-07-07

### Added
- `XENO_TRUST_PROXY` flag to opt into trusting `X-Forwarded-For` for per-client rate limiting behind a
  reverse proxy; off by default so a directly exposed port cannot be spoofed.
- `pipeline --dry-run`, and a documented build invariant for the query instruction.

### Fixed
- Distinguish retryable timeouts from permanent errors in the HTML fetch.
- Bounds-safe dense query result zipping.

## [1.1.2] - 2026-07-03

### Fixed
- Ship `render.js` in the built wheel.
- Emit multi-tag game membership from the HTML parser; keep column alignment when a stat cell is blank.
- Retry 5xx like 429 and surface terminal 4xx in the MediaWiki client.
- Single-source `__version__` from installed package metadata.

### Changed
- Double-checked locking on the heavy singleton caches; evict idle hosts from the rate-limiter map;
  validate and cap `AskRequest.history`.

## [1.1.1] - 2026-06-29

### Changed
- Cap `chromadb` to `<1.6` so a fresh install can read the shipped `data-v2` index.
- Render the architecture and pipeline diagrams as self-contained SVGs.

### Added
- `.gitattributes` normalizing line endings to LF; a static landing screenshot alongside the demo GIF.

## [1.1.0] - 2026-06-28

### Changed
- Switch the dense embedder to `Qwen/Qwen3-Embedding-0.6B` (1024-dim, instruction-tuned queries).

### Added
- A 200-question gold answer key (25 per game across all 8 titles) and a free, LLM-less retrieval
  evaluation harness (`eval/run_gold_eval.py`).

## [1.0.0] - 2026-06-27

First public release.

### Added
- End-to-end RAG over the Xeno Series Wiki: a resumable offline build (MediaWiki API harvest, hybrid
  rendered-HTML plus wikitext parse, chunk, embed into ChromaDB, build a SQLite FTS5 BM25 index) and a
  per-query serving path (hybrid dense plus lexical retrieval, RRF fusion, cross-encoder rerank,
  grounded and source-cited generation through a provider-agnostic LLM adapter).
- A FastAPI streaming web UI with per-game theming and a game filter, and a CLI.
- Dual licensing: MIT for the code, CC BY-SA 4.0 for the wiki-derived data.

[1.3.3]: https://github.com/yib7/xeno-series-rag/compare/v1.3.2...v1.3.3
[1.3.2]: https://github.com/yib7/xeno-series-rag/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/yib7/xeno-series-rag/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/yib7/xeno-series-rag/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.2.0
[1.1.3]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.3
[1.1.2]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.2
[1.1.1]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.1
[1.1.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.0
[1.0.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.0.0
