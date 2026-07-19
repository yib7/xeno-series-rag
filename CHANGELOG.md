# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html): the code version tracks the application,
and the corpus/vector-store release assets are tagged separately (`data-v1`, `data-v2`).

## [1.3.2] - 2026-07-18

A code-quality and robustness release that closes an internal code audit: two P1 build-time
data-integrity fixes plus nine smaller robustness, input-validation, and tooling fixes. No data changes:
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

[1.3.2]: https://github.com/yib7/xeno-series-rag/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/yib7/xeno-series-rag/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/yib7/xeno-series-rag/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.2.0
[1.1.3]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.3
[1.1.2]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.2
[1.1.1]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.1
[1.1.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.0
[1.0.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.0.0
