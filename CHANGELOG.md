# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html): the code version tracks the application,
and the corpus/vector-store release assets are tagged separately (`data-v1`, `data-v2`).

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

[1.3.0]: https://github.com/yib7/xeno-series-rag/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.2.0
[1.1.3]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.3
[1.1.2]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.2
[1.1.1]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.1
[1.1.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.1.0
[1.0.0]: https://github.com/yib7/xeno-series-rag/releases/tag/v1.0.0
