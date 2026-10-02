# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html): the code version tracks the application,
and the corpus/vector-store release assets are tagged separately (`data-v1`, `data-v2`).

## [Unreleased]

## [1.4.0] - 2026-10-01

Each question is now routed to an answer tier automatically, and two Jev-driven gates (off-topic and
answerability) can stop a question before it reaches Gemini. The release also hardens the build
scripts, the web server and the UI.

Breaking change for scripts: the CLI `--model` flag is removed. Use `--tier {fast,thinking,scholar}`
to force a tier, or leave it off and let the router choose. The web UI no longer has a model or
answer-style selector.

No data changes: retrieval depths per tier match the old answer styles, so the shipped `data-v2`
vector store and retrieval behavior are unaffected.

### Added
- Jev (TypeSafe AI) auto-routing: each question is sent through `xeno_rag/router.py`, which asks
  Jev's "System One" decision model to pick an answer tier (fast, thinking, or scholar) instead of
  showing a selector. Routing plus the coverage check together cost about $0.0002 per on-topic
  question (an upper bound from the Jev price and the passage cap). Routing falls back to a fixed tier
  (default `thinking`) with no network call when `TYPESAFE_API_KEY` is unset, `router.provider` is
  `fixed`, or the call fails or returns a low-confidence choice.
- Off-topic gate: the routing call also asks Jev whether a question is about the Xeno series at all.
  A confidently off-topic question gets a canned reply immediately, with no query embedding wait,
  retrieval, rerank, or Gemini call. Config `router.off_topic_gate` / `off_topic_confidence`. Live gate
  eval (`eval/run_jev_gates_eval.py`): 0/200 gold questions wrongly blocked and 30/30 hand-written
  off-topic prompts caught at every threshold swept (0.7/0.8/0.9). Shipped at 0.7, the lowest that
  meets the ship rule.
- Answerability check (`xeno_rag/answerability.py`): after rerank, Jev judges whether the retrieved
  chunks cover the question. A `not_covered` verdict below Scholar depth escalates retrieval once to
  Scholar and checks again (a second SSE `event: tier` carries `source: "escalated"`, so the web UI
  replaces the caption instead of appending). A `not_covered` verdict that survives escalation, or
  starts at Scholar depth, declines with a "the wiki pages I found don't seem to cover that" message, still shows the
  closest sources, and makes no Gemini call. Config `router.answerability_check` / `decline_confidence` /
  `answerability_passages`. Live gate eval: gold false-decline rate 0.5% (1/200) at every threshold
  swept, 0/10 on the follow-up set. Shipped at 0.7, where 18 of the 20 hand-written not-covered cases
  (90%) decline. Only 1 of the 200 gold questions escalated to Scholar.
- The check judges the same `merge_fragmented_pages` blocks the Gemini prompt is built from, not the
  raw retrieval chunks, so stat pages whose chunks are one-line fragments no longer false-decline. The
  passage trim is 1500 characters to fit a merged block. On a follow-up it also receives the previous
  question as `state["previous_question"]`, the way routing does, so a terse "and what is her element?"
  is judged with its antecedent.
- Format hint: the same routing call picks table, list or prose, added as one line in the prompt before
  the question. Over the 260-question live eval set: prose 209, list 38, table 13.
- Concurrent routing: the query embedding starts on a background thread before the Jev call returns,
  taking Jev's round trip off the critical path.
- SSE `event: declined`, sent between a not-covered message and its sources. The UI heads those sources
  "Closest matches (N wiki pages)" instead of "Grounded in N wiki pages", because they did not support
  an answer.
- Web UI: a caption under each answer ("Fast mode", "Thinking mode", "Scholar mode") driven by the SSE
  `event: tier` sent before the answer starts streaming.
- CLI: `--tier {fast,thinking,scholar}` forces a tier. The chosen tier prints to stderr.
- `eval/run_jev_gates_eval.py` and `eval/jev_gates_cases.json`: a live-eval harness for the two gates
  (paid Jev calls, budget-guarded) with 30 hand-written off-topic, 20 not-covered and 10 follow-up
  cases (answerable follow-ups whose antecedent is only in the previous question, e.g. "Who is Nia in
  Xenoblade Chronicles 2?" then "What species is she?"). It sweeps thresholds 0.7/0.8/0.9 from recorded
  confidences with no extra calls and reports the follow-up false-block and false-decline rates next
  to gold's. Re-run with `python -m eval.run_jev_gates_eval` (about 570 Jev calls for a full run).
- `eval/run_gold_eval.py --tier` applies a tier's retrieval depth to the free, retrieval-only gold eval.
- `httpx` as a runtime dependency, used by the Jev client.
- `XENO_ALLOWED_HOSTS` (see Security) and a commented `TYPESAFE_API_KEY` block in `.env.example`.
- `docs/BUILD_FROM_SCRATCH.md`: the from-scratch corpus build, moved out of the README.
- CI: a test matrix on `ubuntu-latest` and `windows-latest` (ruff, pytest, node tests, and the
  `model`-marked real-embedder test), plus a `README setup` job on both that runs the README's numbered
  setup steps literally, downloads the real store, starts the server and checks `/health`.

### Changed
- Gemini models: fast uses `gemini-3.5-flash-lite`; thinking and scholar use `gemini-3.8-flash`, with
  scholar also set to Gemini's high `thinking_level`.
- `config.yaml`: `answer_styles` (keyed by model id) is replaced by `answer_tiers` (keyed by tier name,
  since thinking and scholar now share one model id) plus a new `router` block.
- A forced `--tier` (CLI, evals) no longer skips Jev entirely: with `TYPESAFE_API_KEY` set it still
  makes one Jev call for `topic` and `format`, so the off-topic gate and format hint apply; only the
  tier choice is ignored. Without a key it makes no call.
- Dependencies refreshed from a clean install and `requirements.txt` regenerated (117 pins). The legacy
  `google-generativeai` and `google-api-python-client` packages, which nothing imported, and the unused
  `tqdm` dependency are gone. Ruff is pinned to an exact version (`==0.16.10`) so local and CI lint agree.
- Tests are hermetic: API keys and `.env` are scrubbed per test, and a hashing fake replaces the Qwen
  download. The real embedder runs in one test behind the `model` marker.
- Repo root: `CHANGELOG.md`, `CREDITS.md`, `LICENSE-DATA.md` and `SECURITY.md` moved into `docs/`.
  `LICENSE` stays at the root, and the wheel still ships both license files.
- README: numbered single-action setup steps with the optional ones labelled, stated RAM, disk and key
  requirements, a "What leaves your machine" section, supported platforms (Windows 11 and Linux; macOS is
  not claimed), and re-shot screenshots and demo GIF. The README and `docs/ARCHITECTURE.md` diagrams are
  now Mermaid blocks that show routing, both gates and the escalate-once path.
- UI contrast and accessibility: button and badge ink, accent text, links and placeholders now meet
  WCAG AA contrast in all nine themes (the Xenoblade 3 and 2 accents are slightly lighter). Added a page
  heading, a visible keyboard focus ring, `aria-busy` on the streaming answer, and labelled citation
  markers. Long source titles no longer push the page wider, table cells no longer split words, and the
  header and phase indicator fit at 320 px.
- `docs/ARCHITECTURE.md` describes the web hardening (Host check, body cap, input limits, rate limit)
  and the concurrency model, and the module map lists `errors.py` and `fileio.py`.

### Fixed
- The web server closes the answer stream when the client disconnects, so a dropped tab no longer leaves
  a paid model stream running.
- `/health` no longer reports store paths or raw exception text.
- A missing Gemini key or vector store is reported before the paid routing call and before the embedding
  model loads. The queued query embed is cancelled when routing fails.
- Setup failures raise `SetupError` with a message naming the next step (no config, no key, no store, a
  missing articles, titles or chunks file) instead of a traceback. Read paths no longer create an empty
  store as a side effect, and the BM25 rebuild verifies the new index before swapping it in. The CLI
  validates `--k`, and `--game` against the eight known codes, and prints one-line errors.
- Build steps write atomically (chunks, articles, fetch checkpoints, HTML batches), and the `embed_fresh` step
  and `reindex --fresh` check their input before dropping the existing store. A chunk overlap at or
  above the chunk size is rejected up front instead of exploding the corpus. Exhausted-retry HTML fetch
  failures are tagged retryable, and the durable-fetch lock now checks that its holder is a live Python
  process.
- `scripts/setup.py` validates before wiping the old store, extracts atomically, resumes an interrupted
  BM25 step, and falls back to HTTPS when `gh` is installed but not logged in (it crashed before).
- Wikitext parsing keeps prose under `===` and `====` subsection headings, which it used to drop. HTML
  parsing splits oversized infoboxes, fixes spacing around inline tags ("Shulk 's") and keeps `thead`
  rows. These affect future rebuilds only; the shipped `data-v2` store is unchanged.
- UI: the server's error message is shown instead of a generic one, off-topic and failed turns stay out
  of the follow-up history, source links are limited to `http(s)`, and the Copy button no longer appears
  while searching or on a failed turn. Citation pills keep dark ink on their accent fill.
- CLI: the source list no longer crashes on a page title the console code page cannot encode (a Scholar
  answer citing a title with a non-Latin symbol on a redirected Windows console). `eval/analyze.py` has
  the same fix.
- A history entry that is not a dict no longer raises inside routing or query rewriting.

### Security
- The Jev API key is sent only to an https URL on `typesafe.ai`, and redirects are not followed, so a
  tampered `router.url` cannot redirect the bearer token to another host.
- Host header check against DNS rebinding: the server answers only to `localhost`, `127.0.0.1` and
  `[::1]`, and any other Host gets a 400. Set `XENO_ALLOWED_HOSTS` to serve a LAN name or sit behind a
  proxy (`*` turns the check off). This changes behavior for anyone who served the app under another
  hostname.
- Request bodies over 1 MiB are refused with a 413 before parsing.
- Absolute paths are replaced with `<path>` in the setup errors relayed to the browser.
- `scripts/fetch_art.py` downloads only from the wiki's own https hosts.
- `render.js` strips NUL bytes from model text, which could forge an internal placeholder token, and a
  bare URL directly followed by a code span no longer swallows the token into a link.
- Dependency advisories patched: `aiohttp` 3.14.3, `anyio` 4.15.1, `cryptography` 50.0.2, `oauthlib`
  4.0.0, `soupsieve` 2.10, `urllib3` 2.8.0, and the installer `pip` 26.2. The one remaining `pip-audit`
  finding is `chromadb` 1.5.9 (HTTP-server mode only, no fixed release, unreachable from this embedded
  use); the reasoning is in [SECURITY.md](SECURITY.md).
- README and `SECURITY.md` state what leaves the machine: the question and retrieved passages go to
  Gemini, and the question, previous question, game and a few passages go to Jev when it is enabled.

### Removed
- The Fast/Thinking/Scholar selector from the web UI; every question is auto-routed.
- CLI `--model` (replaced by `--tier`).
- `gemini-3.1-pro-preview` and the older `gemini-3.1-flash-lite` / `gemini-3.5-flash` model ids.
- `scripts/build_bm25.py`, a duplicate of `python -m xeno_rag.pipeline bm25` that nothing referenced.
- `docs/pipeline.svg` and `docs/architecture.svg`, which predated routing; replaced by Mermaid.
- Dead code: `parse_html.run`, `parse_wikitext.run`, `router.build_request` and `fetch_page_chunks`.

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

[1.4.0]: https://github.com/yib7/xeno-series-rag/compare/v1.3.3...v1.4.0
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
