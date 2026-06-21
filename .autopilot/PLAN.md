# PLAN — Xeno Series Wiki RAG Chatbot (Cycle 1)

> On autopilot. Resume point = the first unchecked box below. Isolated on branch
> `autopilot/xeno-rag-cycle1` (never `main`). Autonomy contract: `.autopilot/AUTONOMY.md`.
> New ideas → `.autopilot/BACKLOG.md`. Assumptions / reversible decisions →
> `.autopilot/DECISIONS.md`. Shipped history → `.autopilot/MILESTONES.md`.
> Full design: `docs/superpowers/specs/2026-06-21-xeno-rag-design.md`.

## Scope (frozen)

Build a local-first RAG chatbot over the full Xeno Series Wiki: pull all ~36k `ns=0` articles via
the MediaWiki API (this cycle), parse wikitext into prose + structured infobox data, chunk, embed
with `bge-base-en-v1.5` (DirectML→CPU fallback), index in ChromaDB, and answer grounded,
source-cited questions via a CLI and a thin FastAPI+SSE web UI. Generation is provider-agnostic with
Gemini as the default adapter, tested with a mock. **OUT of scope:** BM25/hybrid retrieval, reranker,
incremental refresh, eval set, build-generator sharing, GPU-rental embedding.

## Global constraints (every phase inherits these)
- **Python 3.12** venv at `.venv` (`py -3.12 -m venv .venv`). Run tools via `.venv\Scripts\python.exe`
  (Windows). Never the machine-default 3.14.
- Package is `xeno_rag/` (top-level), installed editable (`pip install -e .`). Tests in `tests/`.
- **MediaWiki etiquette is mandatory** in any code touching the live API: descriptive User-Agent,
  `format=json`+`formatversion=2`+`maxlag=5` on every call, **serial requests only**, configured
  delay between calls, honor `Retry-After`/429 with exponential backoff.
- **No secrets, no real money** (autonomy hard-stops). The Gemini live path reads creds from env and
  is **never invoked in tests or by default code paths** — tests use the mock LLM.
- TDD: write the failing test first, watch it fail, implement minimal, watch it pass, commit. Small
  commits per task.
- All `data/` (raw, processed, vectorstore) is gitignored. The full live pull happens once (SP10).

---

## SP1 — Scaffold, config, venv, deps

**Checkpoint:** `.venv\Scripts\python.exe -m pytest -q` runs (a smoke test passes); `import xeno_rag`
works; `config.yaml` loads; a live `siteinfo` curl returns the ~36k article count.

- [ ] Create `pyproject.toml` (package `xeno_rag`, setuptools) with deps: `requests`,
  `mwparserfromhell`, `pyyaml`, `numpy`, `tqdm`, `sentence-transformers`, `optimum[onnxruntime]`,
  `onnxruntime-directml`, `chromadb`, `google-generativeai`, `fastapi`, `uvicorn[standard]`; dev:
  `pytest`, `pytest-mock`, `httpx`.
- [ ] Create `xeno_rag/__init__.py` and the empty module stubs referenced below; `tests/__init__.py`.
- [ ] Create `config.yaml` with keys: `base_url: https://www.xenoserieswiki.org/w/api.php`,
  `user_agent` (XenoRAG/0.1 + contact), `request_delay_seconds: 2`, `batch_size: 50`, `maxlag: 5`,
  `embed_model: BAAI/bge-base-en-v1.5`, `embed_device: auto`, `bge_query_instruction`,
  `llm_provider: gemini`, `top_k: 8`, and `paths:` for titles/pages/checkpoint/articles/chunks/vectorstore.
- [ ] Create `xeno_rag/config.py` → `load_config(path="config.yaml") -> dict` (yaml.safe_load).
- [ ] `py -3.12 -m venv .venv`; `.venv\Scripts\python.exe -m pip install -e .[dev]`.
- [ ] Tests: `tests/test_config.py` — `load_config()` returns a dict with `base_url`, `batch_size==50`,
  `maxlag==5`, and a `paths` mapping. A `tests/test_smoke.py` asserts `import xeno_rag`.
- [ ] Recon (not a test, run once): `curl -s "https://www.xenoserieswiki.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"` → confirm article count; record it in DECISIONS.

## SP2 — API client (`xeno_rag/api_client.py`)

**Checkpoint:** `pytest tests/test_api_client.py -q` green; one real throttled `siteinfo` call returns
parsed JSON with a statistics block.

**Produces:** `WikiClient(cfg)` with `.get(params: dict, max_retries=6) -> dict`. Injects
`format=json, formatversion=2, maxlag=<cfg>`; sleeps `request_delay_seconds` after each success;
on HTTP 429 reads `Retry-After` then exponential backoff; on JSON `error.code == "maxlag"` backs off;
raises `RuntimeError` when retries exhausted.

- [ ] Tests (mock `requests.Session.get`): (a) success injects the 4 params + sleeps; (b) a 429 then
  200 retries and honors `Retry-After`; (c) a maxlag JSON error then success retries with backoff;
  (d) exhausted retries raise `RuntimeError`. Use `pytest-mock`/monkeypatch; patch `time.sleep`.
- [ ] Implement `WikiClient` per the skeleton in `xeno-rag-plan.md` §5.
- [ ] One live throttled smoke test (marked `@pytest.mark.live`, not in default run) hitting siteinfo.
- [ ] Commit.

## SP3 — Harvest titles (`xeno_rag/harvest_titles.py`)

**Checkpoint:** `pytest tests/test_harvest.py -q` green (mocked pagination yields all records across
`apcontinue` pages and stops); writer produces JSONL with `title`+`pageid`.

**Produces:** `harvest_titles(client, cfg, nonredirects=True) -> Iterator[dict]` ({title, pageid});
`write_titles(records, path)`; `run(cfg)` orchestrates and writes `paths.titles`.

- [ ] Tests: feed a fake client returning 2 `allpages` pages (second has no `continue`) → generator
  yields all rows, updates params with `apcontinue` between pages, terminates. `apfilterredir` set
  when `nonredirects`.
- [ ] Implement using `list=allpages, apnamespace=0, aplimit=max`, paginate on `data["continue"]`.
- [ ] Commit.

## SP4 — Fetch content, batched + resumable (`xeno_rag/fetch_content.py`)

**Checkpoint:** `pytest tests/test_fetch.py -q` green, including a **resume** test: after a simulated
crash at batch k, a restart skips batches `< k` and re-fetches none already written.

**Produces:** `batched(iterable, n=50)`; `fetch_all(client, titles, cfg, start_batch=0)` writing
`paths.pages/pages_NNNNN.jsonl` per batch and `save_checkpoint(i, path)` / `load_checkpoint(path)`
({last_completed_batch}); `run(cfg)` reads titles, resumes from checkpoint.

- [ ] Tests: (a) `batched` chunks correctly incl. remainder; (b) `fetch_all` writes one file per batch
  and advances checkpoint; (c) resume: with checkpoint at k and existing files, restart begins at k+1
  and the mock client is not called for earlier batches.
- [ ] Implement with `prop=revisions&rvprop=content&rvslots=main&titles=A|B|...`.
- [ ] Commit.

## SP5 — Parse wikitext (`xeno_rag/parse_wikitext.py`)

**Checkpoint:** `pytest tests/test_parse.py -q` green over fixture wikitext (character, art-with-infobox,
class, location, lore); `articles.jsonl` records carry `title, pageid, game, url, infoboxes, sections`.

**Produces:** `derive_game(title) -> str` (suffix map → `XC3/XC2/XC1/XCX/XS1/XS2/XS3/XG/...`, else
`series`); `title_to_url(title) -> str`; `parse_article(title, pageid, wikitext, cfg) -> dict | None`
(None for redirect/disambig/<~50-byte stub); `run(cfg)` streams raw pages → `paths.articles`,
logging drop counts.

- [ ] Add `tests/fixtures/*.wikitext` (5 varied pages — small, hand-written or trimmed real samples).
- [ ] Tests: infobox template + params extracted with nested markup stripped; `[[A|B]]→B` and refs
  stripped from prose; sections split on `==`; `derive_game` covers each suffix + `series`;
  `title_to_url` spaces→underscores; redirect/stub returns None.
- [ ] Implement with `mwparserfromhell` (templates, headings, `strip_code`).
- [ ] Commit.

## SP6 — Chunking (`xeno_rag/chunk.py`)

**Checkpoint:** `pytest tests/test_chunk.py -q` green; no empty chunks; every chunk has `url`+`game`;
an infobox chunk reads as a natural sentence.

**Produces:** `chunk_article(article, cfg) -> list[dict]` emitting prose chunks (one per section, split
over token budget ~500–800 with ~80 overlap, breadcrumb prefix `"[GAME] Title > Heading: <text>"`)
and infobox chunks (fields rendered to sentences, `heading="infobox"`); each chunk:
`{chunk_id, pageid, title, game, heading, url, text}`. `run(cfg)` → `paths.chunks`.

- [ ] Tests: a long section splits with overlap; a short section is one chunk; infobox dict →
  sentence string mentioning key fields; every emitted chunk non-empty and has `url`+`game`;
  `chunk_id` unique within an article. Token counting may approximate via whitespace/`len`.
- [ ] Implement. Commit.

## SP7 — Embed + index (`xeno_rag/embed_index.py`)

**Checkpoint:** `pytest tests/test_embed.py -q` green; embeds a few fixture chunks into a **temp**
ChromaDB; collection count == chunk count; a known-topic nearest-neighbor query returns the matching
chunk with correct metadata.

**Produces:** `Embedder(cfg)` with `.encode(texts: list[str]) -> np.ndarray` — tries DirectML
(ONNX runtime provider), falls back to CPU on load failure (logged once); `embed_query(text)` applies
the BGE instruction. `build_index(chunks, cfg, client=None)` writes vectors+metadata+text to a
persistent ChromaDB collection (cosine); `query(text, k, game_filter, cfg) -> list[dict]`.

- [ ] Tests: use CPU device explicitly + a tiny set of real chunks into a `tmp_path` Chroma; assert
  count and that a query for a known sentence ranks the right chunk first and returns its metadata;
  assert `game_filter` restricts results. (Downloads bge-base once — free, no key.)
- [ ] Implement Embedder with try-DirectML/except→CPU; ChromaDB persistent client, cosine space.
- [ ] Commit.

## SP8 — Retrieval + generation (`xeno_rag/rag.py`)

**Checkpoint:** `pytest tests/test_rag.py -q` green using the **mock** LLM: correct chunks retrieved,
`game_filter` honored, prompt contains the context + grounding rules, result surfaces deduped source
URLs. No network call in tests.

**Produces:** `LLMClient` protocol (`.generate(system, prompt) -> str`); `GeminiClient(cfg)` (reads
`GOOGLE_API_KEY`/ADC at call time, raises a clear error if absent); `MockLLM(canned)`; `build_prompt
(question, chunks) -> (system, user)`; `answer(question, game_filter=None, k=8, llm=None, cfg=None)
-> {"answer": str, "sources": list[str]}` (defaults `llm` to Gemini, injectable mock in tests).

- [ ] Tests: with `MockLLM` and a temp index from SP7 fixtures — retrieval returns expected chunks;
  `game_filter="XC3"` excludes other games; system prompt includes "answer only from context / cite
  sources / prefer infobox for stats"; `sources` are the deduped URLs of retrieved chunks;
  `GeminiClient.generate` raises without creds (no network).
- [ ] Implement. Commit.

## SP9 — Interface: CLI + web (`xeno_rag/cli.py`, `xeno_rag/web/`)

**Checkpoint:** `pytest tests/test_cli.py tests/test_web.py -q` green (mock LLM): CLI prints answer +
sources; FastAPI `TestClient` `POST /ask` streams SSE tokens and returns sources; static index page
served.

**Produces:** `cli.main(argv=None)` (`--question`, `--game`, `--k`, prints answer + Sources list);
FastAPI app `web/app.py` with `POST /ask` (body: question, game) streaming tokens via SSE then a
final `sources` event; `web/static/index.html` (question box, game `<select>`, answer pane, sources).
Both call `rag.answer` (LLM injectable for tests).

- [ ] Tests: CLI with injected mock prints expected lines (capsys); `TestClient` hits `/ask`, asserts
  `text/event-stream`, streamed tokens, and a sources payload; GET `/` returns the HTML.
- [ ] Implement. Commit.

## SP10 — Full data population (heavy, live, authorized)

**Checkpoint:** `data/raw/titles.jsonl` has tens of thousands of rows; `data/raw/pages/` batches all
complete with checkpoint at the final batch; `articles.jsonl` + `chunks.jsonl` populated; ChromaDB
collection count == chunk count; a real-data NN query (e.g. a known art's stats) returns on-topic
chunks with correct metadata + URL.

> Runs only after SP1–SP9 are green. Long-running steps run in the background with checkpoint/resume.

- [ ] Run `harvest_titles.run(cfg)` → full `titles.jsonl` (a few minutes).
- [ ] Run `fetch_content.run(cfg)` → full corpus, **resumable**, in background (~1hr, throttled). On
  any interruption, restart resumes from checkpoint.
- [ ] Run `parse_wikitext.run(cfg)` → `articles.jsonl`; record drop counts.
- [ ] Run `chunk.run(cfg)` → `chunks.jsonl`.
- [ ] Run `build_index(chunks, cfg)` → full ChromaDB (DirectML if it loads, else CPU; background).
- [ ] Verify: collection count == chunk count; sample real-data queries on-topic across ≥2 games.

## SP11 — README + finish

**Checkpoint:** `README.md` present with the required sections; full suite
`.venv\Scripts\python.exe -m pytest -q` green.

- [ ] Write `README.md`: one-line description; CC-BY-SA attribution ("Content from the Xeno Series
  Wiki (xenoserieswiki.org), licensed under CC-BY-SA. This project and its derived content are
  likewise CC-BY-SA."); note that data was pulled via the MediaWiki API with rate limiting + a
  descriptive User-Agent; setup + run instructions (venv, install, config, full-pull command, CLI,
  web, and how to add Gemini creds to go live).
- [ ] Run full suite; confirm green.
- [ ] Commit.

## Blocked (filled in during the run)

-
