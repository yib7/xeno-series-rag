# Xeno Series Wiki RAG Chatbot — Design Spec (Cycle 1)

**Date:** 2026-06-21
**Branch:** `autopilot/xeno-rag-cycle1`
**Source plan:** `xeno-rag-plan.md` (this spec applies the locked decisions on top of it)

## 1. Goal

A local-first Retrieval-Augmented Generation chatbot over the full Xeno Series Wiki
(xenoserieswiki.org): answers natural-language questions about Xeno game mechanics, characters,
items, locations, and lore, grounded in wiki content with per-answer source attribution.

## 2. Locked decisions (Cycle 1)

| Area | Decision |
|---|---|
| Data scope | Full live scrape + full embed of all ~36k `ns=0` articles this cycle |
| Generation LLM | **Gemini** (Vertex/GCP) wired as default, behind a provider-agnostic adapter; tested with a **mock** LLM. Live calls need the user's creds+credits and are never invoked autonomously |
| Embeddings | `BAAI/bge-base-en-v1.5` via ONNX+**DirectML** on the AMD Radeon RX 6600 XT, with **automatic CPU fallback** so the embed always completes |
| Vector store | ChromaDB, persistent, cosine space |
| Interface | CLI **and** a thin FastAPI + SSE streaming web UI with a game selector |
| Runtime | Python **3.12** venv (3.14 is the machine default but ML wheels are safest on 3.12) |

**Hard-stops (autonomy contract):** no secrets created/printed/committed; no real money spent. The
Gemini live path and any cloud cost are the user's manual gate. Everything up to the LLM call is
built and tested with a mock.

## 3. Architecture

Linear pipeline; each stage reads the previous stage's on-disk artifact and is independently
testable. Raw data is pulled from the server exactly once and cached.

```
MediaWiki API ──▶ api_client ──▶ harvest_titles ──▶ titles.jsonl
                                  fetch_content  ──▶ data/raw/pages/*.jsonl + checkpoint.json
data/raw ─────▶ parse_wikitext ─▶ data/processed/articles.jsonl
articles ─────▶ chunk ──────────▶ data/processed/chunks.jsonl
chunks ───────▶ embed_index ────▶ data/vectorstore/ (ChromaDB)
question ─────▶ rag (retrieve+filter+prompt+LLM) ─▶ answer + source URLs
                                  cli.py / web (FastAPI+SSE) ─▶ user
```

### Modules (`src/`)
- **`api_client.py`** — persistent `requests.Session`; descriptive User-Agent; injects
  `format=json`, `formatversion=2`, `maxlag=5`; serial requests; configurable delay; retry on HTTP
  429 (`Retry-After`) and JSON `maxlag` errors with exponential backoff; raises on exhaustion so the
  caller can checkpoint.
- **`harvest_titles.py`** — `list=allpages`, `apnamespace=0`, `aplimit=max`, paginate via
  `apcontinue`; optional `apfilterredir=nonredirects`; writes `titles.jsonl` (`title`, `pageid`).
- **`fetch_content.py`** — batches of 50 titles → `prop=revisions&rvprop=content&rvslots=main`;
  one output file per batch (`pages_NNNNN.jsonl`); after each batch update `checkpoint.json`; resume
  from last completed batch on restart.
- **`parse_wikitext.py`** — `mwparserfromhell`: extract every `{{Infobox …}}`/data template into
  `{template, {param: value}}` (nested markup stripped); split prose by `==` headings into
  `{heading, text}`; resolve `[[Link|display]]→display`, strip refs/formatting; derive `game` from
  the title-suffix convention (`(XC3)`,`(XC2)`,`(XS1)`,`(XG)`,`(XCX)`,… → mapping; unsuffixed →
  `series`); build canonical URL; drop redirects, disambiguation, and <~50-byte stubs (with counts).
  Output `articles.jsonl`.
- **`chunk.py`** — (1) prose chunks: one per section, split if over the token budget (~500–800 tok,
  ~80 overlap), breadcrumb prefix `"[XC3] Title > Heading: …"`; (2) infobox chunks: render fields
  into natural-language sentences. Every chunk carries `chunk_id, pageid, title, game, heading|"infobox", url`.
  Output `chunks.jsonl`.
- **`embed_index.py`** — `Embedder` abstraction with `device` selection: try DirectML (ONNX runtime),
  fall back to CPU. Loads `bge-base-en-v1.5`; batch-encodes chunk text; writes vectors + metadata +
  text to a persistent ChromaDB collection (cosine). BGE query instruction is applied at *search*
  time only, not to stored docs.
- **`rag.py`** — `answer(question, game_filter=None, k=8)`: embed query (with BGE instruction);
  retrieve top-k (filter on `game` metadata when set); build a grounded prompt (answer only from
  context; say when context is insufficient; prefer infobox chunks for stats; cite sources); call the
  LLM adapter; return answer + deduped source URLs. **LLM adapter** is an interface with a Gemini
  implementation and a deterministic mock used in tests.
- **`cli.py`** — read a question (arg/stdin), optional `--game`, print answer + sources.
- **`web/`** — FastAPI app: `/ask` endpoint streaming tokens over SSE; minimal static frontend with
  a question box, a game selector (sets `game_filter`), and a sources list.

### Config (`config.yaml`)
`base_url`, `user_agent`, `request_delay_seconds`, `batch_size` (50), `maxlag` (5), `embed_model`,
`embed_device` (`auto`/`directml`/`cpu`), `vectorstore_path`, `llm_provider` (`gemini`), `top_k`,
and all `data/` paths.

## 4. Data flow & artifacts

`data/raw/titles.jsonl` → `data/raw/pages/*.jsonl` + `checkpoint.json` →
`data/processed/articles.jsonl` → `data/processed/chunks.jsonl` → `data/vectorstore/`.
All of `data/raw`, `data/processed`, `data/vectorstore` are gitignored — pulled/derived once.

## 5. Error handling
- API: retry 429/maxlag with backoff; honor `Retry-After`; raise on exhaustion so fetch checkpoints
  and can resume. Serial only, throttled — respectful of a small fan wiki.
- Fetch: per-batch files + checkpoint make a crash non-corrupting and a restart resumable without
  re-fetching.
- Embedder: DirectML load failure → log once, fall back to CPU; the embed always completes.
- LLM: missing creds → the adapter raises a clear "supply credentials to go live" error; tests use
  the mock and never touch the network.

## 6. Testing strategy
Every module is unit-tested against fixtures/mocks before any heavy live run:
- api_client: mocked 429/maxlag/backoff and delay behavior.
- harvest/fetch: mocked `apcontinue` pagination and resume-from-checkpoint.
- parse: fixture wikitext (character page, art with infobox, class, location, lore) → correct
  infobox fields, clean prose, correct `game`, resolvable URL; stub/redirect dropping.
- chunk: no empty chunks; every chunk has `url`+`game`; infobox renders as a clean sentence; overlap.
- embed_index: embed a few fixture chunks into a temp Chroma; collection count matches; a known-topic
  NN query returns the on-topic chunk with correct metadata.
- rag: with the mock LLM — correct chunks retrieved, `game_filter` honored, prompt contains the
  context + grounding rules, answer surfaces source URLs.
- interface: CLI prints answer+sources (mock); FastAPI `TestClient` hits `/ask`, SSE streams,
  response carries sources.

The full live scrape + full embed (SP10) runs only after SP2–SP9 pass on bounded/fixture data, so no
hour is wasted on a pipeline that can't parse what it pulls.

## 7. Out of scope (Cycle 1 → backlog)
BM25 hybrid retrieval; cross-encoder reranker; incremental timestamp-based refresh; eval Q/A set;
sharing structured infobox output with the build generator; GPU-rental embedding path.

## 8. Acceptance (cycle done)
Full suite green; ChromaDB collection count == chunk count over the real corpus; a real-data NN query
returns on-topic chunks with correct metadata + URLs; CLI and web UI both return grounded,
source-cited answers (with the mock LLM, and with live Gemini once the user supplies creds);
README carries the CC-BY-SA attribution, the API-etiquette note, and setup/run instructions.
