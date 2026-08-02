# Xeno Series Wiki RAG Chatbot

A local-first Retrieval-Augmented Generation chatbot that answers natural-language questions about the
[Xeno Series](https://www.xenoserieswiki.org) games (Xenogears, Xenosaga 1 to 3, Xenoblade Chronicles
1/2/3/X), grounded in wiki content with a source link on every answer.

[![CI](https://github.com/yib7/xeno-series-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/yib7/xeno-series-rag/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Data license: CC BY-SA 4.0](https://img.shields.io/badge/data-CC--BY--SA%204.0-lightgrey.svg)](LICENSE-DATA.md)

## Screenshots

<table>
<tr>
<td><img src="docs/screenshot-landing.png" width="400" alt="All-games landing page: the Xeno wordmark, a game selector and Fast/Thinking/Scholar answer-style selector, an ask box, and example questions"></td>
<td><img src="docs/screenshot-search.png" width="400" alt="Xenoblade 2 theme mid-query: the question is submitted and the retrieve-rerank-cite progress indicator is running while the answer streams in"></td>
</tr>
<tr>
<td><img src="docs/screenshot-answer.png" width="400" alt="A grounded, streamed answer about Mythra with inline bracketed citation markers, collapsed under a Grounded in 15 wiki pages source summary"></td>
<td><img src="docs/screenshot-sources.png" width="400" alt="The expanded source panel showing numbered, relevance-tiered source cards pulled from the wiki, each with a percent match score and an excerpt"></td>
</tr>
</table>

This is a complete RAG system built end to end, not a thin wrapper around an API. It pulls ~36k wiki
articles through the MediaWiki API, parses both rendered HTML (for Lua-decoded stat tables) and
wikitext (for prose), chunks and embeds them locally, and serves answers through a hybrid retriever
(dense vectors plus lexical BM25, fused and reranked) behind a streaming web UI and a CLI. The LLM is
pluggable; everything up to generation runs and is tested without any API key.

## Tech stack

| Layer | Choice |
|---|---|
| Language / runtime | Python 3.12 |
| Retrieval | ChromaDB (dense, cosine) + SQLite FTS5 (lexical BM25), fused with Reciprocal Rank Fusion |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` via sentence-transformers (CPU query embedding; corpus indexed once on a Colab GPU) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Google Gemini via `google-genai`, behind a provider-agnostic adapter (mockable) |
| Web | FastAPI + Server-Sent Events, vanilla-JS frontend with per-game theming |
| Data source | MediaWiki API (not an HTML scraper), with API etiquette baked in |
| Tests | pytest (Python) + node:test (frontend renderer) |

## What it does

- **Grounded answers with inline citations.** Every answer is built only from retrieved wiki context.
  Inline `[n]` markers link each claim to a numbered source card, and every source page URL is
  surfaced, so answers are checkable against the wiki.
- **Hybrid retrieval.** Dense embedding vectors catch paraphrase and meaning; a lexical BM25 index catches
  exact proper nouns and rare terms. The two are fused with Reciprocal Rank Fusion, then a
  cross-encoder reranks the result. This fixed the class of failure where an exact term (for example
  "mimeosomes") embedded poorly and returned nothing useful. The dense side uses
  `Qwen/Qwen3-Embedding-0.6B`, an instruction-tuned decoder embedder: queries are prefixed with a short
  `"Instruct: …\nQuery:"` task instruction while documents are embedded plain. That asymmetric
  query/document split is the convention the model was trained for.
- **Series-aware game filtering.** Most wiki pages carry no `(XCn)` title suffix, so they are tagged
  `series` and surface under every game. Picking a game retrieves that game's pages plus the shared
  `series` bucket, with a multi-tag membership schema so cross-appearance characters resolve to their
  home games.
- **Three answer styles.** Fast, Thinking, and Scholar pair a Gemini model with a retrieval depth, so
  "how the model reasons" and "how much it reads" scale together. The backend keeps a strict allowlist.
- **Per-game theming.** Selecting a game re-themes the page with that game's palette, logo, display
  font, and a faded key-art background.

## Corpus (this build)

| Stage | Count |
|---|---|
| Titles harvested (`ns=0`, non-redirect) | 36,181 |
| Articles parsed (after dropping redirects, stubs, disambiguation) | 34,060 |
| Retrieval chunks (prose + infobox/stat-block sentences) | 289,196 |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` (1024-dim, cosine; instruction-tuned, last-token pooling) |
| Vector store | ChromaDB (persistent, local) |

## Setup

Requires Python 3.12 (the ML wheels are most reliable there) and about 2.2 GB of free disk for the
prebuilt vector store.

```bash
# create the virtual environment
python3.12 -m venv .venv      # Windows: py -3.12 -m venv .venv

# activate it
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# install the project (editable, with dev tools)
pip install -e ".[dev]"
```

The commands below assume the virtual environment is activated, so `python` is the project's
interpreter. If you would rather not activate it, substitute `.venv/bin/python` (macOS/Linux) or
`.venv\Scripts\python` (Windows) for `python`.

Configuration lives in `config.yaml` (API URL, User-Agent, request delays, embedding model and device,
model names, paths). Set your own contact in the User-Agent before any live wiki pull, as a courtesy to
the wiki.

### Enable live answers (Gemini)

Generation is provider-agnostic; the default adapter is Google Gemini. Copy `.env.example` to `.env`
and add your key (`.env` is gitignored and must never be committed):

```
GEMINI_API_KEY=your-key-here
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Without a key, retrieval still
works and the whole stack is testable with a mock LLM; only live generation needs it.

## Quick start: prebuilt data (recommended)

To try the app without scraping the wiki or running the multi-hour embed, download the prebuilt vector
store from the GitHub release:

```bash
python -m scripts.setup
```

This downloads the vector-store release asset (a compressed archive of just over 1 GB), verifies its
checksum, extracts it into `data/vectorstore/` (about 2.2 GB on disk), and rebuilds the BM25 index
locally so it matches the shipped vectors. Re-run with
`--force` to refresh. The download uses a plain HTTPS request, or the GitHub CLI (`gh`) if it is
installed (handy for a progress bar). Then add a Gemini key as above and skip to
[Ask questions](#ask-questions).

## Build the corpus from scratch (optional)

Only needed if you want to regenerate the data yourself; the prebuilt store above is far faster. The
full pull hits the live wiki for ~36k articles, and the stat pages are fetched as rendered HTML one
page per request (throttled to the wiki's `Crawl-delay: 5`), so a from-scratch build takes several
hours. It is fully resumable, so run it deliberately:

```bash
python -m xeno_rag.pipeline all     # harvest -> fetch_wikitext -> fetch -> parse -> chunk -> embed -> bm25
```

Each step is independently runnable and resumable (`fetch` resumes from its checkpoint; `embed` skips
chunks already indexed):

```bash
python -m xeno_rag.pipeline harvest         # list all article titles
python -m xeno_rag.pipeline fetch_wikitext  # pull raw wikitext for every title (resumable)
python -m xeno_rag.pipeline fetch           # pull rendered HTML for the stat pages (resumable)
python -m xeno_rag.pipeline parse           # hybrid HTML + wikitext -> articles.jsonl
python -m xeno_rag.pipeline chunk           # articles -> chunks.jsonl
python -m xeno_rag.pipeline embed           # chunks -> ChromaDB (resumable)
python -m xeno_rag.pipeline bm25            # build the BM25 lexical index from the collection
```

## Ask questions

CLI:

```bash
python -m xeno_rag.cli -q "How much power does Infinity Blade have?"
python -m xeno_rag.cli -q "Who is the protagonist?" --game XC2
python -m xeno_rag.cli -q "Compare the Vandhams across games" --model gemini-3.5-flash
```

Web UI with token-by-token SSE streaming, a game filter, the Fast/Thinking/Scholar selector, per-game
theming, a Stop control that halts a running answer while keeping the partial text, inline citation
markers, and client-side Markdown rendering:

```bash
python -m uvicorn xeno_rag.web.app:app --port 8000
# open http://127.0.0.1:8000
```

The three answer styles map to Gemini models: Fast is `gemini-3.1-flash-lite`, Thinking is
`gemini-3.5-flash`, and Scholar is `gemini-3.1-pro-preview` with the deepest retrieval (built for
broad, whole-series questions, and overkill for simple lookups). Live answers need `GEMINI_API_KEY`.

For a long-running deployment, set `XENO_WARM=1` to load the models at startup instead of on the first
question, and poll `GET /health` for store, index, and version status.

## Demo

![Animated walkthrough: selecting Xenoblade 2, asking about Mythra, and getting a grounded streamed answer whose inline bracketed citations link to numbered, rank-tiered source cards from the wiki](docs/demo.gif)

## How it works

<p align="center">
  <img src="docs/pipeline.svg" width="600"
       alt="Pipeline: an offline index-build lane (MediaWiki API to harvest_titles, fetch, parse_html.run_hybrid, chunk, into a ChromaDB vector store and a SQLite FTS5 BM25 store) feeding a per-query serving lane (question to retrieve, rerank, grounded prompt to LLM, answer with sources).">
</p>

The path down to the two stores is built once, offline; everything from `question` onward runs per query.

**Why two fetch paths.** The wiki's stat tables (Element, HP, weapon, resistances) are generated by Lua
modules that decode internal numeric codes (`Atr=7` becomes "Light") only when rendering HTML; they are
absent from raw wikitext, and no batch API returns them decoded. So the ~7,593 pages with stat/data
templates are fetched as rendered HTML and parsed with BeautifulSoup, while the rest keep their clean
wikitext prose. Rebuild everything with `python -m xeno_rag.pipeline rebuild`.

For a deeper walkthrough of the modules and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation

Retrieval quality is measured against a hand-built **gold question set of 200 questions (25 per game
across all 8 Xeno titles)**, with a deliberate spread of categories: characters, enemy and boss stats,
art and attack values, collectible locations, quests, world and lore, items, and mechanics. Every
question has a documented correct answer grounded in the indexed corpus and linked to its source wiki
page. The full set is human-readable in [eval/QUESTIONS.md](eval/QUESTIONS.md) (machine-readable
[eval/gold_questions.json](eval/gold_questions.json)).

The harness scores the production hybrid retriever (dense `Qwen/Qwen3-Embedding-0.6B` + lexical BM25,
RRF-fused and cross-encoder reranked) on whether the gold source page is surfaced under each
question's game filter. This "source-page hit rate" is free (no LLM call) and is exactly the signal a
retrieval change moves. On the current gold set the retriever finds the correct grounding page for
**all 200 questions across all 8 games (100%)**. Methodology and the per-question breakdown are in
[eval/](eval/) and [docs/eval/](docs/eval/).

```bash
python -m eval.run_gold_eval            # retrieval scoring against the 200-question gold set (free)
```

## Tests

```bash
pytest -q                  # offline unit and integration suite (no API key, no live calls)
node --test tests/js/*.test.mjs   # frontend renderer (also wrapped into the pytest run)
pytest -m live -q          # one real, throttled API smoke test (opt in)
ruff check .               # lint
```

## Attribution and license

This project is dual-licensed, because it bundles two different kinds of thing:

- **Code** (the pipeline, web app, scripts, config) is **MIT** ([LICENSE](LICENSE)).
- **Wiki-derived data** (the corpus and embeddings in the release asset, the `tests/fixtures/` wiki
  text and HTML, parsed articles, chunks, and generated answers) is **CC BY-SA 4.0**
  ([LICENSE-DATA.md](LICENSE-DATA.md)), the same license the
  [Xeno Series Wiki](https://www.xenoserieswiki.org) uses. Share-alike requires anything derived from
  that content to stay CC BY-SA.

Every answer surfaces the source page URLs it relied on, satisfying the attribution requirement in the
output itself. Data was pulled through the MediaWiki API (not an HTML scraper) with a descriptive
User-Agent, `maxlag=5`, serial requests, and a configurable delay, out of respect for a small,
donation-funded fan wiki.

### Game artwork, logos, and trademarks

No official game artwork, logos, key art, or box art is included in this repository. Those assets are
the property of their respective owners (Nintendo, Monolith Soft, Bandai Namco Entertainment, and
Square Enix), and all rights are reserved to them. The per-game logo and key-art files the UI can
display are fetched locally by `scripts/fetch_art.py`, are git-ignored, and are never redistributed
here. When they are absent the UI falls back to styled text wordmarks, so the app runs fully without
them.

This is an unofficial, non-commercial fan project. It is not affiliated with, endorsed by, or sponsored
by any of those rights holders. Game and series names are trademarks of their respective owners and are
used here only for identification and descriptive purposes.

Security notes (posture, input handling, dependency audit) are in [SECURITY.md](SECURITY.md).

## Limitations

- Local-only: no hosted demo. The app binds `127.0.0.1` and runs on your machine, so there is no live
  URL to try it from.
- Purpose-built, not a framework: parsing, chunking, and tagging are shaped around this one wiki's
  structure, not a general-purpose RAG toolkit you can point at another site.
- No wiki mirror or bulk redistribution: the shipped store is a derived, embedded index for answering
  questions, not a redistributable copy of the wiki's raw content.
- A few narrow gaps: non-mainline media pages (anime, spinoffs, albums) fall back to a series-wide tag
  instead of a precise game filter; a handful of topics with no single wiki page to consolidate them
  (for example the Solaris caste hierarchy) lean on scattered context; the `eval/analyze.py` dev script
  hits a Windows-console encoding error on non-ASCII output.
