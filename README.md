# Xeno Series Wiki RAG Chatbot

A local-first Retrieval-Augmented Generation chatbot over the full
[Xeno Series Wiki](https://www.xenoserieswiki.org) — answers natural-language questions about Xeno
game mechanics, characters, items, locations, and lore (Xenogears, Xenosaga 1–3, Xenoblade
Chronicles 1/2/3/X), grounded in wiki content with **source attribution on every answer**.

> Built end-to-end: pulls the wiki via the MediaWiki API, parses wikitext into prose + structured
> infobox data, chunks, embeds locally with BGE, indexes in ChromaDB, and answers via a
> provider-agnostic LLM (Gemini by default) through a CLI and a streaming web UI.

## Corpus (this build)

| Stage | Count |
|---|---|
| Titles harvested (`ns=0`, non-redirect) | 36,181 |
| Articles parsed (after dropping redirects/stubs/disambig) | 34,010 |
| Retrieval chunks (prose + infobox/stat-block-as-sentence) | 95,890 |
| Embedding model | `BAAI/bge-base-en-v1.5` (768-dim, cosine) |
| Vector store | ChromaDB (persistent, local) |

## Attribution & license

Content from the **Xeno Series Wiki (xenoserieswiki.org)**, licensed under **CC-BY-SA**. This
project and its derived content (parsed articles, chunks, generated answers) are likewise
**CC-BY-SA**. Every answer surfaces the source page URLs it relied on, satisfying the attribution
requirement in the output itself.

Data was pulled via the **MediaWiki API** (not an HTML scraper) with a descriptive `User-Agent`,
`maxlag=5`, **serial** requests, and a configurable delay — respectful of a small, donation-funded
fan wiki. The full pull is gated behind an explicit command; bounded development fetches are
throttled the same way.

## Setup

Requires Python 3.12 (ML wheels), ~1 GB disk for the corpus + vector store.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[dev]
```

Configuration lives in `config.yaml` (API URL, User-Agent, delays, embedding model/device, model
names, paths). Set the embed `User-Agent` contact before any live pull.

### Enable live answers (Gemini)

Generation is provider-agnostic; the default adapter is Google Gemini. Put your key in a **`.env`**
file (gitignored — never commit it):

```
GEMINI_API_KEY=your-key-here
```

The app auto-loads `.env`. Without a key, retrieval still works and the layers are testable with a
mock LLM; live generation raises a clear "set GEMINI_API_KEY" error.

## Quick start: prebuilt data (recommended)

To try the app without scraping the wiki or running the multi-hour embed, download the prebuilt
vector store from the GitHub release:

```bash
.venv\Scripts\python.exe -m scripts.setup
```

This downloads the `zohar-rag-vectorstore.zip` release asset (~1.1 GB), verifies its checksum, extracts it to
`data/vectorstore/`, and rebuilds the BM25 index locally so it matches the shipped vectors. Re-run
with `--force` to refresh. The download uses the GitHub CLI (`gh`) — install it and run
`gh auth login` first (required while the repo is private). Then add a Gemini key (above) and skip to
[Ask questions](#ask-questions).

## Build the corpus (from scratch)

*Optional — only if you want to regenerate the data yourself; the prebuilt store above is far faster.*
The full pull hits the live wiki (~36k articles, throttled, ~35 min) — run deliberately:

```bash
.venv\Scripts\python.exe -m xeno_rag.pipeline all        # harvest -> fetch -> parse -> chunk -> embed -> bm25
```

Each step is independently runnable and **resumable** (`fetch` resumes from its checkpoint; `embed`
skips chunks already indexed):

```bash
.venv\Scripts\python.exe -m xeno_rag.pipeline harvest    # list all titles
.venv\Scripts\python.exe -m xeno_rag.pipeline fetch      # pull wikitext (resumable)
.venv\Scripts\python.exe -m xeno_rag.pipeline parse      # wikitext -> articles.jsonl
.venv\Scripts\python.exe -m xeno_rag.pipeline chunk      # articles -> chunks.jsonl
.venv\Scripts\python.exe -m xeno_rag.pipeline embed      # chunks -> ChromaDB (resumable)
.venv\Scripts\python.exe -m xeno_rag.pipeline bm25       # build the BM25 lexical index from the collection
```

> **Retrieval is hybrid:** dense BGE vectors + a lexical **BM25** index (SQLite FTS5, built by the
> `bm25` step from the collection), fused with Reciprocal Rank Fusion, then reordered by a
> cross-encoder **reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, downloaded on first query). This
> fixes exact proper-noun / concept recall (e.g. "mimeosomes"). Toggle via `use_bm25` / `use_reranker`
> in `config.yaml`; with both off it falls back to dense-only.

> **Embedding device:** defaults to `auto` → CPU here. The repo includes a DirectML (AMD GPU) code
> path, but `onnxruntime-directml` conflicts with the CPU `onnxruntime` ChromaDB needs, so this build
> embeds on CPU. To use a GPU, embed in an isolated env (or an NVIDIA/CUDA box) and copy the
> `data/vectorstore/` folder back — it's portable as long as the same model embeds queries.

## Ask questions

CLI:

```bash
.venv\Scripts\python.exe -m xeno_rag.cli -q "How much power does Infinity Blade have?"
.venv\Scripts\python.exe -m xeno_rag.cli -q "Who is the protagonist?" --game XC2
# default model is gemini-3.1-flash-lite (fast); override for harder multi-hop questions:
.venv\Scripts\python.exe -m xeno_rag.cli -q "Compare the Vandhams across games" --model gemini-3.5-flash
```

Web UI (FastAPI + SSE streaming) with a game filter, a **Faster / Thinking** model selector,
per-game theming (each Xeno game re-themes the page with its own colour palette, real game logo,
a display font matched to the game's identity, and a faded key-art background wash), and clean
client-side Markdown rendering of answers:

```bash
.venv\Scripts\python.exe -m uvicorn xeno_rag.web.app:app --port 8000
# open http://127.0.0.1:8000
```

"Faster" uses `gemini-3.1-flash-lite`; "Thinking" uses `gemini-3.5-flash`. Requires `GEMINI_API_KEY`
in `.env` for live answers.

**Game filter is series-inclusive.** Most wiki pages have no `(XCn)` title suffix, so they are
tagged `series` (recurring bosses, characters, lore). Selecting a game retrieves that game's pages
**plus** the `series` bucket — so e.g. an "Xenoblade 1" question still surfaces the `Metal Face`
boss pages, which would be hidden by a hard per-game filter.

## How it works

```
MediaWiki API ─▶ api_client ─▶ harvest_titles ─▶ titles.jsonl
                               fetch_content   ─▶ data/raw/pages/*.jsonl       (raw wikitext, all pages)
                               fetch_html      ─▶ data/raw/html/*.jsonl.gz      (rendered HTML, stat pages)
data/raw ─────▶ parse_html.run_hybrid ─▶ articles.jsonl   (HTML facts for stat pages, wikitext prose for the rest)
articles ─────▶ chunk           ─▶ chunks.jsonl    (section-aware prose + table-facts-as-sentences)
chunks ───────▶ embed_index     ─▶ data/vectorstore (ChromaDB, cosine)
question ─────▶ rag (retrieve + game filter + grounded prompt + LLM) ─▶ answer + source URLs
                               cli / web (SSE)
```

**Why two fetch paths.** The wiki's stat tables (Element, HP, weapon, resistances…) are generated by
Lua modules that decode internal numeric codes (`Atr=7`→"Light") *only when rendering HTML* — they're
absent from raw wikitext, and no batch API returns them decoded. So the ~7,593 pages with stat/data
templates are fetched as **rendered HTML** (`action=parse`, one page/call, throttled to the robots.txt
`Crawl-delay: 5`) and parsed with BeautifulSoup; the other ~28k pages keep their clean wikitext prose.
Rebuild everything with `python -m xeno_rag.pipeline rebuild` (fetch → hybrid parse → chunk → fresh embed).

Grounding rules in the system prompt: answer only from retrieved context, say when context is
insufficient, prefer structured infobox chunks for stats, and cite source URLs.

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q          # offline unit/integration suite
.venv\Scripts\python.exe -m pytest -m live -q  # one real throttled API smoke test
```

## Development notes

- `data/` (raw, processed, vectorstore) and `.env` are gitignored — the corpus is pulled/derived
  once and the key is never committed.
- Per-game **brand art** (`xeno_rag/web/static/art/`) is gitignored (copyrighted logos/box art).
  Drop a logo + key-art master per game (`xenoblade-1_logo.png`, `xenoblade-1_keyart.png`, …) into
  that folder, then run `python scripts/optimize_art.py` to derive the web set the UI loads
  (`<code>-logo.png` trimmed/whittled to ~240px tall; `<code>-bg.jpg` downscaled for the wash).
  A dark logo is flipped to white (`whiten`) and a dark-lettered colour logo gets a light halo
  (`glow`) via the `ART` map in `index.html`. If a logo is missing the UI falls back to a styled
  text wordmark, so the app still looks right. (`scripts/fetch_art.py` can source a starter set from
  the Xeno Series Wiki + Wikimedia Commons.)
- Per-game **display fonts** load from Google Fonts (Cinzel, Orbitron, Chakra Petch, Rajdhani,
  Spectral, Fredoka, Oswald, Saira Condensed) and apply to the brand/badge/labels/answer headings
  only; body text stays a clean readable sans for legibility.
- After changing parsing/chunking, rebuild the index with `python scripts/reindex.py` (drops the
  ChromaDB collection and re-embeds every chunk — a plain `embed` skips ids already present).
- The structured infobox output (`parse_wikitext`) is reusable by a planned multi-game build
  generator.
