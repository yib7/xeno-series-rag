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
| Retrieval chunks (prose + infobox-as-sentence) | 88,338 |
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

## Build the corpus

The full pull hits the live wiki (~36k articles, throttled, ~35 min) — run deliberately:

```bash
.venv\Scripts\python.exe -m xeno_rag.pipeline all        # harvest -> fetch -> parse -> chunk -> embed
```

Each step is independently runnable and **resumable** (`fetch` resumes from its checkpoint; `embed`
skips chunks already indexed):

```bash
.venv\Scripts\python.exe -m xeno_rag.pipeline harvest    # list all titles
.venv\Scripts\python.exe -m xeno_rag.pipeline fetch      # pull wikitext (resumable)
.venv\Scripts\python.exe -m xeno_rag.pipeline parse      # wikitext -> articles.jsonl
.venv\Scripts\python.exe -m xeno_rag.pipeline chunk      # articles -> chunks.jsonl
.venv\Scripts\python.exe -m xeno_rag.pipeline embed      # chunks -> ChromaDB (resumable)
```

> **Embedding device:** defaults to `auto` → CPU here. The repo includes a DirectML (AMD GPU) code
> path, but `onnxruntime-directml` conflicts with the CPU `onnxruntime` ChromaDB needs, so this build
> embeds on CPU. To use a GPU, embed in an isolated env (or an NVIDIA/CUDA box) and copy the
> `data/vectorstore/` folder back — it's portable as long as the same model embeds queries.

## Ask questions

CLI:

```bash
.venv\Scripts\python.exe -m xeno_rag.cli -q "How much power does Infinity Blade have?"
.venv\Scripts\python.exe -m xeno_rag.cli -q "Who is the protagonist?" --game XC2
```

Web UI (FastAPI + SSE streaming + game selector):

```bash
.venv\Scripts\python.exe -m uvicorn xeno_rag.web.app:app --reload
# open http://127.0.0.1:8000
```

## How it works

```
MediaWiki API ─▶ api_client ─▶ harvest_titles ─▶ titles.jsonl
                               fetch_content   ─▶ data/raw/pages/*.jsonl (+ checkpoint)
data/raw ─────▶ parse_wikitext ─▶ articles.jsonl   (prose sections + infobox fields + game + url)
articles ─────▶ chunk           ─▶ chunks.jsonl    (section-aware prose + infobox-as-sentence)
chunks ───────▶ embed_index     ─▶ data/vectorstore (ChromaDB, cosine)
question ─────▶ rag (retrieve + game filter + grounded prompt + LLM) ─▶ answer + source URLs
                               cli / web (SSE)
```

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
- The structured infobox output (`parse_wikitext`) is reusable by a planned multi-game build
  generator.
