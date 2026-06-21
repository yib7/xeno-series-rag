# Milestones — Xeno Series Wiki RAG Chatbot

The project's durable accomplishment log. **Append-only:** unlike `PLAN.md` (which is reset or
archived at the end of each cycle), this file is never reset. It's how the project remembers what
it has shipped across every autopilot cycle — and what lets a new cycle safely reset `PLAN.md`
without losing history. Update it at the end of each cycle, right after
`finishing-a-development-branch`.

## Current state

A working local-first RAG chatbot over the full Xeno Series Wiki. The full corpus is built and
indexed: **36,181 titles → 34,010 articles → 88,338 chunks** embedded with `bge-base-en-v1.5`
(CPU) into a persistent ChromaDB (cosine). Questions are answered grounded + source-cited via a
provider-agnostic LLM (live **Gemini `gemini-2.5-flash`**, key in gitignored `.env`; mock for tests)
through a CLI and a FastAPI+SSE web UI with a game filter. Python 3.12 venv; **59 tests** green.
Stack: requests, mwparserfromhell, sentence-transformers, chromadb, google-genai, fastapi.

## Cycles (newest first)

### Cycle 1 — Full Xeno-wiki RAG build — 2026-06-21 — branch `autopilot/xeno-rag-cycle1` → pending merge

- SP1 Scaffold — package `xeno_rag`, `config.yaml`, 3.12 venv, deps (3 tests).
- SP2 API client — MediaWiki etiquette: UA, `maxlag`, 429/`Retry-After`, backoff, serial+delay (6).
- SP3 Harvest — `allpages` ns=0 pagination → titles.jsonl (5).
- SP4 Fetch — batched, **resumable** content pull w/ checkpoint (4).
- SP5 Parse — wikitext → prose sections + structured infoboxes; game tag; url; drops (11).
- SP6 Chunk — section-aware prose + infobox-as-sentence, breadcrumbs, overlap (8).
- SP7 Embed/index — `bge-base` embedder (DirectML→CPU fallback) + ChromaDB; **resumable** (6).
- SP8 RAG — retrieve + game filter + grounded prompt + Gemini adapter (google-genai) + mock (5).
- SP9 Interface — CLI + FastAPI/SSE web UI with game selector (6).
- SP10 Full data run — harvested/fetched/parsed/chunked/embedded the entire wiki; verified live
  grounded Gemini answers end-to-end.
- SP11 — README (CC-BY-SA attribution, API-etiquette note, setup/run), pipeline driver.
- Deferred (→ BACKLOG): BM25/hybrid retrieval, cross-encoder reranker, incremental refresh, eval
  set, build-generator data sharing, GPU embedding (DirectML blocked by onnxruntime conflict; CPU used).

<!-- prepend each new cycle above this line -->
