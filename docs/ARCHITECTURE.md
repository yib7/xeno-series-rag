# Architecture

A newcomer's map of how this RAG chatbot works, end to end. Two halves: an **offline corpus build**
that turns the Xeno Series Wiki into a searchable index, and an **online query path** that answers a
question against that index with a cited, grounded response.

## The big picture

<p align="center">
  <img src="architecture.svg" width="720"
       alt="Big picture: an offline build-once pipeline (MediaWiki API, harvest titles, fetch rendered HTML, parse, chunk, embed into ChromaDB, build a BM25 index) and a per-question online path (question with game filter, retrieval query, dense and lexical search, RRF fusion, cross-encoder rerank, grounded prompt, Gemini generation, answer with deduped source links).">
</p>

The build is a linear pipeline where every stage reads the previous stage's on-disk artifact, so each
stage is independently runnable, resumable, and testable. The query path is a single function
(`rag.answer` / `rag.answer_stream`) composed from the retrieval, rerank, and generation modules.

## Offline: the corpus build

Driven by `xeno_rag/pipeline.py` (`python -m xeno_rag.pipeline all`). Steps, in order:

1. **harvest** (`harvest_titles.py`) lists every `ns=0` article title via `list=allpages`, paginating
   on `apcontinue`.
2. **fetch** (`fetch_html.py`) pulls each page's *rendered* HTML through `action=parse`. Rendered HTML
   is used because the wiki's stat and data tables are produced by Lua modules: the raw wikitext only
   holds template calls, so the actual decoded values (stats, drops, resistances) exist only after the
   server renders them. Wikitext is still kept for prose-heavy pages.
3. **parse** (`parse_html.py`, `parse_wikitext.py`) runs a hybrid parse: HTML table extraction for
   stat/data pages, wikitext prose extraction for the rest, merged into one article record per page.
   Infobox and data templates become structured field maps; prose is split by `==` headings.
4. **chunk** (`chunk.py`) produces two chunk kinds: prose chunks (one per section, split to a token
   budget with overlap, prefixed with a `"[XC3] Title > Heading"` breadcrumb) and infobox chunks
   (structured fields rendered into natural-language sentences). Every chunk carries
   `chunk_id, pageid, title, game, heading, url`.
5. **embed** (`embed_index.py`) encodes chunk text with `Qwen/Qwen3-Embedding-0.6B` (a 1024-dim decoder
   embedder with last-token pooling, so the tokenizer is left-padded) and writes vectors, metadata, and
   text to a persistent ChromaDB collection (cosine space). The one-time corpus indexing runs on a GPU
   (Colab); at serve time a single query embeds on CPU in well under a second. Embedding is asymmetric:
   an `"Instruct: …\nQuery:"` instruction is prepended only to queries at search time, never to stored
   documents, the convention Qwen3-Embedding was trained on. This instruction is the `query_instruction`
   field in `config.yaml`. **It MUST match, character-for-character, the instruction used to embed the
   corpus on Colab**. The store is built there, served here, and a mismatch silently lands query and
   document vectors in different spaces (retrieval quietly degrades, no error). Treat it as a build
   invariant: change it in one place and you must re-embed.
6. **bm25** (`bm25_index.py`) builds a lexical SQLite FTS5 index over the same embedded collection, so
   its document set and game tags match the dense index exactly.

A full live pull is large (~36k articles, roughly 19h at the throttle) and is gated behind an explicit
command. Fetching is checkpointed per batch, so a crash or sleep resumes without re-fetching.
`scripts/run_fetch_durable.py` wraps the long pull in a self-healing scheduled task that survives
Windows Modern Standby. The prebuilt index is published as a GitHub release asset so most users never
run the pull at all (see `scripts/setup.py`).

## Online: the query path

`rag.answer(question, cfg, game_filter=...)` composes these steps:

1. **Retrieval query** (`rag._retrieval_query`) optionally folds in the previous question for
   conversational follow-ups, without polluting retrieval with the whole session. Its embedding is
   submitted to a small background `ThreadPoolExecutor` immediately, running concurrently with the Jev
   routing call below rather than after it; retrieval then awaits that future instead of embedding
   again.
2. **Routing and gates** (`router.route`, see "Answer tiers, routing, and gates" below) picks the tier
   and, from the same Jev call, a topic and a format. An off-topic topic short-circuits here: the
   canned `OFF_TOPIC_MESSAGE` is returned/streamed immediately, with no retrieval, rerank, or Gemini
   call (the in-flight embedding future is cancelled — a no-op if it already started, in which case
   it finishes in the background and is discarded).
3. **Hybrid retrieval** (`retrieve.py`) runs two independent searches: dense nearest-neighbour over
   ChromaDB and lexical BM25 over the FTS5 index, then fuses their rankings with Reciprocal Rank
   Fusion. Lexical recall fixes the case where an exact proper noun (a boss name, a mechanic) embeds
   poorly but matches a keyword cleanly. A `game` metadata filter scopes results to a selected game
   plus series-wide pages. Because that filter is hard, a character tagged for only some games in a
   subseries could be hidden entirely when asked under one of the others; so when the filter matches
   nothing, or its best candidate trails the best *unfiltered* match by at least `retrieve_relax_gap`
   (default 0.10 cosine units, i.e. a strictly closer page is being excluded), retrieval relaxes to
   unfiltered for that one query and lets the reranker re-sort. Well-populated filters (gap ~0) are
   untouched.
4. **Rerank** (`rerank.py`) reorders the fused candidates with a `cross-encoder/ms-marco-MiniLM-L-6-v2`
   model and attaches a relevance score, which the web UI turns into relevance-tiered source cards.
5. **Merge + answerability check.** Before Jev sees anything, `retrieve.merge_fragmented_pages` folds
   a stat page's fragmented factblock chunks (one-line "Introduction: X is an enemy..." scraps) into
   one coherent profile block — the exact same merge step 6 uses for the prompt, run here first so
   the check judges the same text Gemini will. `answerability.py` (see below) then asks Jev whether
   the top merged, reranked chunks actually cover the question; a not-covered verdict escalates
   retrieval once to Scholar depth (steps 3-5 repeat at that depth) and, if it still isn't covered,
   the pipeline stops here and declines instead of calling Gemini.
6. **Prompt** (`rag.build_prompt`) assembles a grounded prompt from the same merged chunks step 5
   checked: answer only from the retrieved context,
   say so when the context is insufficient, prefer infobox chunks for stats, and cite sources. When
   Jev's `format` answer is usable, one line steering the answer toward a table, a bullet list, or
   prose is inserted before the question.
7. **Generation** (`rag.GeminiClient`) calls the LLM behind a small adapter interface. Tests use a
   deterministic `MockLLM` and never touch the network. Credentials are read from the environment at
   call time.

### Answer tiers, routing, and gates

Each question is answered at one of three tiers, each pairing a Gemini model with a retrieval depth
(configured in `config.yaml` under `answer_tiers`):

- **fast** (`gemini-3.5-flash-lite`): lean retrieval for quick, focused lookups (a stat, level,
  location, drop, or who/what something is).
- **thinking** (`gemini-3.8-flash`): wider candidate pools and more kept chunks for explanations and
  comparisons across a few topics or one game's story arc. This is also the fallback tier.
- **scholar** (`gemini-3.8-flash`, `thinking_level: high`): the deepest retrieval profile, built for
  broad synthesis across many pages or several games.

`xeno_rag/router.py` picks the tier before retrieval runs, from one request to Jev (TypeSafe AI's
"System One" decision model, model id `jev-latest`) that carries **three independent questions** over
one shared state (the question, the previous question from history for terse follow-ups, and the
selected game): `tier` (fast/thinking/scholar), `topic` (`xeno`/`off_topic`), and `format`
(`table`/`list`/`prose`). Jev returns a typed `{"choice": ..., "confidence": <float>}` per question
instead of generated text, which is what keeps routing cheap ($0.042 per 1M input tokens, output
free — see the call-count and total-cost breakdown below). Each answer is parsed independently, so one malformed answer never
discards the others. The shared `router._jev_call` helper falls back to `router.fallback_tier` (default
`thinking`) for the tier, and to no gate/no hint for topic/format, with no HTTP call at all when
`TYPESAFE_API_KEY` is unset or `router.provider` is `fixed`, and on any failure once a call is made:
a timeout, a connection error, a non-2xx response, or a malformed reply. There are no retries, and
routing never raises into the answer path. `router.apply_tier(cfg, tier)` then merges the chosen tier's
retrieval-depth keys and model id over the base config; `rag.answer()` / `answer_stream()` accept an
optional `tier` override (used by the CLI's `--tier` flag and the gold eval's `--tier`) that skips only
the tier choice (`Route.source == "override"`) — Jev is still called for `topic`/`format` when a key is
set, so the off-topic gate and format hint still work on a forced tier; the answerability check below
is skipped for a forced tier. `answer_stream()` yields a `("tier", {"tier": ..., "source": ...})` event
first, before any retrieval, which the web layer forwards as an SSE `event: tier` so the UI can caption
the answer ("Fast mode", "Thinking mode", "Scholar mode") without a selector.

Two gates build on routing, both **off by default in code** (`router.off_topic_gate`,
`router.answerability_check` default `False`, so an old config keeps today's behaviour exactly) but
**on in the shipped `config.yaml`**, per the gate eval's ship rule (see below):

- **Off-topic gate.** This one rides the SAME routing call: `topic` is one of the three questions in
  the single Jev request above, so the gate costs no extra call. When `topic == "off_topic"` at or
  above `router.off_topic_confidence` (code default 0.8; the shipped config ships 0.7 — the eval's
  lowest swept threshold already clears the ship rule at 0/200 gold false-blocks), `answer()`/
  `answer_stream()` return the canned `OFF_TOPIC_MESSAGE` immediately — no retrieval, rerank, or
  Gemini call, and the concurrently-running query embedding (see step 1 above) is cancelled rather
  than awaited or left to run to completion. `answer_stream()` yields no `tier` event in this case, so
  the UI shows no mode caption.
- **Answerability check** (`xeno_rag/answerability.py`). This is a SEPARATE Jev call, made only after
  rerank (routing's `topic`/`format` answers are already in hand by then). `answerability.check()`
  sends Jev the question, the previous question when this is a follow-up, plus up to
  `router.answerability_passages` (default 8) passages — the same `merge_fragmented_pages`-merged,
  breadcrumb-stripped blocks the Gemini prompt itself is built from (not the raw fragmented chunks),
  each trimmed to 1500 chars — and gets back a `coverage` verdict (`answered`/`partial`/
  `not_covered`). A `not_covered` verdict at or above `router.decline_confidence` (code default 0.8;
  the shipped config ships 0.9 — the eval's gold false-decline only clears the 1% ship rule at that
  threshold) below Scholar depth triggers one re-retrieve at Scholar depth (same query vector) and a
  second check; `answer_stream()` yields a **second** `("tier", {"tier": "scholar", "source":
  "escalated"})` event for this, which the web UI's tier handler replaces the caption with rather than
  appending. If the (possibly escalated) verdict is still `not_covered`, the pipeline declines: it
  returns `NOT_COVERED_MESSAGE` with the closest sources and never constructs a Gemini client. At most
  one escalation and two checks happen per question, and the check is skipped entirely for a forced
  tier or when Jev is unavailable.

**Jev call count per `/ask`:** 0 (no key, or `router.provider: fixed`), 1 (an off-topic question, a
forced `--tier` (CLI and evals only) with a key, routing itself failed, or `router.answerability_check` is off), 2 (a
normal on-topic question: routing + one coverage check), or 3 (the coverage check escalates once, so
a second coverage check runs at Scholar depth). Routing + the coverage check together cost about
$0.0002 per on-topic question (measured by `eval/run_jev_gates_eval.py`).

### Game tagging

Most wiki pages have no `(XCn)` title suffix, so they are tagged `series` (recurring characters,
bosses, lore) and surface under every game filter. Pages that clearly belong to one or more games
carry those game tags. Cross-subseries cameo characters use a multi-tag membership schema so, for
example, KOS-MOS resolves to her home games rather than leaking into an unrelated filter.

## Web app

`xeno_rag/web/app.py` is a FastAPI app.

`/ask` streams the answer token by token over Server-Sent Events; a `sources` event carries the
deduped, relevance-scored source list. Generation is cancellable end to end: the browser drives the
fetch with an `AbortController` (the Ask button becomes a Stop control mid-stream and keeps the partial
answer), and the server checks `request.is_disconnected()` between chunks, so a stopped request also
stops pulling from the paid model stream.

`/health` is a cheap monitoring target: a directory stat, a read-only chunk count, and the BM25 row
count, returned as an always-200 JSON body that reports `degraded` instead of crashing when the store
or index is missing. An optional `XENO_WARM=1` startup hook loads the heavy retrieval singletons
(embedder, reranker, ChromaDB, BM25) at boot instead of inside the first question.

The static frontend (`web/static/index.html` + `render.js`) provides a question box, a game selector
that re-themes the page per game (palette, logo, display font, key-art wash), size-tiered source
bubbles, and client-side Markdown rendering. Inline `[n]` markers in an answer become superscript links
to the matching numbered source cards. The renderer is unit-tested with Node's test runner
(`tests/js/`), wrapped into the pytest suite so the browser-side logic is covered too.

## Module map

| Module | Responsibility |
|---|---|
| `api_client.py` | MediaWiki session: User-Agent, `maxlag`, serial requests, 429/maxlag backoff |
| `harvest_titles.py` | Enumerate all article titles |
| `fetch_html.py` / `fetch_content.py` | Pull rendered HTML / wikitext, checkpointed |
| `parse_html.py` / `parse_wikitext.py` | Hybrid parse to article records |
| `chunk.py` | Prose + infobox chunking with breadcrumbs |
| `embed_index.py` | Qwen3-Embedding vectors into ChromaDB (CPU; DirectML for encoder models) |
| `bm25_index.py` | SQLite FTS5 lexical index |
| `retrieve.py` | Dense + BM25 retrieval, RRF fusion, game filter |
| `rerank.py` | Cross-encoder reranking + relevance scores |
| `router.py` | Jev routing (`route`, `apply_tier`) and the shared `_jev_call`, fallback rules |
| `answerability.py` | Post-rerank Jev coverage check feeding `rag._ground`'s escalate/decline logic |
| `rag.py` | Retrieval query, prompt build, LLM adapter, tier routing/gates wiring |
| `config.py` | YAML config + `.env` loading |
| `pipeline.py` | Build orchestrator (harvest -> ... -> bm25) |
| `cli.py` | Command-line question interface |
| `web/app.py` | FastAPI + SSE server |

## Data and configuration

Everything tunable lives in `config.yaml`: API etiquette, embed model and device, vectorstore path,
retrieval depths, and the hybrid/rerank toggles. All build artifacts (`data/raw`, `data/processed`,
`data/vectorstore`) are gitignored; they are pulled or derived, never committed. Secrets are read from
`.env` (see `.env.example`): the Gemini API key, required for live answers, and the optional Jev
`TYPESAFE_API_KEY`, needed for tier routing and the off-topic/answerability gates (without it every
question uses the fallback tier with both gates off).

## Testing

Every module is unit-tested against fixtures and mocks, with no live API or LLM calls in the default
suite (`pytest -m 'not live'`). Fixtures cover wikitext and rendered-HTML parsing, chunking, embedding
into a temporary Chroma collection, hybrid retrieval, reranking, the grounded prompt, the CLI, and the
FastAPI endpoint. The frontend renderer has its own Node test suite, run from within pytest.
