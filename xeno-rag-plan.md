# Xeno Series Wiki RAG Chatbot: Build Plan

A plan for building a Retrieval-Augmented Generation chatbot over the full
Xeno Series Wiki (xenoserieswiki.org), covering all Xeno games (Xenogears,
Xenosaga 1-3, Xenoblade Chronicles 1/2/3/X and their expansions).

This document is written to drive a Claude Code session. Work through it phase
by phase. Each phase has a clear deliverable and acceptance check.

---

## 0. Goal and scope

Build a local-first RAG system that can answer natural-language questions about
Xeno game mechanics, characters, items, locations, and lore, grounded in wiki
content with source attribution.

**Scope decisions (locked):**

- All games, all namespaces of interest (main article namespace `ns=0` is the
  priority; optionally pull `Category` pages for structure).
- Data source: the wiki's MediaWiki API, not an HTML scraper.
- Local-first: scrape once, cache to disk, iterate offline. Never re-hit the
  server to tweak parsing or chunking.

**Known scale (verified):**

- ~36,144 articles in `ns=0`, ~215 MB of raw wikitext.
- API batching (50 titles per request) makes a full content pull roughly 1 hour
  at a gentle request cadence. A naive page-by-page scrape would take ~50 hours,
  so the API path is mandatory.

**Architectural note (ties to your build generator):**

The structured data you extract from infoboxes here (art power/recharge, class
stats, blade data, accessory properties) is the same data layer your planned
multi-game build generator needs. Design the parser (Phase 3) so its structured
output is reusable by both projects. The shared-engine / per-game-adapter shape
you already sketched maps cleanly onto the per-game template parsing here.

---

## 1. Constraints and API etiquette (read before writing code)

This is a small, donation-funded fan wiki run by a handful of admins. Be a good
citizen. The technical posture below is both the respectful choice and the one
least likely to get you rate-limited or blocked.

**Licensing:** Content is CC-BY-SA. You may use, transform, and redistribute it,
provided you (a) attribute the wiki and (b) license derivative content under
CC-BY-SA. Surface a source link with every answer the chatbot gives, and put an
attribution + license notice in your README.

**robots.txt reality check:** `robots.txt` sets `Crawl-delay: 5` and disallows
`/w/` (which technically includes `/w/api.php`). robots.txt is a crawler-indexing
directive, not an access-control or terms mechanism, and the MediaWiki API is the
officially sanctioned way to bulk-read a MediaWiki site. The genuinely respectful
read: use the API, identify yourself, throttle, and honor backoff signals. If you
want to be maximally courteous before pulling the entire wiki, drop a one-line
heads-up in their Discord (linked on the main page). It is a 3-admin community and
they are approachable.

**Mandatory API etiquette:**

1. Set a descriptive `User-Agent` with project name and a contact. MediaWiki
   etiquette requires this and a generic UA can get you blocked.
   Example: `XenoRAG/0.1 (https://github.com/<you>/xeno-rag; <contact-email>)`
2. Send `maxlag=5` on every API call. If the server is under load it returns a
   lag error; on that error, sleep and retry. This lets the site shed your load
   automatically.
3. Serial requests only. No concurrency, no thread pools against the API.
4. Honor `Retry-After` headers and HTTP 429. Back off exponentially.
5. Keep a delay between requests. Default to 2 seconds with `maxlag` active as a
   reasonable middle ground for the efficient API. Bump to 5 seconds if you want
   to match the robots.txt crawl-delay exactly. Make it a config value.

**Resumability:** A full pull is thousands of requests. The fetcher must
checkpoint progress and resume after an interruption without re-fetching what it
already has.

---

## 2. Tech stack

Defaults chosen for a clean portfolio piece that runs on your hardware with no
mandatory cloud cost. Alternatives noted where your existing credits/tools apply.

| Layer | Default | Alternatives / notes |
|---|---|---|
| Language | Python 3.11+ | |
| HTTP | `requests` + a small retry wrapper | `httpx` if you prefer |
| Wikitext parsing | `mwparserfromhell` | The standard lib for parsing templates/sections |
| Chunking | custom, section-aware | LangChain splitters if you want the abstraction |
| Embeddings | `sentence-transformers`, `BAAI/bge-base-en-v1.5` (local, free) | Vertex AI text-embeddings (spend GCP credits), Voyage, OpenAI |
| Vector store | ChromaDB (local, file-based) | Qdrant via Docker for a more production feel (you already use Docker) |
| Generation LLM | model-agnostic; default Claude (Haiku for cheap, Sonnet for quality) | Gemini via Vertex (GCP credits) |
| Interface | CLI first, then a thin web UI | FastAPI + a simple frontend, or reuse a CodeCaster-style SSE stream |

Local embeddings are recommended: 36k articles becomes on the order of 100k+
chunks, and embedding that locally on your GPU is free and rate-limit-free.
Reserve cloud embeddings for if you want multilingual coverage later.

---

## 3. Repository structure

```
xeno-rag/
  README.md                  # includes CC-BY-SA attribution + license notice
  pyproject.toml             # or requirements.txt
  config.yaml                # delay, user-agent, batch size, paths, model names
  data/
    raw/                     # cached API responses (gitignored)
      titles.jsonl           # all ns=0 titles + pageids
      pages/                 # batched wikitext dumps, e.g. pages_00001.jsonl
      checkpoint.json        # resume state
    processed/
      articles.jsonl         # one record per article: prose + structured fields
      chunks.jsonl           # final chunks ready for embedding
    vectorstore/             # Chroma persistent dir (gitignored)
  src/
    api_client.py            # session, user-agent, maxlag, retry/backoff
    harvest_titles.py        # Phase 4: list all ns=0 titles
    fetch_content.py         # Phase 5: batched content pull, resumable
    parse_wikitext.py        # Phase 6: wikitext -> {prose, infoboxes, sections}
    chunk.py                 # Phase 7: section-aware chunking + infobox-to-text
    embed_index.py           # Phase 8: embed + write to vector store
    rag.py                   # Phase 9: retrieve + generate
    cli.py                   # Phase 10: ask questions from terminal
  tests/
```

`.gitignore` must exclude `data/raw/`, `data/vectorstore/`, and any API keys.
The whole point of caching raw data is so you pull from the server exactly once.

---

## 4. Phase 0: Setup and reconnaissance

**Deliverable:** project scaffold, config, and a confirmed data path.

1. Scaffold the repo above. Add `mwparserfromhell`, `requests`,
   `sentence-transformers`, `chromadb`, and your chosen LLM SDK.
2. Reconnaissance, in order of preference:
   - **Check for an XML dump first.** If the wiki publishes a database/XML dump
     (look on `Special:Statistics`, any `Special:Export` full-export option, or a
     dumps directory), that is the single cleanest acquisition: one download, no
     rate limiting, everything at once. If a dump exists, skip Phases 4-5 and
     parse the dump instead.
   - If no dump, confirm the live API works:
     ```bash
     curl -s "https://www.xenoserieswiki.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"
     ```
     You should see the article count (~36k). This confirms the API path.
3. Write `config.yaml` with: `base_url`, `user_agent`, `request_delay_seconds`,
   `batch_size` (start at 50), `maxlag` (5), and all the `data/` paths.

**Acceptance:** API responds with stats, config loads, scaffold imports cleanly.

---

## 5. Phase 1: API client with etiquette baked in

**Deliverable:** `api_client.py` exposing one `get(params)` function that every
later phase uses.

Requirements:

- A persistent `requests.Session` with the descriptive `User-Agent` header set.
- Inject `format=json`, `formatversion=2`, and `maxlag=5` into every request.
- Retry wrapper: on HTTP 429 or a `maxlag` error in the JSON, read `Retry-After`
  (or default to an exponential backoff starting at a few seconds) and retry up
  to N times. On hard failure, raise so the caller can checkpoint and stop.
- Sleep `request_delay_seconds` after each successful request.

Skeleton:

```python
import time
import requests

class WikiClient:
    def __init__(self, cfg):
        self.base = cfg["base_url"]            # https://www.xenoserieswiki.org/w/api.php
        self.delay = cfg["request_delay_seconds"]
        self.maxlag = cfg["maxlag"]
        self.s = requests.Session()
        self.s.headers["User-Agent"] = cfg["user_agent"]

    def get(self, params, max_retries=6):
        params = {**params, "format": "json", "formatversion": 2, "maxlag": self.maxlag}
        backoff = 3
        for attempt in range(max_retries):
            r = self.s.get(self.base, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", backoff))
                time.sleep(wait); backoff *= 2; continue
            data = r.json()
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                time.sleep(backoff); backoff *= 2; continue
            time.sleep(self.delay)
            return data
        raise RuntimeError("API retries exhausted")
```

**Acceptance:** a single test call returns parsed JSON and the client sleeps
between calls.

---

## 6. Phase 2: Harvest all titles

**Deliverable:** `data/raw/titles.jsonl`, one record per article with `title`
and `pageid`.

Use `list=allpages` over namespace 0, paginating with `apcontinue`:

```python
def harvest(client):
    params = {"action": "query", "list": "allpages",
              "apnamespace": 0, "aplimit": "max"}  # aplimit=max -> 500 for non-bots
    while True:
        data = client.get(params)
        for p in data["query"]["allpages"]:
            yield {"title": p["title"], "pageid": p["pageid"]}
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
```

Notes:

- `aplimit=max` returns up to 500 titles per request for normal users, so this is
  roughly 75 requests for 36k titles. Fast, a few minutes.
- Optionally set `apfilterredir=nonredirects` to skip redirect pages, which are
  noise for RAG. You can also keep redirects separately if you later want them as
  alias hints for retrieval.

**Acceptance:** `titles.jsonl` line count is in the tens of thousands and matches
the order of magnitude of the article count.

---

## 7. Phase 3: Fetch content (batched and resumable)

**Deliverable:** `data/raw/pages/*.jsonl` containing raw wikitext for every
article, plus a working resume checkpoint.

Strategy:

- Read titles in batches of 50. Fetch wikitext with:
  ```
  action=query&prop=revisions&rvprop=content&rvslots=main&titles=A|B|C...
  ```
- Write each batch to its own file (`pages_00001.jsonl`, etc.) so a crash never
  corrupts a big single file.
- After each successful batch, update `checkpoint.json` with the last completed
  batch index. On startup, resume from there.

Why wikitext and not plain-text extracts: the wiki encodes the valuable
structured data (art stats, class data, item properties) inside infobox
templates. Plain-text extraction throws that away. You want the raw wikitext so
Phase 6 can pull template parameters into structured fields. This is the single
most important quality decision in the project.

Skeleton:

```python
def batched(iterable, n=50):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf:
        yield buf

def fetch_all(client, titles, start_batch=0):
    for i, batch in enumerate(batched(titles)):
        if i < start_batch:
            continue
        titles_param = "|".join(t["title"] for t in batch)
        data = client.get({"action": "query", "prop": "revisions",
                            "rvprop": "content", "rvslots": "main",
                            "titles": titles_param})
        write_batch(i, data["query"]["pages"])
        save_checkpoint(i)
```

**Acceptance:** all batches complete, checkpoint reaches the final batch, and a
forced mid-run kill followed by restart resumes correctly without re-fetching.

Run this once. From here on, never touch the server again unless content changes.

---

## 8. Phase 4: Parse wikitext into prose + structured data

**Deliverable:** `data/processed/articles.jsonl`, one record per article:

```json
{
  "title": "Infinity Blade (XC3) (Noah)",
  "pageid": 70047,
  "game": "XC3",
  "url": "https://www.xenoserieswiki.org/wiki/Infinity_Blade_(XC3)_(Noah)",
  "infoboxes": [
    {"template": "Infobox XC3 art", "fields": {"power": "...", "recharge": "...", "type": "Talent Art"}}
  ],
  "sections": [
    {"heading": "Introduction", "text": "..."},
    {"heading": "Mechanics", "text": "..."}
  ]
}
```

Use `mwparserfromhell`:

- Parse each page's wikitext.
- Extract every `{{Infobox ...}}` (and other data templates) into
  `{template_name, {param: value}}`. Strip nested markup from values.
- Split prose by `==` section headings into `{heading, text}`.
- Strip remaining markup (links, refs, formatting) from prose text. Resolve
  `[[Link|display]]` to `display`.
- Derive `game` from the title suffix convention this wiki uses, for example
  `(XC3)`, `(XC2)`, `(XS1)`, `(XG)`, `(XCX)`. Build a small mapping. Pages without
  a suffix are series-level or cross-game; tag them `series`.
- Build the canonical URL from the title:
  `https://www.xenoserieswiki.org/wiki/` + title with spaces as underscores.

Skip or flag: pages that are pure redirects, disambiguation pages, and near-empty
stubs (for example wikitext under ~50 bytes). Keep a count of what you drop.

**Acceptance:** spot-check 10 varied pages (a character, an art, a class, a
location, a lore page). Infobox fields are captured, prose is clean, `game` is
correct, URL resolves.

---

## 9. Phase 5: Chunking (section-aware, infobox-as-text)

**Deliverable:** `data/processed/chunks.jsonl`, the retrieval units.

Two chunk types:

1. **Prose chunks.** One chunk per section, split further if a section exceeds
   your token budget. Target ~500 to 800 tokens with ~80 token overlap on splits.
   Prepend a breadcrumb so the chunk is self-describing:
   `"[XC3] Infinity Blade (Noah) > Mechanics: <text>"`.

2. **Infobox chunks.** Render structured fields into natural-language sentences so
   they embed and retrieve well. A raw template dump does not retrieve; a sentence
   does. Example:
   `"[XC3] Infinity Blade is a Talent Art for Noah. Power: X. Recharge: Y. Type: Talent Art."`
   This step is what makes the bot able to answer stat and mechanics questions,
   and it is also the exact text-rendering you can reuse for the build generator.

Every chunk record carries metadata: `chunk_id`, `pageid`, `title`, `game`,
`heading` (or `"infobox"`), and `url`. Metadata travels into the vector store so
you can filter (for example, restrict retrieval to `game == "XC3"`) and so you can
cite the source URL in answers.

**Acceptance:** chunk count is reasonable (likely 100k+), no chunk is empty, every
chunk has a resolvable `url` and a `game` tag, and a sample infobox chunk reads as
a clean sentence.

---

## 10. Phase 6: Embed and index

**Deliverable:** a populated, persistent vector store in `data/vectorstore/`.

- Load `BAAI/bge-base-en-v1.5` via `sentence-transformers`. For BGE models,
  prepend the recommended query instruction to queries at search time (not to
  the stored documents). Check the model card for the exact instruction string.
- Batch-encode chunks on the GPU. Show a progress bar; this is the longest local
  compute step but it is one-time.
- Write vectors + metadata + chunk text to ChromaDB with a persistent client.
  Use cosine space.

If you instead spend GCP credits: swap the encoder for Vertex AI text-embeddings,
keep batch sizes within the endpoint's per-request limits, and add the same
retry/backoff discipline you used for the wiki API.

**Acceptance:** the collection count equals your chunk count; a manual nearest-
neighbor query for a known topic returns on-topic chunks with correct metadata.

---

## 11. Phase 7: Retrieval + generation

**Deliverable:** `rag.py` with an `answer(question, game_filter=None)` function.

Pipeline:

1. Embed the question (with the BGE query instruction if using BGE).
2. Retrieve top-k chunks (start k=8). If `game_filter` is set, filter on the
   `game` metadata field so an XC3 question does not pull XC2 chunks.
3. Optional reranking for quality: a cross-encoder reranker (for example
   `BAAI/bge-reranker-base`) over the top ~30 candidates, then keep the best 8.
   Adds latency, improves precision. Make it a toggle.
4. Build a prompt: system instruction to answer only from the provided context,
   to say when the context does not cover the question, and to cite the source
   titles/URLs it used. Pass the retrieved chunks as context.
5. Call the LLM (Claude Haiku/Sonnet, or Gemini). Return the answer plus the list
   of source URLs from the chunks that were used.

Grounding rules to put in the system prompt: do not invent mechanics or numbers,
prefer the structured infobox chunks for stat questions, and surface the source
links so the CC-BY-SA attribution requirement is satisfied in the output itself.

**Acceptance:** ask 10 questions spanning lore, a specific art's stats, a class
comparison, and a cross-game character-name question (there are several recurring
"Vandham"-type names across games, which is a good retrieval stress test). Answers
are grounded and cite sources.

---

## 12. Phase 8: Interface

**Deliverable:** something you can demo and put in the portfolio.

- Start with `cli.py`: read a question from stdin, print the answer and sources.
- Then a thin web UI. Given your CodeCaster work, a FastAPI backend with an SSE
  token stream and a minimal frontend is a natural fit and lets you reuse
  patterns you already have. Add a game selector that sets the `game_filter`.

**Acceptance:** a non-technical person can ask a question and get a sourced answer
without touching the terminal.

---

## 13. Time and cost estimate

- Title harvest: a few minutes.
- Content pull: ~1 hour at a 2s delay (one time). Longer at 5s.
- Parsing + chunking: minutes, all local, re-runnable for free.
- Embedding: tens of minutes on GPU, one time.
- Cost: $0 if you embed locally and use a Claude/Gemini plan you already have.
  Cloud embeddings would be a few dollars at most for this corpus.

The expensive resource is the wiki's goodwill, not compute. Pull once, cache,
attribute, and throttle.

---

## 14. Stretch goals

- Hybrid retrieval: combine dense (embeddings) with BM25 keyword search. Wikis
  have many exact proper nouns where keyword matching helps.
- Share the Phase 4 structured output with the build generator as a common data
  module (the per-game adapter pattern you already planned).
- Periodic refresh: re-pull only pages whose `revisions` timestamp changed since
  your last run, using `rvprop=timestamp`. Keeps the index current without a full
  re-scrape.
- Evaluation set: write 30 to 50 question/answer pairs and measure retrieval hit
  rate and answer grounding. Good portfolio signal, and reuses the eval discipline
  from your other projects.

---

## 15. README must include

- One-line description and a screenshot or demo gif.
- Attribution: "Content from the Xeno Series Wiki (xenoserieswiki.org), licensed
  under CC-BY-SA. This project and its derived content are likewise CC-BY-SA."
- A note that data was pulled via the MediaWiki API with rate limiting and a
  descriptive User-Agent.
- Setup and run instructions.
