# Build the corpus from scratch

You do not need this to run the app: README Step 4 downloads the prebuilt vector store, which is far
faster. Use this page only to regenerate the data yourself.

The full pull hits the live wiki for ~36k articles, and the stat pages are fetched as rendered HTML one
page per request (throttled to the wiki's `Crawl-delay: 5`), so a from-scratch build takes several
hours. It is fully resumable, so run it deliberately. Before any live pull, set your own contact in the
`User-Agent` in `config.yaml`, as a courtesy to the wiki.

Run these with the virtual environment from the README setup activated:

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

The embed step is the slow one on a CPU (hours); the shipped store was embedded once on a GPU.

To re-process a corpus you have already fetched (new parse, chunk or embed logic, no new network
pulls), run `python -m xeno_rag.pipeline rebuild`. It re-parses, re-chunks, re-embeds into a fresh
collection and rebuilds the BM25 index.
