# Credits and attribution

The code in this repository is original and MIT-licensed. It builds on the third-party data, fonts,
models, and libraries listed below. Each is used under a license that permits redistribution, with the
attribution recorded here.

## Data

- **Xeno Series Wiki** (https://www.xenoserieswiki.org). The corpus, the shipped vector-store release
  asset, and the wiki text/HTML fixtures under `tests/fixtures/` are derived from wiki content, which is
  licensed CC BY-SA 4.0. Anything derived from it stays under the same license; see
  [LICENSE-DATA.md](LICENSE-DATA.md). Every answer surfaces the source page URLs it used. The README
  screenshots and demo GIF show answer and page text derived from the wiki.

## Fonts

Both are bundled as Latin-subset woff2 under the SIL Open Font License 1.1 (full texts in
`xeno_rag/web/static/fonts/`).

- **Cinzel** (variable): Copyright 2020 The Cinzel Project Authors
  (https://github.com/NDISCOVER/Cinzel). Used for UI chrome.
- **Spectral**: Copyright 2017 The Spectral Project Authors
  (https://github.com/productiontype/Spectral). Used for chat and generated text.

## Models

- **Qwen3-Embedding-0.6B** (Alibaba, Apache 2.0): dense query and document embeddings.
- **cross-encoder/ms-marco-MiniLM-L-6-v2** (Apache 2.0): candidate reranking.
- **Google Gemini**: answer generation, called as a hosted API behind a provider-agnostic adapter. No
  model weights are bundled.
- **Jev** (TypeSafe AI, https://typesafe.ai): the decision model that routes each question to an answer
  tier and gates off-topic or unanswerable questions, called as an optional hosted API with your own
  key. No model weights or TypeSafe AI code are bundled.

## Key libraries

Runtime dependencies, as declared in `pyproject.toml`: sentence-transformers and transformers
(embedding and reranking), ChromaDB (vector store), NumPy, SQLite FTS5 (lexical BM25, part of the
Python standard library), FastAPI and uvicorn (web server and SSE), google-genai (Gemini adapter),
httpx (HTTP client for the Jev calls), requests (MediaWiki API client), mwparserfromhell (wikitext
parsing), BeautifulSoup with lxml (rendered-HTML parsing), and PyYAML (config). The optional `[gpu]`
extra adds Optimum and ONNX Runtime DirectML. Development tooling is pytest, pytest-mock and ruff.
All are installed from PyPI under their own licenses (MIT, BSD, Apache 2.0, MPL 2.0 or similar) and
none is vendored in this repository. Full pinned versions are in [requirements.txt](../requirements.txt).

## Game artwork, logos, and trademarks

The Zohar glyph and the starfield background in the UI are original inline SVG drawings, not copied
from any game asset.

No official game artwork, logo, key art, or box art file is included in this repository. Those assets
are the property of their respective owners (Nintendo, Monolith Soft, Bandai Namco Entertainment, and
Square Enix), and all rights are reserved to them. The per-game logo and key-art files the UI can display
are fetched locally by `scripts/fetch_art.py`, are gitignored, and are never redistributed here; the UI
falls back to styled text wordmarks when they are absent. The one place official artwork appears is the
README screenshots and demo GIF (`docs/`), which capture the running UI and show the Xenoblade Chronicles
2 logo and background as the app rendered them locally. They are included to illustrate the interface
only. This is an unofficial, non-commercial fan project, not affiliated with or endorsed by any of
those rights holders.
