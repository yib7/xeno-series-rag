"""Section-aware chunking + infobox-as-text rendering.

Two chunk types:
  - prose chunks: one per section (split over a token budget, with overlap), each prefixed with a
    self-describing breadcrumb "[GAME] Title > Heading: ...".
  - infobox chunks: structured template fields rendered into a natural-language sentence so they
    embed and retrieve well (a raw template dump does not retrieve; a sentence does).
"""

import json
import os

DEFAULT_MAX_TOKENS = 600
DEFAULT_OVERLAP = 80


def split_with_overlap(text: str, max_tokens: int, overlap: int):
    """Split text into overlapping word-windows (whitespace approximates tokens)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [text.strip()]
    step = max(1, max_tokens - overlap)
    windows = []
    i = 0
    while i < len(words):
        windows.append(" ".join(words[i:i + max_tokens]))
        if i + max_tokens >= len(words):
            break
        i += step
    return windows


def _render_infobox(article: dict, ib: dict) -> str:
    breadcrumb = f"[{article['game']}] {article['title']} > infobox: "
    parts = [f"{ib['template']}."]
    for key, val in ib["fields"].items():
        if str(val).strip():
            parts.append(f"{key.capitalize()}: {val}.")
    return breadcrumb + " ".join(parts)


def chunk_article(article: dict, cfg: dict):
    """Return the list of chunk records for one article."""
    max_tokens = cfg.get("chunk_max_tokens", DEFAULT_MAX_TOKENS)
    overlap = cfg.get("chunk_overlap_tokens", DEFAULT_OVERLAP)
    game = article["game"]
    title = article["title"]
    pageid = article["pageid"]
    url = article["url"]

    chunks = []

    def add(heading, text):
        text = text.strip()
        if not text:
            return
        chunks.append({
            "chunk_id": f"{pageid}-{len(chunks):04d}",
            "pageid": pageid,
            "title": title,
            "game": game,
            "heading": heading,
            "url": url,
            "text": text,
        })

    for ib in article.get("infoboxes", []):
        add("infobox", _render_infobox(article, ib))

    for sec in article.get("sections", []):
        heading = sec["heading"]
        breadcrumb = f"[{game}] {title} > {heading}: "
        for window in split_with_overlap(sec["text"], max_tokens, overlap):
            add(heading, breadcrumb + window)

    return chunks


def _iter_articles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(cfg: dict, articles=None) -> int:
    """Chunk all articles → chunks.jsonl. Returns the chunk count."""
    if articles is None:
        articles = _iter_articles(cfg["paths"]["articles"])
    out_path = cfg["paths"]["chunks"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for article in articles:
            for chunk in chunk_article(article, cfg):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
    return n
