"""Section-aware chunking + infobox-as-text rendering.

Two chunk types:
  - prose chunks: one per section (split over a token budget, with overlap), each prefixed with a
    self-describing breadcrumb "[GAME] Title > Heading: ...".
  - infobox chunks: structured template fields rendered into a natural-language sentence so they
    embed and retrieve well (a raw template dump does not retrieve; a sentence does).
"""

import json
import os

from .errors import SetupError
from .fileio import atomic_text_writer, require_input

DEFAULT_MAX_TOKENS = 600
DEFAULT_OVERLAP = 80

# Terse stat-block keys mapped to readable labels. Helps the embedder match natural-language questions
# ("what level...") and lets the LLM read the stat without decoding the abbreviation.
_STAT_LABELS = {
    "lv": "Level", "lvl": "Level", "hp": "HP", "str": "STR", "agi": "AGI", "eth": "Ether",
    "dex": "DEX", "luck": "Luck", "exp": "EXP", "ap": "AP", "sp": "SP", "atk": "Attack",
    "def": "Defense", "recharge": "Recharge", "power": "Power",
}


def _label(key: str) -> str:
    return _STAT_LABELS.get(key.strip().lower(), key.capitalize())


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
            parts.append(f"{_label(key)}: {val}.")
    return breadcrumb + " ".join(parts)


def chunk_params(cfg: dict) -> tuple[int, int]:
    """``(max_tokens, overlap)`` from the config, validated. An overlap >= the window size makes the
    window advance one word at a time (one chunk per word), which silently explodes the corpus."""
    max_tokens = cfg.get("chunk_max_tokens", DEFAULT_MAX_TOKENS)
    overlap = cfg.get("chunk_overlap_tokens", DEFAULT_OVERLAP)
    if not isinstance(max_tokens, int) or not isinstance(overlap, int) or max_tokens < 1 or not 0 <= overlap < max_tokens:
        raise SetupError(
            f"config.yaml chunking is unusable: chunk_max_tokens={max_tokens!r} and "
            f"chunk_overlap_tokens={overlap!r}. Need whole numbers with chunk_max_tokens >= 1 and "
            "0 <= chunk_overlap_tokens < chunk_max_tokens."
        )
    return max_tokens, overlap


def chunk_article(article: dict, cfg: dict):
    """Return the list of chunk records for one article."""
    max_tokens, overlap = chunk_params(cfg)
    game = article["game"]
    games = article.get("games")          # multi-tag membership (may be absent on legacy articles)
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
            "games": games,
            "heading": heading,
            "url": url,
            "text": text,
        })

    # Legacy wikitext path: template-field infoboxes.
    # A huge data template is split like any other text; later windows carry the breadcrumb too.
    for ib in article.get("infoboxes", []):
        infobox_crumb = f"[{game}] {title} > infobox: "
        for k, window in enumerate(split_with_overlap(_render_infobox(article, ib), max_tokens, overlap)):
            add("infobox", window if k == 0 else infobox_crumb + window)

    # Rendered-HTML path: fact tables already rendered to "Label: value." lines. A big table
    # (drops, skill lists) is split over the token budget so it doesn't form one giant chunk.
    for fb in article.get("factblocks", []):
        heading = fb.get("heading") or "infobox"
        breadcrumb = f"[{game}] {title} > {heading}: "
        for window in split_with_overlap(" ".join(fb["lines"]), max_tokens, overlap):
            add(heading, breadcrumb + window)

    for sec in article.get("sections", []):
        heading = sec["heading"]
        breadcrumb = f"[{game}] {title} > {heading}: "
        for window in split_with_overlap(sec["text"], max_tokens, overlap):
            add(heading, breadcrumb + window)

    return chunks


def _iter_articles(path: str):
    require_input(path, "Articles file", "python -m xeno_rag.pipeline parse")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(cfg: dict, articles=None) -> int:
    """Chunk all articles into chunks.jsonl. Returns the chunk count."""
    if articles is None:
        articles = _iter_articles(cfg["paths"]["articles"])
    out_path = cfg["paths"]["chunks"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0
    with atomic_text_writer(out_path) as out:
        for article in articles:
            for chunk in chunk_article(article, cfg):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
    return n
