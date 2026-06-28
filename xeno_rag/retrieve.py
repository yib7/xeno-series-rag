"""Hybrid retrieval: fuse dense (Qwen3-Embedding) + lexical (BM25) candidates, optionally rerank, diversify.

Why: dense embeddings generalize but miss exact proper nouns / rare concept terms when near-duplicate
ancillary pages crowd the candidate window (the "mimeosomes -> only Skell weapon SKUs" failure). BM25
nails exact terms; the two are merged with Reciprocal Rank Fusion (rank-based, scale-free, robust),
then an optional cross-encoder reranks the fused set for final precision. The per-page cap from the
dense path still applies last so one long page can't dominate the answer context.

Degrades gracefully: if the BM25 index isn't built (or ``use_bm25`` is off) it runs dense-only; the
reranker is likewise gated by ``use_reranker``. Both components are injectable for tests.
"""

import logging
import os

from . import embed_index
from .bm25_index import Bm25Index
from .rerank import Reranker

log = logging.getLogger(__name__)

_BM25_CACHE = {}
_RERANKER_CACHE = {}


def rrf_fuse(rank_lists, k: int = 60):
    """Reciprocal Rank Fusion: score each id by sum of 1/(k + rank) across the ranked lists it
    appears in, then sort descending. Returns the fused list of ids (best first)."""
    scores = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def _get_bm25(cfg):
    """Open (and cache) the persisted BM25 index, or None if it hasn't been built yet."""
    path = cfg.get("paths", {}).get("bm25", os.path.join("data", "vectorstore", "bm25.sqlite3"))
    if not os.path.exists(path):
        log.warning("BM25 index not found at %s; running dense-only. Build it with `pipeline bm25`.", path)
        return None
    if path not in _BM25_CACHE:
        _BM25_CACHE[path] = Bm25Index(path=path)
    return _BM25_CACHE[path]


def _get_reranker(cfg):
    name = cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if name not in _RERANKER_CACHE:
        _RERANKER_CACHE[name] = Reranker(cfg)
    return _RERANKER_CACHE[name]


def retrieve(text: str, cfg: dict, k: int = None, game_filter: str = None, embedder=None,
             client=None, bm25=None, reranker=None):
    """Retrieve the top-k chunks for ``text`` via dense + BM25 fusion (+ optional rerank), page-capped.

    Returns a list of result dicts (same shape as ``embed_index.query``)."""
    if k is None:
        k = cfg.get("top_k", 8)
    per_page_cap = cfg.get("max_chunks_per_page", 2)
    n_cand = cfg.get("hybrid_candidates", 60)

    dense = embed_index.dense_query(text, cfg, n=n_cand, game_filter=game_filter,
                                    embedder=embedder, client=client)

    use_bm25 = cfg.get("use_bm25", True)
    if use_bm25 and bm25 is None:
        bm25 = _get_bm25(cfg)
    if use_bm25 and bm25 is not None:
        bm_ids = bm25.search(text, n=n_cand, game_filter=game_filter)
        dense_map = {d["chunk_id"]: d for d in dense}
        fused_ids = rrf_fuse([[d["chunk_id"] for d in dense], bm_ids], k=cfg.get("rrf_k", 60))
        extra = embed_index.fetch_chunks([c for c in fused_ids if c not in dense_map], cfg, client=client)
        candidates = [dense_map.get(c) or extra.get(c) for c in fused_ids]
        candidates = [c for c in candidates if c]
    else:
        candidates = dense

    if cfg.get("use_reranker", True):
        if reranker is None:
            reranker = _get_reranker(cfg)
        candidates = reranker.rerank(text, candidates[:cfg.get("rerank_candidates", 50)])

    return embed_index.cap_per_page(candidates, k, per_page_cap)


def _strip_breadcrumb(text: str) -> str:
    """Drop the "[GAME] Title > " prefix, keeping "Heading: content" so each merged line stays
    self-labelled (infobox / Enemy / Drops)."""
    t = text or ""
    return t.split(" > ", 1)[-1].strip() if " > " in t else t.strip()


def merge_fragmented_pages(chunks, cfg: dict, fetch_fn=None, client=None):
    """Auto-merge: replace a retrieved *stat page*'s fragmented factblock chunks with ONE block
    carrying the page's full profile (infobox + stats + resistances + drops), pulled from its
    siblings — so the LLM sees a coherent enemy/item profile instead of 30-token scraps.

    A page is "stat-fragmented" when it has at least ``merge_min_small`` sibling chunks shorter than
    ``merge_small_chars`` (the bimodal stat-page signature). Prose pages (few, large chunks) pass
    through untouched. The merged body is capped at ``merge_max_chars``. Retrieval granularity is
    unchanged — this only enriches what is sent to the model — so no re-embed is needed. ``fetch_fn``
    (pageid -> sibling dicts) is injectable for tests; it defaults to the live collection.
    """
    if not cfg.get("merge_stat_pages", True):
        return chunks
    small_chars = cfg.get("merge_small_chars", 400)
    min_small = cfg.get("merge_min_small", 3)
    max_chars = cfg.get("merge_max_chars", 4000)
    if fetch_fn is None:
        def fetch_fn(pid):
            return embed_index.fetch_page_chunks(pid, cfg, client=client)

    out = []
    decided = {}  # pageid -> merged block (stat page, emit once) or None (prose, keep each chunk)
    for c in chunks:
        pid = c.get("pageid")
        if pid in decided:
            if decided[pid] is None:
                out.append(c)            # prose page: keep every retrieved chunk
            continue                     # stat page already emitted its single block -> skip
        try:
            sibs = fetch_fn(pid) or []
        except Exception as exc:  # noqa: BLE001 - a fetch hiccup must not break answering
            log.warning("merge_fragmented_pages fetch failed for page %s: %s", pid, exc)
            sibs = []
        n_small = sum(1 for s in sibs if len(s.get("text", "")) < small_chars)
        if sibs and n_small >= min_small:
            body, total = [], 0
            for s in sorted(sibs, key=lambda s: s.get("chunk_id", "")):
                line = _strip_breadcrumb(s.get("text", ""))
                if body and total + len(line) + 1 > max_chars:
                    break
                body.append(line)
                total += len(line) + 1
            header = f"[{c.get('game')}] {c.get('title')} > combined: "
            block = {**c, "chunk_id": f"{pid}-merged", "heading": "combined",
                     "text": header + "\n".join(body)}
            decided[pid] = block
            out.append(block)
        else:
            decided[pid] = None
            out.append(c)
    return out
