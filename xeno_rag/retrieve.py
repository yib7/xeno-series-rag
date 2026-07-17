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
import threading

from . import embed_index
from .bm25_index import Bm25Index
from .parse_wikitext import filter_membership
from .rerank import Reranker

log = logging.getLogger(__name__)

_BM25_CACHE = {}
_RERANKER_CACHE = {}
# Guard the first build of each cached singleton (see embed_index): the sync /ask runs in a FastAPI
# threadpool, so two concurrent cold-start requests must not each construct a reranker / BM25 index.
# Double-checked locking builds exactly once; the already-cached fast path never takes the lock.
_BM25_LOCK = threading.Lock()
_RERANKER_LOCK = threading.Lock()


def rrf_fuse(rank_lists, k: int = 60):
    """Reciprocal Rank Fusion: score each id by sum of 1/(k + rank) across the ranked lists it
    appears in, then sort descending. Returns the fused list of ids (best first)."""
    scores = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def _get_bm25(cfg):
    """Open (and cache) the persisted BM25 index, or None if it hasn't been built yet.

    The cache entry is revalidated with a cheap ``os.stat`` per lookup (mtime + size): a rebuild
    ``os.replace``s a new file into place, and without the check a long-running server would keep
    serving the replaced (POSIX: deleted-inode) index through its old sqlite connection until
    restart. On a change the stale connection is closed and the file reopened."""
    path = cfg.get("paths", {}).get("bm25", os.path.join("data", "vectorstore", "bm25.sqlite3"))
    try:
        st = os.stat(path)
    except OSError:
        log.warning("BM25 index not found at %s; running dense-only. Build it with `pipeline bm25`.", path)
        return None
    sig = (st.st_mtime_ns, st.st_size)
    entry = _BM25_CACHE.get(path)
    if entry is None or entry[1] != sig:
        with _BM25_LOCK:
            entry = _BM25_CACHE.get(path)
            if entry is None or entry[1] != sig:
                if entry is not None:
                    log.info("BM25 index changed on disk (%s); reopening.", path)
                    try:
                        entry[0].close()
                    except Exception as exc:  # noqa: BLE001 - closing a stale handle is best-effort
                        log.warning("closing stale BM25 connection failed: %s", exc)
                entry = (Bm25Index(path=path), sig)
                _BM25_CACHE[path] = entry
    return entry[0]


def _get_reranker(cfg):
    name = cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if name not in _RERANKER_CACHE:
        with _RERANKER_LOCK:
            if name not in _RERANKER_CACHE:
                _RERANKER_CACHE[name] = Reranker(cfg)
    return _RERANKER_CACHE[name]


def _fuse_candidates(text, cfg, dense, game_filter, use_bm25, bm25, client, n_cand):
    """Fuse a dense candidate list with BM25 hits under ``game_filter`` via RRF (dense-only if BM25 is
    off/absent). Factored out of ``retrieve`` so the shared-cast fallback can re-fuse the *relaxed*
    (unfiltered) dense list through the identical path."""
    if use_bm25 and bm25 is not None:
        bm_ids = bm25.search(text, n=n_cand, game_filter=game_filter)
        dense_map = {d["chunk_id"]: d for d in dense}
        fused_ids = rrf_fuse([[d["chunk_id"] for d in dense], bm_ids], k=cfg.get("rrf_k", 60))
        extra = embed_index.fetch_chunks([c for c in fused_ids if c not in dense_map], cfg, client=client)
        candidates = [dense_map.get(c) or extra.get(c) for c in fused_ids]
        return [c for c in candidates if c]
    return dense


def _best_distance(dense):
    """Smallest (nearest) cosine distance in a dense candidate list, or ``None`` if it is empty / has
    no finite distances. ``dense_query`` returns nearest-first, but take the min defensively."""
    dists = [d.get("distance") for d in dense if d.get("distance") is not None]
    return min(dists) if dists else None


def _filter_starved(best_filtered, best_unfiltered, gap: float) -> bool:
    """Decide whether a hard per-game filter is *starving* the query — i.e. hiding the page it is
    actually about — so retrieval should relax the filter.

    The signal is the distance gap between the best in-filter candidate and the best UNFILTERED one.
    They are equal when the globally-nearest chunk is inside the filter (a well-populated filter ->
    gap ~0, DON'T relax, precision untouched); the gap only grows when the filter excludes a strictly
    closer, more relevant page — exactly the shared-cast miss (a Xenosaga character tagged for 2 of
    the 3 episodes, asked under the third: his page is dense rank 1 unfiltered but excluded by the
    hard episode filter). ``gap`` defaults to 0.10 in ``retrieve`` (see the call site for the rationale
    and the measured separation). Also relax when the filter returned nothing at all but the
    unfiltered query did."""
    if best_filtered is None:
        return best_unfiltered is not None       # filter matched nothing; unfiltered has candidates
    if best_unfiltered is None:
        return False
    return (best_filtered - best_unfiltered) >= gap


def retrieve(text: str, cfg: dict, k: int = None, game_filter: str = None, embedder=None,
             client=None, bm25=None, reranker=None):
    """Retrieve the top-k chunks for ``text`` via dense + BM25 fusion (+ optional rerank), page-capped.

    Returns a list of result dicts (same shape as ``embed_index.query``)."""
    if k is None:
        k = cfg.get("top_k", 8)
    per_page_cap = cfg.get("max_chunks_per_page", 2)
    n_cand = cfg.get("hybrid_candidates", 60)

    use_bm25 = cfg.get("use_bm25", True)
    if use_bm25 and bm25 is None:
        bm25 = _get_bm25(cfg)

    # Embed the query text exactly once and reuse the vector for every dense query below (the filtered
    # pass and the fallback's unfiltered pass differ only in their ``where`` clause), so a filtered
    # request never re-embeds the same Qwen query.
    emb = embedder if embedder is not None else embed_index._get_embedder(cfg)
    qemb = emb.embed_query(text)

    dense = embed_index.dense_query(text, cfg, n=n_cand, game_filter=game_filter,
                                    embedder=emb, client=client, query_embedding=qemb)
    candidates = _fuse_candidates(text, cfg, dense, game_filter, use_bm25, bm25, client, n_cand)

    # Shared-cast filter fallback. A hard per-game filter (``where={"g_<game>": True}``) can exclude
    # the very page a question is about: a Xenosaga character tagged for only 2 of the 3 episodes,
    # asked under the missing episode, is dropped entirely though he is dense rank 1 unfiltered
    # ("Who is Joachim Mizrahi?" under XS2 — his membership is {XS1,XS3}). Detect this by comparing
    # the best in-filter dense distance to the best UNFILTERED one: they match when the globally
    # nearest page is inside the filter (well-populated -> gap ~0, no relaxation, precision intact),
    # and diverge only when the filter hides a closer page. When the gap clears ``retrieve_relax_gap``
    # (or the filter matched nothing), relax to unfiltered so the excluded page can surface; the
    # reranker then re-sorts by query relevance. Threshold 0.10: live cosine-distance gaps measured
    # 0.122 for the starved Mizrahi/XS2 case vs 0.000-0.057 for well-populated filters, so 0.10 sits
    # between them and only fires on a genuine exclusion. Only runs when a real base-game filter is
    # active (``filter_membership`` is None for no filter / the 'series'/'XS' display labels), adding
    # at most one extra HNSW search (the query vector is embedded once above and reused) on filtered
    # requests — cheap next to the cross-encoder.
    if filter_membership(game_filter) is not None:
        gap = cfg.get("retrieve_relax_gap", 0.10)
        dense_unf = embed_index.dense_query(text, cfg, n=n_cand, game_filter=None,
                                            embedder=emb, client=client, query_embedding=qemb)
        if _filter_starved(_best_distance(dense), _best_distance(dense_unf), gap):
            log.info("retrieve: per-game filter %r starved (relaxing to unfiltered)", game_filter)
            candidates = _fuse_candidates(text, cfg, dense_unf, None, use_bm25, bm25, client, n_cand)

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
    (pageid -> sibling dicts) is injectable for tests; by default the siblings of ALL distinct
    retrieved pages come from the live collection in ONE batched ``$in`` query — the per-page
    ``collection.get`` was an N+1 metadata scan on the hot path (up to ``top_k`` sequential scans
    of a ~169k-row store per question, worst on the high-``top_k`` Scholar tier).
    """
    if not cfg.get("merge_stat_pages", True):
        return chunks
    small_chars = cfg.get("merge_small_chars", 400)
    min_small = cfg.get("merge_min_small", 3)
    max_chars = cfg.get("merge_max_chars", 4000)
    if fetch_fn is None:
        try:
            sib_map = embed_index.fetch_pages_chunks(
                [c.get("pageid") for c in chunks], cfg, client=client)
        except Exception as exc:  # noqa: BLE001 - a fetch hiccup must not break answering
            log.warning("merge_fragmented_pages batched sibling fetch failed: %s", exc)
            sib_map = {}

        def fetch_fn(pid):
            return sib_map.get(pid, [])

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
