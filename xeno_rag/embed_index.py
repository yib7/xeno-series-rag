"""Embed chunks and index them in ChromaDB.

The production embedder is ``Qwen/Qwen3-Embedding-0.6B``, a decoder with last-token pooling, so it
has no reliable DirectML/ONNX path: set ``embed_device: cpu`` (a single query still embeds on CPU in
well under a second). For an encoder embedder the Embedder instead prefers DirectML (AMD GPU on
Windows) via the ONNX backend and falls back to CPU. Either family wants a query instruction
prepended to *queries* only, not to stored documents; pass any per-model load kwargs via
``embed_model_kwargs`` / ``embed_tokenizer_kwargs`` (Qwen needs left-padded tokenization for its
last-token pooling).
"""

import json
import logging
import threading

import chromadb
import numpy as np
from chromadb.config import Settings

from .parse_wikitext import filter_membership, membership_flags, membership_from_game

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

# Process-wide caches so the heavy model load + store open happen once, not per request. Without
# these, every /ask reloaded the embedder (~1.2GB for Qwen3-0.6B) and re-opened ChromaDB, which dominated latency
# (the BM25 index and reranker are cached the same way in retrieve.py). Injected embedder/client args
# still bypass these: tests pass their own.
_EMBEDDER_CACHE = {}
_CLIENT_CACHE = {}
# Guards the *first* build of each cached singleton. FastAPI runs the sync /ask in a threadpool, so
# two concurrent cold-start requests could both miss the cache and each construct an Embedder (~1.2GB)
# / open a client before either wrote back, doubling peak memory. Double-checked locking below builds
# exactly once; the fast path (cache already populated) never takes the lock.
_EMBEDDER_LOCK = threading.Lock()
_CLIENT_LOCK = threading.Lock()


def _get_embedder(cfg: dict):
    """Build (and cache) the Embedder, keyed by model + device so a config change loads a fresh one."""
    key = (cfg.get("embed_model", DEFAULT_MODEL), cfg.get("embed_device", "auto"))
    if key not in _EMBEDDER_CACHE:
        with _EMBEDDER_LOCK:
            if key not in _EMBEDDER_CACHE:
                _EMBEDDER_CACHE[key] = Embedder(cfg)
    return _EMBEDDER_CACHE[key]


def _get_client(cfg: dict):
    """Open (and cache) the persistent Chroma client, keyed by vectorstore path."""
    path = cfg["paths"]["vectorstore"]
    if path not in _CLIENT_CACHE:
        with _CLIENT_LOCK:
            if path not in _CLIENT_CACHE:
                _CLIENT_CACHE[path] = chromadb.PersistentClient(path=path, settings=_CHROMA_SETTINGS)
    return _CLIENT_CACHE[path]


def _l2_normalize(embs):
    """L2-normalize rows for cosine, but safely: a degenerate chunk can yield a zero / non-finite
    raw vector, and dividing by its ~0 norm produces NaN, which ChromaDB rejects outright. Sanitize
    non-finite values to 0 and guard the zero-norm denominator so every row stays finite (a junk
    chunk just gets a harmless zero vector instead of crashing the whole index build)."""
    embs = np.nan_to_num(np.asarray(embs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embs / norms).tolist()


class Embedder:
    def __init__(self, cfg: dict):
        self.model_name = cfg.get("embed_model", DEFAULT_MODEL)
        # Prepended to QUERIES only (never stored docs). Qwen's format is
        # "Instruct: <task>\nQuery:" concatenated directly before the question text.
        self.query_instruction = cfg.get("query_instruction", "")
        self.model, self.backend = self._load(cfg)

    def _load(self, cfg: dict):
        from sentence_transformers import SentenceTransformer

        device = cfg.get("embed_device", "auto")
        if device in ("auto", "directml"):
            try:
                model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={"provider": "DmlExecutionProvider"},
                )
                log.info("Embedder using DirectML (ONNX) backend")
                return model, "directml"
            except Exception as exc:  # noqa: BLE001 - any failure -> CPU fallback
                log.warning("DirectML embedding unavailable (%s); falling back to CPU", exc)
        # CPU (torch) path. Optional per-model load kwargs let a decoder embedder load correctly:
        # Qwen3-Embedding uses last-token pooling, which needs left-padded tokenization (set via
        # embed_tokenizer_kwargs). An encoder (mean pooling) needs none, so its call is unchanged.
        st_kwargs = {}
        if cfg.get("embed_model_kwargs"):
            st_kwargs["model_kwargs"] = cfg["embed_model_kwargs"]
        if cfg.get("embed_tokenizer_kwargs"):
            st_kwargs["tokenizer_kwargs"] = cfg["embed_tokenizer_kwargs"]
        return SentenceTransformer(self.model_name, device="cpu", **st_kwargs), "cpu"

    def encode(self, texts):
        """Embed documents (no query instruction). Returns a list of float lists."""
        embs = self.model.encode(
            list(texts), normalize_embeddings=False, convert_to_numpy=True
        )
        return _l2_normalize(embs)  # safe L2 (won't NaN on a degenerate chunk)

    def embed_query(self, text: str):
        embs = self.model.encode(
            [self.query_instruction + text], normalize_embeddings=False, convert_to_numpy=True
        )
        return _l2_normalize(embs)[0]


def _collection(cfg: dict, client=None):
    if client is None:
        client = _get_client(cfg)
    return client.get_or_create_collection(
        name=cfg.get("collection_name", "xeno_wiki"),
        metadata={"hnsw:space": "cosine"},
    )


def drop_collection(cfg: dict, client=None) -> None:
    """Delete the collection so a rebuild re-embeds every chunk. Needed after a re-parse/re-chunk:
    build_index skips ids already present, so changed *text* under an existing id would otherwise
    keep its stale embedding."""
    if client is None:
        client = _get_client(cfg)
    try:
        client.delete_collection(cfg.get("collection_name", "xeno_wiki"))
    except Exception as exc:  # noqa: BLE001 - nothing to drop is fine
        log.info("drop_collection: %s", exc)


def _metadata(chunk: dict) -> dict:
    # Multi-tag membership: an explicit `games` set if the chunk carries one, else derived from the
    # single `game` label (back-compat for legacy chunks). Stored as per-game boolean flags so a
    # cross-appearance page (e.g. KOS-MOS) is filterable under each of its games.
    games = chunk.get("games")
    member = set(games) if games else membership_from_game(chunk.get("game"))
    meta = {
        "pageid": chunk["pageid"],
        "title": chunk["title"],
        "game": chunk["game"],
        "heading": chunk["heading"],
        "url": chunk["url"],
    }
    meta.update(membership_flags(member))
    return meta


def build_index(chunks, cfg: dict, embedder=None, client=None, batch_size: int = 256) -> int:
    """Embed chunks and add them to the persistent ChromaDB collection. Returns the collection count."""
    if embedder is None:
        embedder = Embedder(cfg)
    collection = _collection(cfg, client)

    batch = []

    def flush():
        if not batch:
            return
        ids = [c["chunk_id"] for c in batch]
        existing = set(collection.get(ids=ids)["ids"])
        new = [c for c in batch if c["chunk_id"] not in existing]
        if new:
            collection.add(
                ids=[c["chunk_id"] for c in new],
                embeddings=embedder.encode([c["text"] for c in new]),
                metadatas=[_metadata(c) for c in new],
                documents=[c["text"] for c in new],
            )
        batch.clear()

    for chunk in chunks:
        # ChromaDB rejects None metadata values, so a chunk with a missing pageid would abort the
        # whole build mid-batch. `action=parse` should always return a pageid, making this near
        # unreachable. Skip defensively with a loud log rather than crash a multi-hour embed run.
        if chunk.get("pageid") is None:
            log.warning("build_index: skipping chunk %r (title=%r): missing pageid",
                        chunk.get("chunk_id"), chunk.get("title"))
            continue
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush()
    flush()
    return collection.count()


def _iter_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(cfg: dict, embedder=None) -> int:
    """Embed all chunks from cfg['paths']['chunks'] into the persistent store. Returns the count."""
    return build_index(_iter_chunks(cfg["paths"]["chunks"]), cfg, embedder=embedder)


def _where(game_filter: str):
    """Membership game filter: a chosen game narrows to chunks whose ``g_<game>`` flag is set, i.e.
    pages that belong to that game (cross-appearance pages belong to several, ubiquitous/`series`
    pages belong to all). Falsy / non-base filter -> ``None`` (no restriction). Shares
    ``parse_wikitext.filter_membership`` with the BM25 filter."""
    game = filter_membership(game_filter)
    if game is None:
        return None
    return {f"g_{game}": True}


def dense_query(text: str, cfg: dict, n: int = None, game_filter: str = None, embedder=None,
                client=None, query_embedding=None):
    """Return up to ``n`` nearest chunks (cosine) as result dicts, **uncapped**: the raw dense
    candidate list for the hybrid retriever to fuse / rerank.

    ``query_embedding`` lets a caller supply an already-computed query vector so the same text isn't
    re-embedded across calls (the hybrid retriever runs a filtered *and* an unfiltered dense query for
    one question: same vector, different ``where``). When ``None`` the vector is embedded from ``text``
    as before, so every existing caller is unaffected."""
    if n is None:
        n = max(cfg.get("top_k", 8) * 5, 40)
    if query_embedding is None:
        if embedder is None:
            embedder = _get_embedder(cfg)
        query_embedding = embedder.embed_query(text)
    collection = _collection(cfg, client)
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=_where(game_filter),
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    # ChromaDB contracts these four arrays to be equal-length. Fuse them with a strict zip (the same
    # pattern fetch_chunks / fetch_page_chunks use) so a ragged payload (an API change or a corrupt
    # store) fails loudly with a ValueError here, rather than silently IndexError-ing on an unguarded
    # docs[i] / metas[i] or fabricating misaligned rows by index.
    out = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
        out.append({
            "chunk_id": cid,
            "text": doc,
            "distance": dist,
            **(meta or {}),
        })
    return out


def _is_infobox(it) -> bool:
    """A stat page's infobox chunk: its identity card carrying the Lua-decoded Location / Species /
    Level range. Marked by the ``infobox`` heading (breadcrumb ``> infobox:`` as a fallback)."""
    if (it.get("heading") or "").strip().lower() == "infobox":
        return True
    return "> infobox:" in (it.get("text") or "").lower()


def cap_per_page(items, k: int, per_page_cap: int):
    """Keep at most ``per_page_cap`` chunks per page, in rank order, until ``k`` results, so one long
    page can't monopolize the answer context. Preserves input order (the caller's ranking).

    Within a page's cap, its highest-ranked infobox chunk is guaranteed a slot whenever the page is
    cited at all: the infobox holds the page's identity facts (Location / Species / Level range), so
    boilerplate (Introduction) plus a generic stat chunk must not evict it: the bug where
    "Where is Territorial Rotbart?" lost the Bionis' Leg infobox to the per-page cap."""
    # Per page, choose which chunks are eligible (by list index, a stable unique identity): reserve
    # one slot for the top infobox chunk, then fill the rest with the highest-ranked remaining chunks.
    by_page = {}
    for idx, it in enumerate(items):
        by_page.setdefault(it.get("pageid"), []).append(idx)
    eligible = set()
    for idxs in by_page.values():
        infobox_idx = next((i for i in idxs if _is_infobox(items[i])), None)
        picked = [infobox_idx] if (infobox_idx is not None and per_page_cap > 0) else []
        for i in idxs:
            if len(picked) >= per_page_cap:
                break
            if i != infobox_idx:
                picked.append(i)
        eligible.update(picked)

    out = []
    for idx, it in enumerate(items):
        if idx not in eligible:
            continue
        out.append(it)
        if len(out) >= k:
            break
    return out


def fetch_chunks(ids, cfg: dict, client=None):
    """Materialize result dicts for the given chunk_ids from the collection (used for BM25-only hits
    not already in the dense list). Returns {chunk_id: dict}."""
    ids = [i for i in ids if i]
    if not ids:
        return {}
    collection = _collection(cfg, client)
    res = collection.get(ids=ids, include=["documents", "metadatas"])
    out = {}
    for cid, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
        m = meta or {}
        out[cid] = {"chunk_id": cid, "text": doc, "distance": None, **m}
    return out


def fetch_page_chunks(pageid, cfg: dict, client=None):
    """Return every chunk of one page (by ``pageid`` metadata), ordered by chunk_id, the page's
    siblings, used by the answer-time auto-merge to reassemble a fragmented stat page into a full
    profile. Returns ``[{chunk_id, text, ...meta}]`` (empty if the pageid is missing)."""
    return fetch_pages_chunks([pageid], cfg, client=client).get(pageid, [])


def fetch_pages_chunks(pageids, cfg: dict, client=None):
    """Batched sibling lookup: every chunk of every given page in ONE metadata-filtered
    ``collection.get`` (``$in``), grouped by pageid with each page's chunks ordered by chunk_id.
    The answer-time auto-merge inspects up to ``top_k`` distinct pages per question; issuing one
    ``get`` per page was an N+1 scan of a ~169k-row store on the hot path. Returns
    ``{pageid: [{chunk_id, text, ...meta}]}`` (pages with no chunks are simply absent)."""
    pids = [p for p in dict.fromkeys(pageids) if p is not None]
    if not pids:
        return {}
    collection = _collection(cfg, client)
    # Chroma rejects an empty ``$in`` list (guarded above); a single pid uses plain equality.
    where = {"pageid": pids[0]} if len(pids) == 1 else {"pageid": {"$in": pids}}
    res = collection.get(where=where, include=["documents", "metadatas"])
    out = {}
    for cid, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
        m = meta or {}
        out.setdefault(m.get("pageid"), []).append({"chunk_id": cid, "text": doc, **m})
    for chunks in out.values():
        chunks.sort(key=lambda c: c.get("chunk_id", ""))
    return out


def query(text: str, cfg: dict, k: int = None, game_filter: str = None, embedder=None, client=None):
    """Dense-only retrieval, diversified by page (over-fetch then per-page cap). Kept as the dense
    primitive; the hybrid path lives in ``retrieve.retrieve``."""
    if k is None:
        k = cfg.get("top_k", 8)
    per_page_cap = cfg.get("max_chunks_per_page", 2)
    dense = dense_query(text, cfg, n=max(k * 5, 40), game_filter=game_filter,
                        embedder=embedder, client=client)
    return cap_per_page(dense, k, per_page_cap)
