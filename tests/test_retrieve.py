"""Tests for hybrid retrieval: RRF fusion of dense + BM25, optional rerank, page-diversified."""

import pytest

from xeno_rag import retrieve as retrieve_mod
from xeno_rag.bm25_index import Bm25Index
from xeno_rag.embed_index import Embedder, build_index
from xeno_rag.retrieve import rrf_fuse, retrieve, merge_fragmented_pages

CHUNKS = [
    {"chunk_id": "1-0", "pageid": 1, "title": "Mimeosome", "game": "XCX", "heading": "Introduction",
     "url": "https://w/Mimeosome", "text": "A mimeosome is an artificial body used by humanity on Mira."},
    {"chunk_id": "2-0", "pageid": 2, "title": "Claymore", "game": "XCX", "heading": "infobox",
     "url": "https://w/Claymore", "text": "Claymore is a Skell weapon with high physical attack."},
    {"chunk_id": "3-0", "pageid": 3, "title": "Elma", "game": "XCX", "heading": "Introduction",
     "url": "https://w/Elma", "text": "Elma is a BLADE colonel stationed in New Los Angeles."},
]


# --- RRF fusion (pure) ---

def test_rrf_rewards_agreement_across_lists():
    # 'b' is in both lists; it should beat items appearing in only one.
    order = rrf_fuse([["a", "b", "c"], ["b", "d"]])
    assert order[0] == "b"
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_surfaces_item_present_in_only_one_list():
    # An item only BM25 found (not in the dense list) still makes it into the fused ranking.
    order = rrf_fuse([["a", "b"], ["z"]])
    assert "z" in order


# --- hybrid retrieve (real embedder + tmp index; BM25/reranker injected) ---

@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("vs_retr")
    return {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
        "collection_name": "retr_test", "top_k": 3, "max_chunks_per_page": 2,
        "hybrid_candidates": 10, "rrf_k": 60,
        "use_bm25": False, "use_reranker": False,
        "paths": {"vectorstore": str(d)},
    }


@pytest.fixture(scope="module")
def embedder(cfg):
    return Embedder(cfg)


@pytest.fixture(scope="module")
def indexed(cfg, embedder):
    build_index(CHUNKS, cfg, embedder=embedder)
    return True


class FakeBm25:
    def __init__(self, ids):
        self.ids = ids
        self.seen = None

    def search(self, query, n=60, game_filter=None):
        self.seen = (query, game_filter)
        return self.ids


class FakeCE:
    def predict(self, pairs):
        # rank by query-word overlap so tests are deterministic
        out = []
        for q, p in pairs:
            qs = set(q.lower().split())
            out.append(sum(1 for w in p.lower().split() if w in qs))
        return out


def test_retrieve_dense_only_returns_capped_results(cfg, embedder, indexed):
    res = retrieve("artificial body mimeosome", cfg, embedder=embedder)
    assert res and res[0]["title"] == "Mimeosome"
    assert len(res) <= cfg["top_k"]


def test_retrieve_fuses_bm25_only_hit(cfg, embedder, indexed):
    # BM25 surfaces Elma (3-0) for a query the dense side ranks elsewhere; fusion must include it.
    bm25 = FakeBm25(["3-0"])
    cfg2 = {**cfg, "use_bm25": True}
    res = retrieve("weapon", cfg2, embedder=embedder, bm25=bm25)
    assert "3-0" in [r["chunk_id"] for r in res]
    assert bm25.seen[0] == "weapon"          # query forwarded to BM25


def test_retrieve_passes_game_filter_to_bm25(cfg, embedder, indexed):
    bm25 = FakeBm25([])
    retrieve("anything", {**cfg, "use_bm25": True}, embedder=embedder, bm25=bm25, game_filter="XCX")
    assert bm25.seen[1] == "XCX"


def test_retrieve_applies_reranker(cfg, embedder, indexed):
    from xeno_rag.rerank import Reranker
    res = retrieve("Skell weapon attack", {**cfg, "use_reranker": True}, embedder=embedder,
                   reranker=Reranker(model=FakeCE()))
    assert res[0]["title"] == "Claymore"     # the passage richest in query words ranked first


# --- BM25 cache revalidation: a rebuilt index file must be reopened, not served stale ---

def _bm25_chunk(cid, pid, title, text):
    return {"chunk_id": cid, "pageid": pid, "title": title, "game": "XC1",
            "heading": "Introduction", "url": "u", "text": text}


def test_get_bm25_reopens_after_index_file_is_replaced(tmp_path):
    """`pipeline bm25` swaps a new file into place via os.replace; the serving cache must notice
    (mtime/size stat) and reopen instead of serving the old connection's stale index forever."""
    import os
    path = str(tmp_path / "bm25.sqlite3")
    Bm25Index.build([_bm25_chunk("1-0", 1, "Shulk", "Shulk wields the Monado.")], path=path).close()
    cfg = {"paths": {"bm25": path}}
    retrieve_mod._BM25_CACHE.clear()

    idx1 = retrieve_mod._get_bm25(cfg)
    assert idx1.search("Monado", n=3) == ["1-0"]
    assert retrieve_mod._get_bm25(cfg) is idx1     # unchanged file -> cached instance reused

    # Simulate the rebuild swap. The cached connection is closed first because on Windows an open
    # handle blocks the replace (the production path is "stop the server, rebuild, restart"; the
    # revalidation exists for the POSIX deployment and for any same-process rebuild).
    idx1.close()
    new = str(tmp_path / "new.sqlite3")
    Bm25Index.build([_bm25_chunk("2-0", 2, "Rex", "Rex is a salvager with Pyra.")], path=new).close()
    os.replace(new, path)

    idx2 = retrieve_mod._get_bm25(cfg)
    assert idx2 is not idx1, "a replaced index file must be reopened"
    assert idx2.search("salvager Pyra", n=3) == ["2-0"]    # new content served
    assert idx2.search("Monado", n=3) == []                # old content gone
    idx2.close()
    retrieve_mod._BM25_CACHE.clear()


def test_get_bm25_missing_file_returns_none(tmp_path):
    retrieve_mod._BM25_CACHE.clear()
    assert retrieve_mod._get_bm25({"paths": {"bm25": str(tmp_path / "absent.sqlite3")}}) is None


# --- auto-merging: consolidate a stat page's fragmented factblocks into one rich block ---

def _c(cid, pid, heading, text, title="Mon", game="XC1", url="https://w/Mon"):
    return {"chunk_id": cid, "pageid": pid, "heading": heading, "text": text,
            "title": title, "game": game, "url": url}


def test_merge_consolidates_fragmented_stat_page():
    """A retrieved stat-page chunk is replaced by ONE block carrying the page's full profile —
    including high-value sibling factblocks (resistances, drops) that retrieval ranked too low to
    surface. This is the Rotbart fix: 30-token scraps -> a coherent enemy profile, no re-embed."""
    siblings = [
        _c("9-0000", 9, "infobox", "[XC1] Mon > infobox: Species: Gogol. Location: Bionis' Leg. Level 81."),
        _c("9-0001", 9, "Enemy", "[XC1] Mon > Enemy: HP: 416200. STR: 1721."),
        _c("9-0002", 9, "Enemy", "[XC1] Mon > Enemy: Break: 100% Res. Topple: Vul."),  # not retrieved
        _c("9-0003", 9, "Drops", "[XC1] Mon > Drops: Gogol Horn 83%."),               # not retrieved
    ]
    retrieved = [siblings[0]]  # only the infobox was actually retrieved
    out = merge_fragmented_pages(retrieved, {"merge_min_small": 3}, fetch_fn=lambda pid: siblings)
    assert len(out) == 1
    txt = out[0]["text"]
    assert "Location: Bionis' Leg" in txt   # the retrieved infobox content kept
    assert "Break: 100% Res" in txt          # sibling resistances pulled in
    assert "Gogol Horn" in txt               # sibling drops pulled in
    assert out[0]["title"] == "Mon" and out[0]["url"] == "https://w/Mon"  # source metadata preserved


def test_merge_leaves_prose_page_unchanged():
    """A page with only a couple of large prose chunks is not fragmented -> passed through as-is."""
    big = "word " * 200
    sibs = [_c("2-0000", 2, "Introduction", f"[XC1] Lore > Introduction: {big}"),
            _c("2-0001", 2, "Story", f"[XC1] Lore > Story: {big}")]
    out = merge_fragmented_pages([sibs[0]], {}, fetch_fn=lambda pid: sibs)
    assert out == [sibs[0]]


def test_merge_respects_char_budget():
    sibs = [_c(f"9-{i:04d}", 9, "Enemy", "[XC1] Mon > Enemy: " + "a" * 100) for i in range(20)]
    out = merge_fragmented_pages([sibs[0]], {"merge_min_small": 3, "merge_max_chars": 500},
                                 fetch_fn=lambda pid: sibs)
    assert len(out) == 1
    assert len(out[0]["text"]) <= 500 + 120   # body capped near budget (+ short breadcrumb header)


def test_merge_disabled_passthrough():
    sibs = [_c(f"9-{i:04d}", 9, "Enemy", "x") for i in range(5)]
    out = merge_fragmented_pages([sibs[0]], {"merge_stat_pages": False}, fetch_fn=lambda pid: sibs)
    assert out == [sibs[0]]


class SpyCollection:
    """Fake Chroma collection over in-memory chunk dicts; counts ``get`` calls and understands the
    two metadata filters the sibling fetch uses ({"pageid": X} and {"pageid": {"$in": [...]}})."""

    def __init__(self, rows):
        self.rows = rows
        self.get_calls = 0

    def get(self, ids=None, where=None, include=None):
        self.get_calls += 1
        pid = (where or {}).get("pageid")
        if isinstance(pid, dict):
            wanted = pid["$in"]
            assert wanted, "Chroma rejects an empty $in list"
        else:
            wanted = [pid]
        hit = [r for r in self.rows if r["pageid"] in wanted]
        return {"ids": [r["chunk_id"] for r in hit],
                "documents": [r["text"] for r in hit],
                "metadatas": [{k: v for k, v in r.items() if k not in ("chunk_id", "text")}
                              for r in hit]}


class SpyClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


def test_merge_default_path_batches_sibling_lookup_into_one_call():
    """N retrieved pages -> exactly ONE collection.get (the batched $in query), and the merged
    output is identical to the injected per-page fetch_fn path (the pre-batching behavior)."""
    stat_sibs = [_c(f"9-{i:04d}", 9, "Enemy", f"[XC1] Mon > Enemy: stat {i}") for i in range(5)]
    prose = _c("3-0", 3, "Introduction", "[XC1] Other > Introduction: " + "y " * 300,
               title="Other", url="https://w/Other")
    retrieved = [stat_sibs[0], prose, stat_sibs[2]]
    col = SpyCollection(stat_sibs + [prose])
    cfg = {"merge_min_small": 3, "collection_name": "xeno_wiki"}

    out = merge_fragmented_pages(retrieved, cfg, client=SpyClient(col))

    assert col.get_calls == 1, "sibling lookup must be a single batched query, not one per page"
    expected = merge_fragmented_pages(retrieved, cfg,
                                      fetch_fn=lambda pid: stat_sibs if pid == 9 else [prose])
    assert out == expected
    pids = [c["pageid"] for c in out]
    assert pids.count(9) == 1 and 3 in pids


def test_merge_default_path_survives_collection_failure():
    """A batched-fetch failure must degrade to passthrough (no merge), never break answering."""
    class BrokenCollection:
        def get(self, **kwargs):
            raise RuntimeError("store offline")

    retrieved = [_c("9-0000", 9, "Enemy", "x")]
    out = merge_fragmented_pages(retrieved, {}, client=SpyClient(BrokenCollection()))
    assert out == retrieved


def test_merge_collapses_multiple_hits_from_one_page():
    sibs = [_c(f"9-{i:04d}", 9, "Enemy", f"[XC1] Mon > Enemy: stat {i}") for i in range(5)]
    other = _c("3-0", 3, "Introduction", "[XC1] Other > Introduction: " + "y " * 300)
    retrieved = [sibs[0], sibs[2], other]
    out = merge_fragmented_pages(retrieved, {"merge_min_small": 3},
                                 fetch_fn=lambda pid: sibs if pid == 9 else [other])
    pids = [c["pageid"] for c in out]
    assert pids.count(9) == 1   # the stat page's two hits collapse into one block
    assert 3 in pids            # the prose page is retained separately
