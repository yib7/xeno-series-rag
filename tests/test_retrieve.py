"""Tests for hybrid retrieval: RRF fusion of dense + BM25, optional rerank, page-diversified."""

import pytest

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
        "embed_model": "BAAI/bge-base-en-v1.5", "embed_device": "cpu",
        "bge_query_instruction": "Represent this sentence for searching relevant passages: ",
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
    from xeno_rag.rerank import Reranker
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


def test_merge_collapses_multiple_hits_from_one_page():
    sibs = [_c(f"9-{i:04d}", 9, "Enemy", f"[XC1] Mon > Enemy: stat {i}") for i in range(5)]
    other = _c("3-0", 3, "Introduction", "[XC1] Other > Introduction: " + "y " * 300)
    retrieved = [sibs[0], sibs[2], other]
    out = merge_fragmented_pages(retrieved, {"merge_min_small": 3},
                                 fetch_fn=lambda pid: sibs if pid == 9 else [other])
    pids = [c["pageid"] for c in out]
    assert pids.count(9) == 1   # the stat page's two hits collapse into one block
    assert 3 in pids            # the prose page is retained separately
