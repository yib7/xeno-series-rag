"""Tests for the cross-encoder reranker (model injected so no weights load in tests)."""

from xeno_rag.rerank import Reranker


class FakeCE:
    """Stand-in CrossEncoder: scores a (query, passage) pair by how many query words it contains."""

    def predict(self, pairs):
        out = []
        for q, p in pairs:
            qwords = set(q.lower().split())
            out.append(sum(1 for w in p.lower().split() if w in qwords))
        return out


def items(*texts):
    return [{"chunk_id": str(i), "text": t} for i, t in enumerate(texts)]


def test_rerank_orders_by_relevance_score():
    r = Reranker(cfg={}, model=FakeCE())
    res = r.rerank("monado shulk", items(
        "a page about gears and weather",      # 0 query words
        "shulk wields the monado in battle",   # 2 query words -> best
        "the monado is a sword",               # 1 query word
    ))
    assert [i["chunk_id"] for i in res] == ["1", "2", "0"]


def test_rerank_empty_is_empty():
    r = Reranker(cfg={}, model=FakeCE())
    assert r.rerank("anything", []) == []


def test_rerank_preserves_item_dicts():
    r = Reranker(cfg={}, model=FakeCE())
    res = r.rerank("monado", items("no match here", "the monado"))
    assert res[0]["text"] == "the monado"
    # original keys preserved; the cross-encoder score rides along for downstream relevance sizing
    assert {"chunk_id", "text"} <= set(res[0])


def test_rerank_attaches_score_in_order():
    r = Reranker(cfg={}, model=FakeCE())
    res = r.rerank("monado shulk", items(
        "a page about gears",        # 0 query words
        "shulk wields the monado",   # 2 -> best
        "the monado is a sword",     # 1
    ))
    assert all(isinstance(it["_score"], float) for it in res)
    scores = [it["_score"] for it in res]
    assert scores == sorted(scores, reverse=True)   # best-first: non-increasing
    assert res[0]["_score"] == 2.0
