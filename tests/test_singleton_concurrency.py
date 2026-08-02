"""Concurrency tests for the lazy heavy-singleton caches.

FastAPI serves the sync ``/ask`` in a threadpool, so on a cold start two concurrent first requests
can both miss an unsynchronized cache and each construct a heavy singleton (Embedder ~1.2GB model,
Chroma client, reranker, BM25 index) before either writes back. The getters must guard the first
build with a lock (double-checked locking) so the constructor runs exactly once under a race.

These tests never load a real model/store: every underlying constructor is monkeypatched with a fake
that (a) counts constructions and (b) waits on a Barrier so both threads are provably inside the
critical-section window at the same time. Against unsynchronized getters the count is 2 (RED); with
the lock it must be 1 (GREEN).
"""

import threading

import pytest

from xeno_rag import embed_index, retrieve


def _race(getter, cfg, n_threads=2):
    """Run ``getter(cfg)`` from ``n_threads`` threads whose starts are aligned by a Barrier, so they
    all enter the (unlocked) getter within the same window. Returns the list of returned instances."""
    ready = threading.Barrier(n_threads)
    results = [None] * n_threads
    errors = []

    def worker(i):
        try:
            ready.wait()  # align all threads to hit the getter together
            results[i] = getter(cfg)
        except Exception as exc:  # noqa: BLE001 - surface thread errors to the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"worker threads raised: {errors}"
    return results


def _slow_ctor(counter, ctor_barrier):
    """A fake constructor factory: each instance increments ``counter`` and waits on ``ctor_barrier``
    so, if two threads both reach construction, they overlap inside it (forcing the race) rather than
    one finishing before the other starts. The Barrier's timeout lets a *correctly locked* getter,
    where only one thread ever constructs, proceed instead of deadlocking."""

    class _Fake:
        def __init__(self, *args, **kwargs):
            with counter["lock"]:
                counter["n"] += 1
            try:
                ctor_barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                pass  # only one constructor ran (locked getter) -> barrier times out, that's fine

    return _Fake


@pytest.fixture
def counter():
    return {"n": 0, "lock": threading.Lock()}


def test_get_embedder_constructs_once_under_race(monkeypatch, counter):
    embed_index._EMBEDDER_CACHE.clear()
    barrier = threading.Barrier(2)
    monkeypatch.setattr(embed_index, "Embedder", _slow_ctor(counter, barrier))

    cfg = {"embed_model": "m", "embed_device": "cpu"}
    results = _race(embed_index._get_embedder, cfg)

    assert counter["n"] == 1, "Embedder must be constructed exactly once under a concurrent first call"
    assert results[0] is results[1], "both racing threads must receive the same cached instance"


def test_get_client_constructs_once_under_race(monkeypatch, counter):
    embed_index._CLIENT_CACHE.clear()
    barrier = threading.Barrier(2)
    monkeypatch.setattr(embed_index.chromadb, "PersistentClient", _slow_ctor(counter, barrier))

    cfg = {"paths": {"vectorstore": "/tmp/vs_race"}}
    results = _race(embed_index._get_client, cfg)

    assert counter["n"] == 1, "Chroma client must be opened exactly once under a concurrent first call"
    assert results[0] is results[1]


def test_get_reranker_constructs_once_under_race(monkeypatch, counter):
    retrieve._RERANKER_CACHE.clear()
    barrier = threading.Barrier(2)
    monkeypatch.setattr(retrieve, "Reranker", _slow_ctor(counter, barrier))

    cfg = {"rerank_model": "r"}
    results = _race(retrieve._get_reranker, cfg)

    assert counter["n"] == 1, "Reranker must be constructed exactly once under a concurrent first call"
    assert results[0] is results[1]


def test_get_bm25_constructs_once_under_race(monkeypatch, tmp_path, counter):
    retrieve._BM25_CACHE.clear()
    # _get_bm25 returns None unless the index file exists; create it so the ctor path is exercised.
    bm25_path = tmp_path / "bm25.sqlite3"
    bm25_path.write_text("", encoding="utf-8")
    barrier = threading.Barrier(2)
    monkeypatch.setattr(retrieve, "Bm25Index", _slow_ctor(counter, barrier))

    cfg = {"paths": {"bm25": str(bm25_path)}}
    results = _race(retrieve._get_bm25, cfg)

    assert counter["n"] == 1, "Bm25Index must be constructed exactly once under a concurrent first call"
    assert results[0] is results[1]
