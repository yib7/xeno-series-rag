"""Tests for the manual reindex CLI (`scripts/reindex.py`).

`scripts/` is not an importable package, so the module is loaded by file path via
``importlib.util.spec_from_file_location``. Two things are exercised:

1. Importing the module must have no side effects (no config read, no embedder build, no store
   touch) -- everything must live inside ``main()``.
2. ``main()`` must delegate the index build to ``embed_index.build_index``, which inherits its
   ``pageid is None`` skip-with-log guard (audit findings P2-1 / P2-9) instead of re-implementing
   its own flush loop that lacks the guard.
"""

import importlib.util
import json
import os

REINDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "reindex.py")


def _load_reindex_module():
    spec = importlib.util.spec_from_file_location("reindex", REINDEX_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reindex_module_import_has_no_side_effects(tmp_path, monkeypatch):
    """Loading the module object must not read config.yaml, build an embedder, or touch a store.

    Proof: chdir to a tmp dir with NO config.yaml present and import the module. A module-level
    `open("config.yaml")` would raise FileNotFoundError at import time; everything lives inside
    main(), so the import just defines it.
    """
    monkeypatch.chdir(tmp_path)
    module = _load_reindex_module()  # must not raise FileNotFoundError
    assert hasattr(module, "main"), "module must expose an importable main() entry point"


def test_reindex_skips_none_pageid_chunk(tmp_path, caplog):
    """A chunk with pageid=None must be skipped with a logged warning (via build_index's guard),
    not abort the whole manual reindex -- reproduces audit finding P2-1 against the CLI path.

    Reuses the stub shapes from tests/test_embed.py::test_build_index_skips_chunk_with_none_pageid
    (a _RecordingCollection/_StubClient/_StubEmbedder matching ChromaDB's get/add/count contract).
    """
    import logging

    class _RecordingCollection:
        def __init__(self):
            self.added = []

        def get(self, ids=None, **kw):
            return {"ids": []}

        def add(self, ids, embeddings, metadatas, documents):
            self.added.extend(ids)

        def count(self):
            return len(self.added)

    class _StubClient:
        def __init__(self, col):
            self._col = col

        def get_or_create_collection(self, name, metadata=None):
            return self._col

    class _StubEmbedder:
        def encode(self, texts):
            return [[0.0, 1.0] for _ in texts]

    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        {"chunk_id": "ok-0", "pageid": 1, "title": "Fine", "game": "XC1",
         "heading": "Introduction", "url": "u", "text": "fine"},
        {"chunk_id": "None-0", "pageid": None, "title": "Broken", "game": "XC1",
         "heading": "Introduction", "url": "u", "text": "broken"},
    ]
    chunks_path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")

    cfg = {"collection_name": "s1", "paths": {"chunks": str(chunks_path)}}
    col = _RecordingCollection()
    module = _load_reindex_module()

    with caplog.at_level(logging.WARNING, logger="xeno_rag.embed_index"):
        total = module.main(cfg=cfg, client=_StubClient(col), embedder=_StubEmbedder(), argv=[])

    assert total == 1 and col.added == ["ok-0"]          # good chunk indexed, bad one skipped
    assert any("missing pageid" in r.message for r in caplog.records)


def test_reindex_fresh_with_missing_chunks_drops_nothing(tmp_path):
    """--fresh must check its input before dropping the collection."""
    import pytest

    from xeno_rag.errors import SetupError

    dropped = []

    class _Client:
        def delete_collection(self, name):
            dropped.append(name)

    module = _load_reindex_module()
    cfg = {"paths": {"chunks": str(tmp_path / "missing.jsonl"), "vectorstore": str(tmp_path / "vs")},
           "collection_name": "c"}
    with pytest.raises(SetupError, match="Nothing was changed"):
        module.main(cfg=cfg, client=_Client(), embedder=object(), argv=["--fresh"])
    assert dropped == []


def test_reindex_missing_chunks_fails_before_the_model_or_store_is_touched(tmp_path, monkeypatch):
    """The input check comes first: no embedding model load, no Chroma client, even without --fresh."""
    import pytest

    from xeno_rag.errors import SetupError

    module = _load_reindex_module()
    touched = []
    monkeypatch.setattr(module, "Embedder", lambda cfg: touched.append("embedder"))
    monkeypatch.setattr(module.chromadb, "PersistentClient", lambda **kw: touched.append("client"))
    cfg = {"paths": {"chunks": str(tmp_path / "missing.jsonl"), "vectorstore": str(tmp_path / "vs")},
           "collection_name": "c"}
    with pytest.raises(SetupError, match="pipeline chunk"):
        module.main(cfg=cfg, argv=[])
    assert touched == []
    assert not (tmp_path / "vs").exists()
