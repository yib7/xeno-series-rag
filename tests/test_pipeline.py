"""Tests for the corpus-build pipeline dispatch (no network, no model)."""

from xeno_rag import pipeline


def _stub_all(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "load_config", lambda: {})
    monkeypatch.setattr(pipeline.harvest_titles, "run", lambda cfg: calls.append("harvest"))
    monkeypatch.setattr(pipeline.fetch_content, "run", lambda cfg: calls.append("fetch"))
    monkeypatch.setattr(pipeline.parse_wikitext, "run", lambda cfg: calls.append("parse"))
    monkeypatch.setattr(pipeline.chunk, "run", lambda cfg: calls.append("chunk"))
    monkeypatch.setattr(pipeline.embed_index, "run", lambda cfg: calls.append("embed"))


def test_all_runs_every_step_in_order(monkeypatch):
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["all"])
    assert calls == ["harvest", "fetch", "parse", "chunk", "embed"]


def test_single_step_runs_only_that_step(monkeypatch):
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["parse"])
    assert calls == ["parse"]
