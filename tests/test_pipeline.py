"""Tests for the corpus-build pipeline dispatch (no network, no model)."""

from xeno_rag import pipeline


def _stub_all(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "load_config", lambda: {})
    monkeypatch.setattr(pipeline.harvest_titles, "run", lambda cfg: calls.append("harvest"))
    monkeypatch.setattr(pipeline.fetch_html, "run", lambda cfg, **kw: calls.append("fetch"))
    monkeypatch.setattr(pipeline.parse_html, "run_hybrid", lambda cfg: calls.append("parse"))
    monkeypatch.setattr(pipeline.chunk, "run", lambda cfg: calls.append("chunk"))
    monkeypatch.setattr(pipeline.embed_index, "run", lambda cfg: calls.append("embed"))
    monkeypatch.setattr(pipeline.bm25_index, "run", lambda cfg: calls.append("bm25"))


def test_all_runs_every_step_in_order(monkeypatch):
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["all"])
    assert calls == ["harvest", "fetch", "parse", "chunk", "embed", "bm25"]


def test_single_step_runs_only_that_step(monkeypatch):
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["parse"])
    assert calls == ["parse"]


def test_dry_run_previews_plan_without_side_effects(monkeypatch, caplog):
    """`--dry-run` must expand and print the ordered plan but execute nothing — no config load, no
    step calls, no network/store touches. Guards against accidentally kicking off a ~19h live pull."""
    calls = []
    _stub_all(monkeypatch, calls)
    # If load_config or any step were invoked, calls would be non-empty; also fail loudly on load.
    monkeypatch.setattr(pipeline, "load_config",
                        lambda: (_ for _ in ()).throw(AssertionError("config loaded during dry-run")))
    import logging
    with caplog.at_level(logging.INFO):
        pipeline.main(["rebuild", "--dry-run"])
    assert calls == []                                  # nothing executed
    plan = " -> ".join(["fetch", "parse", "chunk", "embed_fresh", "bm25"])
    assert any(plan in rec.message for rec in caplog.records), "dry-run should print the expanded plan"
