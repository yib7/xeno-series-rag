"""Tests for the corpus-build pipeline dispatch (no network, no model)."""

from xeno_rag import pipeline


def _stub_all(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "load_config", dict)
    monkeypatch.setattr(pipeline.harvest_titles, "run", lambda cfg: calls.append("harvest"))
    monkeypatch.setattr(pipeline.fetch_content, "run", lambda cfg: calls.append("fetch_wikitext"))
    monkeypatch.setattr(pipeline.fetch_html, "run", lambda cfg, **kw: calls.append("fetch"))
    monkeypatch.setattr(pipeline.parse_html, "run_hybrid", lambda cfg: calls.append("parse"))
    monkeypatch.setattr(pipeline.chunk, "run", lambda cfg: calls.append("chunk"))
    monkeypatch.setattr(pipeline.embed_index, "run", lambda cfg: calls.append("embed"))
    monkeypatch.setattr(pipeline.bm25_index, "run", lambda cfg: calls.append("bm25"))


def test_all_runs_every_step_in_order(monkeypatch):
    """`all` must include the wikitext content pull: without it, `parse` finds an empty
    pages dir and silently emits an HTML-only corpus missing the ~28k prose pages."""
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["all"])
    assert calls == ["harvest", "fetch_wikitext", "fetch", "parse", "chunk", "embed", "bm25"]


def test_single_step_runs_only_that_step(monkeypatch):
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["parse"])
    assert calls == ["parse"]


def test_retry_timeouts_step_dispatches_and_stays_out_of_meta_steps(monkeypatch):
    """`retry_timeouts` is an explicit human-run recovery step (hits the live API): runnable on its
    own, but never folded into `all` or `rebuild`."""
    calls = []
    _stub_all(monkeypatch, calls)
    monkeypatch.setattr(pipeline.fetch_html, "retry_timeouts",
                        lambda cfg, **kw: calls.append("retry_timeouts") or 0)
    pipeline.main(["retry_timeouts"])
    assert calls == ["retry_timeouts"]
    assert "retry_timeouts" not in pipeline.META["all"]
    assert "retry_timeouts" not in pipeline.META["rebuild"]


def test_fetch_wikitext_step_reaches_fetch_content_run(monkeypatch):
    """`fetch_content.run` must be reachable from the CLI as its own step, not only
    through `all`."""
    calls = []
    _stub_all(monkeypatch, calls)
    pipeline.main(["fetch_wikitext"])
    assert calls == ["fetch_wikitext"]


def test_dry_run_previews_plan_without_side_effects(monkeypatch, caplog):
    """`--dry-run` must expand and print the ordered plan but execute nothing: no config load, no
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
    plan = "fetch -> parse -> chunk -> embed_fresh -> bm25"
    assert any(plan in rec.message for rec in caplog.records), "dry-run should print the expanded plan"


def test_dry_run_all_plan_includes_fetch_wikitext(monkeypatch, caplog):
    calls = []
    _stub_all(monkeypatch, calls)
    import logging
    with caplog.at_level(logging.INFO):
        pipeline.main(["all", "--dry-run"])
    assert calls == []
    plan = "harvest -> fetch_wikitext -> fetch -> parse -> chunk -> embed -> bm25"
    assert any(plan in rec.message for rec in caplog.records)


def test_embed_fresh_with_missing_chunks_drops_nothing_and_exits_cleanly(monkeypatch, tmp_path):
    """The destructive step must check its input BEFORE dropping the collection: dropping first and
    then failing on a missing chunks file would leave the app with no store at all. The user gets the message, not a traceback."""
    calls = []
    _stub_all(monkeypatch, calls)
    cfg = {"paths": {"chunks": str(tmp_path / "chunks.jsonl")}}
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline.embed_index, "drop_collection", lambda c, client=None: calls.append("drop"))
    try:
        pipeline.main(["embed_fresh"])
    except SystemExit as exc:
        assert "chunks" in str(exc) and "Nothing was changed" in str(exc)
    else:
        raise AssertionError("expected a clean SystemExit")
    assert calls == []


def test_embed_fresh_drops_then_embeds_when_chunks_exist(monkeypatch, tmp_path):
    calls = []
    _stub_all(monkeypatch, calls)
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "1-0"}\n', encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_config", lambda: {"paths": {"chunks": str(chunks)}})
    monkeypatch.setattr(pipeline.embed_index, "drop_collection", lambda c, client=None: calls.append("drop"))
    pipeline.main(["embed_fresh"])
    assert calls == ["drop", "embed"]
