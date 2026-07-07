"""Tests for the resumable rendered-HTML fetcher (no network: a fake action=parse client)."""

import os

import requests

from xeno_rag import fetch_html
from xeno_rag.fetch_html import fetch_one, iter_html_records


class FakeClient:
    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    def get(self, params):
        title = params["page"]
        self.calls.append(title)
        if title in self.missing:
            return {"error": {"code": "missingtitle"}}
        return {"parse": {"pageid": len(title), "title": title,
                          "text": f"<div class='mw-parser-output'><p>{title} prose</p></div>",
                          "wikitext": f"{title} wikitext"}}


def cfg_for(tmp_path):
    return {"html_batch_size": 2,
            "paths": {"html": str(tmp_path / "html"),
                      "html_checkpoint": str(tmp_path / "html_ckpt.json")}}


def titles(*names):
    return [{"title": n} for n in names]


def test_fetch_one_returns_html_and_wikitext():
    rec = fetch_one(FakeClient(), "Mythra")
    assert rec["title"] == "Mythra"
    assert "prose" in rec["html"] and rec["wikitext"] == "Mythra wikitext"
    assert "error" not in rec


def test_fetch_one_records_error_without_raising():
    rec = fetch_one(FakeClient(missing={"Ghost"}), "Ghost")
    assert rec["error"] == "missingtitle"


class RaisingClient:
    """A client whose ``get`` always raises the configured exception — to check how fetch_one
    categorizes the failure (retryable timeout vs. permanent)."""
    def __init__(self, exc):
        self.exc = exc

    def get(self, params):
        raise self.exc


def test_fetch_one_marks_timeout_as_retryable():
    """A ``requests.Timeout`` is transient (server slow / network blip), so it must be recorded
    distinctly as ``timeout:...`` — the resume logic can re-attempt these rather than treating them
    like a permanent 'missing page'. Otherwise a flaky window silently drops real pages."""
    rec = fetch_one(RaisingClient(requests.Timeout("read timed out")), "Mythra")
    assert rec["title"] == "Mythra"
    assert rec["error"].startswith("timeout:"), rec["error"]


def test_fetch_one_marks_generic_exception_as_permanent_request_error():
    """A non-timeout exception keeps the existing ``request:...`` tag (permanent-until-fixed), so it
    stays distinguishable from a retryable timeout."""
    rec = fetch_one(RaisingClient(ValueError("boom")), "Rex")
    assert rec["error"].startswith("request:"), rec["error"]
    assert "timeout:" not in rec["error"]


def test_fetch_all_writes_batches_and_checkpoint(tmp_path):
    cfg = cfg_for(tmp_path)
    client = FakeClient()
    fetch_html.fetch_all(client, titles("A", "B", "C"), cfg, log=None)
    recs = list(iter_html_records(cfg["paths"]["html"]))
    assert [r["title"] for r in recs] == ["A", "B", "C"]
    assert os.path.isfile(cfg["paths"]["html_checkpoint"])


def test_run_resumes_from_checkpoint(tmp_path):
    cfg = cfg_for(tmp_path)
    ts = titles("A", "B", "C", "D")          # batch_size 2 -> batches [A,B],[C,D]
    first = FakeClient()
    fetch_html.run(cfg, client=first, titles=ts)
    assert first.calls == ["A", "B", "C", "D"]
    # a second run starts past the saved checkpoint: nothing re-fetched
    second = FakeClient()
    fetch_html.run(cfg, client=second, titles=ts)
    assert second.calls == []
