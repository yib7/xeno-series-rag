"""Tests for the resumable rendered-HTML fetcher (no network: a fake action=parse client)."""

import os

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
