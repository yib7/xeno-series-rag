"""Tests for the resumable rendered-HTML fetcher (no network: a fake action=parse client)."""

import gzip
import json
import os
import re

import requests

from xeno_rag import fetch_html
from xeno_rag.fetch_html import (
    collect_timeout_titles, fetch_one, iter_html_records, retry_timeouts,
    RETRY_FILE_OFFSET,
)
from xeno_rag.fetch_content import save_checkpoint


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


def test_run_missing_stat_title_list_fails_with_actionable_message(tmp_path):
    """A configured-but-absent `paths.html_titles` must fail naming the file and the options (no
    step in the repo generates the stat-page list), not with a bare open() FileNotFoundError."""
    import pytest

    cfg = cfg_for(tmp_path)
    cfg["paths"]["html_titles"] = str(tmp_path / "titles_stats.jsonl")
    cfg["paths"]["titles"] = str(tmp_path / "titles.jsonl")
    with pytest.raises(FileNotFoundError) as exc:
        fetch_html.run(cfg, client=FakeClient())
    msg = str(exc.value)
    assert "titles_stats.jsonl" in msg
    assert "html_titles" in msg          # names the config knob to change
    assert "fall back" in msg            # explains the full-title-list alternative


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


def _write_raw_batch(html_dir, index, records):
    os.makedirs(html_dir, exist_ok=True)
    with gzip.open(os.path.join(html_dir, f"html_{index:05d}.jsonl.gz"),
                   "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_collect_timeout_titles_only_latest_timeout_records(tmp_path):
    """Only titles whose LATEST record is timeout-tagged are collected: permanent `request:` errors
    and pages already recovered by a later batch are excluded."""
    html_dir = str(tmp_path / "html")
    _write_raw_batch(html_dir, 0, [
        {"title": "Mythra", "error": "timeout:read timed out"},
        {"title": "Rex", "error": "request:boom"},                # permanent — never retried
        {"title": "Nia", "pageid": 3, "html": "<p>x</p>", "wikitext": "w"},
    ])
    _write_raw_batch(html_dir, 1, [
        {"title": "Pyra", "error": "timeout:read timed out"},
        # Mythra already recovered by this later batch — must not be retried again
        {"title": "Mythra", "pageid": 1, "html": "<p>ok</p>", "wikitext": "w"},
    ])
    assert collect_timeout_titles(html_dir) == ["Pyra"]


def test_retry_timeouts_refetches_into_new_offset_batch(tmp_path):
    cfg = cfg_for(tmp_path)
    html_dir = cfg["paths"]["html"]
    _write_raw_batch(html_dir, 0, [
        {"title": "Mythra", "error": "timeout:read timed out"},
        {"title": "Nia", "pageid": 3, "html": "<p>x</p>", "wikitext": "w"},
    ])
    client = FakeClient()

    n = retry_timeouts(cfg, client=client, log=None)

    assert n == 1
    assert client.calls == ["Mythra"]        # only the timeout title, not the healthy page
    # written into the reserved retry block, not right after the highest existing batch (that
    # index -- html_00001 -- belongs to a resumed main fetch; see RETRY_FILE_OFFSET / P1-2)
    assert os.path.isfile(os.path.join(html_dir, f"html_{RETRY_FILE_OFFSET:05d}.jsonl.gz"))
    assert not os.path.isfile(os.path.join(html_dir, "html_00001.jsonl.gz"))
    recs = list(iter_html_records(html_dir))
    latest = {r["title"]: r for r in recs}
    assert "error" not in latest["Mythra"]   # recovered record supersedes the failure
    # the main fetch checkpoint is untouched
    assert not os.path.isfile(cfg["paths"]["html_checkpoint"])


def test_retry_timeouts_is_idempotent_after_recovery(tmp_path):
    """A second pass after a successful recovery finds nothing to do (latest record wins)."""
    cfg = cfg_for(tmp_path)
    _write_raw_batch(cfg["paths"]["html"], 0,
                     [{"title": "Mythra", "error": "timeout:read timed out"}])
    assert retry_timeouts(cfg, client=FakeClient(), log=None) == 1
    second = FakeClient()
    assert retry_timeouts(cfg, client=second, log=None) == 0
    assert second.calls == []


def test_retry_timeouts_no_failures_is_a_noop(tmp_path):
    cfg = cfg_for(tmp_path)
    _write_raw_batch(cfg["paths"]["html"], 0,
                     [{"title": "Nia", "pageid": 3, "html": "<p>x</p>", "wikitext": "w"}])
    client = FakeClient()
    assert retry_timeouts(cfg, client=client, log=None) == 0
    assert client.calls == []


def test_retry_timeouts_still_failing_page_stays_tagged(tmp_path):
    """A retry that times out again writes a fresh timeout record: the page remains collectable by
    a future pass instead of silently disappearing."""
    cfg = cfg_for(tmp_path)
    html_dir = cfg["paths"]["html"]
    _write_raw_batch(html_dir, 0, [{"title": "Mythra", "error": "timeout:read timed out"}])

    assert retry_timeouts(cfg, client=RaisingClient(requests.Timeout("again")), log=None) == 1
    assert collect_timeout_titles(html_dir) == ["Mythra"]


def test_retry_timeouts_does_not_collide_with_resumed_main_fetch(tmp_path):
    """P1-2: retry_timeouts must not write into the index a resumed main fetch will reuse next.

    An INCOMPLETE main pull (checkpoint at batch 1 -- batch 2+ still to come) runs retry_timeouts to
    recover a timeout. The recovery batch must land at RETRY_FILE_OFFSET or higher, never at
    html_00002 -- the exact index a resumed main fetch writes next and would otherwise clobber."""
    cfg = cfg_for(tmp_path)
    html_dir = cfg["paths"]["html"]
    checkpoint = cfg["paths"]["html_checkpoint"]

    # Incomplete main pull: batches 0 and 1 written; checkpoint says batch 1 is the last completed
    # one, i.e. more batches are still to come (the pull is NOT finished).
    _write_raw_batch(html_dir, 0, [
        {"title": "Fast Page", "pageid": 1, "html": "<p>ok</p>", "wikitext": "ok"},
    ])
    _write_raw_batch(html_dir, 1, [
        {"title": "Slow Page", "error": "timeout:read"},
    ])
    save_checkpoint(1, checkpoint)

    n = retry_timeouts(cfg, client=FakeClient(), log=None)
    assert n == 1

    recovery_indices = []
    for name in os.listdir(html_dir):
        m = re.match(r"html_(\d+)\.jsonl\.gz$", name)
        if m and int(m.group(1)) not in (0, 1):
            recovery_indices.append(int(m.group(1)))
    assert recovery_indices, "no recovery batch was written"
    assert min(recovery_indices) >= RETRY_FILE_OFFSET, (
        f"recovery batch landed at {sorted(recovery_indices)}, inside the main-fetch index range"
    )
    assert not os.path.isfile(os.path.join(html_dir, "html_00002.jsonl.gz"))

    # Resumed main fetch continues from checkpoint + 1 = batch 2, writing html_00002.jsonl.gz --
    # exactly the index the old highest-overall-plus-one retry indexing would have used.
    _write_raw_batch(html_dir, 2, [
        {"title": "Next Page", "pageid": 2, "html": "<p>ok</p>", "wikitext": "ok"},
    ])

    assert collect_timeout_titles(html_dir) == []
    latest = {r["title"]: r for r in iter_html_records(html_dir)}
    assert "error" not in latest["Slow Page"]
