"""Tests for batched, resumable content fetching. Fake client, no network."""

import json

from xeno_rag.fetch_content import (
    batched,
    fetch_all,
    load_checkpoint,
    run,
    save_checkpoint,
)


class FakeClient:
    def __init__(self, n_responses):
        # each response returns one fake page echoing the requested titles
        self._n = n_responses
        self.calls = []

    def get(self, params):
        self.calls.append(dict(params))
        titles = params["titles"].split("|")
        return {"query": {"pages": [{"title": t, "pageid": i, "revisions": [{"slots": {"main": {"content": f"wikitext for {t}"}}}]} for i, t in enumerate(titles)]}}


def _titles(n):
    return [{"title": f"Page{i}", "pageid": i} for i in range(n)]


def _cfg(tmp_path, batch_size=2):
    return {
        "batch_size": batch_size,
        "paths": {
            "pages": str(tmp_path / "pages"),
            "checkpoint": str(tmp_path / "checkpoint.json"),
            "titles": str(tmp_path / "titles.jsonl"),
        },
    }


def test_batched_chunks_with_remainder():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "checkpoint.json")
    assert load_checkpoint(path) == -1  # absent → -1
    save_checkpoint(3, path)
    assert load_checkpoint(path) == 3


def test_fetch_all_writes_one_file_per_batch_and_advances_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, batch_size=2)
    client = FakeClient(3)

    fetch_all(client, _titles(5), cfg, start_batch=0)

    pages_dir = tmp_path / "pages"
    assert (pages_dir / "pages_00000.jsonl").is_file()
    assert (pages_dir / "pages_00001.jsonl").is_file()
    assert (pages_dir / "pages_00002.jsonl").is_file()
    assert load_checkpoint(cfg["paths"]["checkpoint"]) == 2
    # batch 0 had 2 titles
    first = (pages_dir / "pages_00000.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(first) == 2
    assert json.loads(first[0])["title"] == "Page0"


def test_resume_skips_completed_batches(tmp_path):
    cfg = _cfg(tmp_path, batch_size=2)
    # pretend batches 0 and 1 already done
    save_checkpoint(1, cfg["paths"]["checkpoint"])
    client = FakeClient(1)

    run(cfg, client=client, titles=_titles(5))

    # only batch index 2 should have been fetched
    assert len(client.calls) == 1
    assert client.calls[0]["titles"] == "Page4"
    assert (tmp_path / "pages" / "pages_00002.jsonl").is_file()
    assert not (tmp_path / "pages" / "pages_00000.jsonl").is_file()


def test_run_without_a_titles_file_is_a_setup_error_naming_the_harvest_step(tmp_path):
    import pytest

    from xeno_rag.errors import SetupError

    cfg = {"paths": {"titles": str(tmp_path / "titles.jsonl"),
                     "checkpoint": str(tmp_path / "cp.json"), "pages": str(tmp_path / "pages")}}
    with pytest.raises(SetupError, match="pipeline harvest"):
        run(cfg, client=FakeClient(1))
    assert not (tmp_path / "pages").exists()
