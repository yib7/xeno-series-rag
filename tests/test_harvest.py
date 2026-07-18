"""Tests for title harvesting. Uses a fake client (no network)."""

import json

import pytest

from xeno_rag.harvest_titles import harvest_titles, write_titles, run


class FakeClient:
    """Replays scripted API responses and records the params of each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, params):
        self.calls.append(dict(params))
        return self._responses.pop(0)


PAGE1 = {
    "query": {"allpages": [
        {"title": "Noah", "pageid": 1},
        {"title": "Mio", "pageid": 2},
    ]},
    "continue": {"apcontinue": "Sena", "continue": "-||"},
}
PAGE2 = {
    "query": {"allpages": [
        {"title": "Sena", "pageid": 3},
    ]},
}


def test_harvest_paginates_until_no_continue():
    client = FakeClient([PAGE1, PAGE2])
    rows = list(harvest_titles(client, {}, nonredirects=True))

    assert rows == [
        {"title": "Noah", "pageid": 1},
        {"title": "Mio", "pageid": 2},
        {"title": "Sena", "pageid": 3},
    ]
    # second call carried apcontinue from the first response
    assert client.calls[1]["apcontinue"] == "Sena"
    # exactly two requests, then stopped (no continue on page 2)
    assert len(client.calls) == 2


def test_harvest_sets_namespace_and_filter():
    client = FakeClient([PAGE2])
    list(harvest_titles(client, {}, nonredirects=True))
    p = client.calls[0]
    assert p["list"] == "allpages"
    assert p["apnamespace"] == 0
    assert p["apfilterredir"] == "nonredirects"


def test_harvest_without_redirect_filter_omits_param():
    client = FakeClient([PAGE2])
    list(harvest_titles(client, {}, nonredirects=False))
    assert "apfilterredir" not in client.calls[0]


def test_write_titles_roundtrip(tmp_path):
    path = tmp_path / "titles.jsonl"
    write_titles(
        [{"title": "Noah", "pageid": 1}, {"title": "Mio", "pageid": 2}],
        str(path),
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"title": "Noah", "pageid": 1}


def test_write_titles_is_atomic_on_failure(tmp_path):
    """If the generator raises partway (mirrors the harvest's paginated fetch exhausting its
    retries), a pre-existing titles file must survive byte-for-byte -- not be left truncated by the
    old "open('w') and write as we go" approach, which would silently under-scope every downstream
    step. No leftover .tmp file should remain either."""
    path = tmp_path / "titles.jsonl"
    original = '{"title": "Noah", "pageid": 1}\n{"title": "Mio", "pageid": 2}\n'
    path.write_text(original, encoding="utf-8")

    def bad_records():
        yield {"title": "Sena", "pageid": 3}
        yield {"title": "Nia", "pageid": 4}
        raise RuntimeError("retries exhausted")

    with pytest.raises(RuntimeError, match="retries exhausted"):
        write_titles(bad_records(), str(path))

    assert path.read_text(encoding="utf-8") == original, "pre-existing file must be untouched"
    assert not (tmp_path / "titles.jsonl.tmp").exists(), "no leftover .tmp file"

    # Happy path: a normal (non-raising) write still produces the right file + count via the same
    # temp-then-replace path.
    good_path = tmp_path / "titles2.jsonl"
    count = write_titles(
        [{"title": "Sena", "pageid": 3}, {"title": "Nia", "pageid": 4}], str(good_path)
    )
    assert count == 2
    lines = good_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == {"title": "Sena", "pageid": 3}
    assert not (tmp_path / "titles2.jsonl.tmp").exists()


def test_run_writes_titles_file(tmp_path):
    path = tmp_path / "out" / "titles.jsonl"
    cfg = {"paths": {"titles": str(path)}}
    client = FakeClient([PAGE1, PAGE2])

    count = run(cfg, client=client)

    assert count == 3
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3
