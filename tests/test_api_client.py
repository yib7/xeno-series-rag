"""Tests for the etiquette-baked API client. Uses an injected fake session (no network)."""

import pytest

from xeno_rag.api_client import WikiClient

CFG = {
    "base_url": "https://example.org/w/api.php",
    "user_agent": "XenoRAG/test",
    "request_delay_seconds": 2,
    "maxlag": 5,
}


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {"ok": True}
        self.headers = headers or {}

    def json(self):
        return self._json


class FakeSession:
    """Records calls and replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Capture sleeps instead of actually sleeping."""
    slept = []
    monkeypatch.setattr("xeno_rag.api_client.time.sleep", lambda s: slept.append(s))
    return slept


def test_get_injects_required_params_and_sleeps(no_sleep):
    session = FakeSession([FakeResponse(json_data={"hello": "world"})])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"hello": "world"}
    sent = session.calls[0]["params"]
    assert sent["format"] == "json"
    assert sent["formatversion"] == 2
    assert sent["maxlag"] == 5
    assert sent["action"] == "query"
    assert 2 in no_sleep  # slept the configured request delay after success


def test_user_agent_header_is_set():
    session = FakeSession([FakeResponse()])
    WikiClient(CFG, session=session)
    assert session.headers["User-Agent"] == "XenoRAG/test"


def test_retries_on_429_and_honors_retry_after(no_sleep):
    session = FakeSession([
        FakeResponse(status_code=429, headers={"Retry-After": "7"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert 7 in no_sleep  # honored Retry-After
    assert len(session.calls) == 2


def test_retries_on_maxlag_error(no_sleep):
    session = FakeSession([
        FakeResponse(json_data={"error": {"code": "maxlag", "info": "waiting"}}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert len(session.calls) == 2


def test_raises_when_retries_exhausted(no_sleep):
    session = FakeSession([FakeResponse(status_code=429) for _ in range(6)])
    client = WikiClient(CFG, session=session)

    with pytest.raises(RuntimeError):
        client.get({"action": "query"}, max_retries=6)


@pytest.mark.live
def test_live_siteinfo_returns_statistics():
    """Real throttled call against the wiki (run with `-m live`)."""
    from xeno_rag.config import load_config

    client = WikiClient(load_config("config.yaml"))
    data = client.get({"action": "query", "meta": "siteinfo", "siprop": "statistics"})
    assert data["query"]["statistics"]["articles"] > 30000
