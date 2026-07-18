"""Tests for the etiquette-baked API client. Uses an injected fake session (no network)."""

import pytest
import requests

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

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"{self.status_code} Error")


class FakeSession:
    """Records calls and replays a scripted list of responses (an Exception entry is raised)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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


def test_retries_on_503_then_succeeds(no_sleep):
    session = FakeSession([
        FakeResponse(status_code=503, json_data={"served-by": "cp1"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}  # did NOT return the 503 body
    assert len(session.calls) == 2  # retried once


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retryable_5xx_statuses(status, no_sleep):
    session = FakeSession([
        FakeResponse(status_code=status, json_data={"transient": True}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert len(session.calls) == 2


def test_5xx_honors_retry_after(no_sleep):
    session = FakeSession([
        FakeResponse(status_code=503, headers={"Retry-After": "9"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert 9 in no_sleep  # honored Retry-After on a 5xx too


def test_5xx_with_non_json_body_is_retried_not_leaked(no_sleep):
    """A 5xx with a non-JSON body must not leak a raw JSONDecodeError."""

    class NonJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    session = FakeSession([
        NonJsonResponse(status_code=502),  # e.g. an HTML 502 gateway page
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert len(session.calls) == 2


def test_5xx_non_json_body_exhausts_to_runtimeerror(no_sleep):
    """Persistent 5xx non-JSON body exhausts to RuntimeError, not JSONDecodeError."""

    class NonJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("no json here")

    session = FakeSession([NonJsonResponse(status_code=503) for _ in range(6)])
    client = WikiClient(CFG, session=session)

    with pytest.raises(RuntimeError):
        client.get({"action": "query"}, max_retries=6)


def test_non_json_200_is_retried_not_raised(no_sleep):
    """A 200 OK carrying a non-JSON body (captive portal, proxy interstitial, Cloudflare challenge
    page) must be treated as transient and retried, not let a raw JSONDecodeError escape and abort
    a multi-hour pull."""
    import json

    class NonJsonOkResponse(FakeResponse):
        def json(self):
            raise json.JSONDecodeError("Expecting value", "", 0)

    session = FakeSession([
        NonJsonOkResponse(status_code=200),
        FakeResponse(status_code=200, json_data={"parse": {"wikitext": "..."}}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"parse": {"wikitext": "..."}}
    assert len(session.calls) == 2  # retried instead of raising


@pytest.mark.parametrize("status", [400, 404])
def test_terminal_4xx_is_surfaced_not_returned(status, no_sleep):
    """A terminal 4xx must raise, not be returned as a 'successful' dict."""
    session = FakeSession([
        FakeResponse(status_code=status, json_data={"error": "nope"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    with pytest.raises(requests.HTTPError):
        client.get({"action": "query"})

    assert len(session.calls) == 1  # did not retry a terminal 4xx


def test_http_date_retry_after_falls_back_to_backoff(no_sleep):
    """`Retry-After` may legally be an HTTP-date; that must not crash a multi-hour pull. The client
    falls back to the computed backoff (first retry: 3s) instead of raising ValueError."""
    session = FakeSession([
        FakeResponse(status_code=429,
                     headers={"Retry-After": "Fri, 10 Jul 2026 12:00:00 GMT"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert 3 in no_sleep  # fell back to the initial backoff, did not raise
    assert len(session.calls) == 2


def test_retry_after_wait_is_clamped(no_sleep):
    """A huge server-supplied Retry-After must not park the pull for hours: clamp to the cap."""
    from xeno_rag.api_client import MAX_RETRY_WAIT_SECONDS

    session = FakeSession([
        FakeResponse(status_code=503, headers={"Retry-After": "86400"}),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert MAX_RETRY_WAIT_SECONDS in no_sleep
    assert all(s <= MAX_RETRY_WAIT_SECONDS for s in no_sleep)


def test_connection_error_is_retried_then_succeeds(no_sleep):
    """A transient network failure gets backoff + retry, not an immediate crash: long unattended
    pulls should survive a blip mid-run."""
    session = FakeSession([
        requests.ConnectionError("connection reset by peer"),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    data = client.get({"action": "query"})

    assert data == {"ok": True}
    assert 3 in no_sleep  # backed off before retrying
    assert len(session.calls) == 2


def test_timeout_is_retried_then_succeeds(no_sleep):
    session = FakeSession([
        requests.Timeout("read timed out"),
        FakeResponse(json_data={"ok": True}),
    ])
    client = WikiClient(CFG, session=session)

    assert client.get({"action": "query"}) == {"ok": True}
    assert len(session.calls) == 2


def test_persistent_connection_errors_exhaust_to_runtimeerror(no_sleep):
    session = FakeSession([requests.ConnectionError("down") for _ in range(6)])
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
