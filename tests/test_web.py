"""Tests for the FastAPI + SSE web layer. Injects a fake answer function (no model/LLM)."""

from fastapi.testclient import TestClient

from xeno_rag.web.app import create_app


def fake_answer(question, **kw):
    return {"answer": "hello world", "sources": ["https://w/x"]}


def test_ask_streams_sse_with_sources():
    client = TestClient(create_app(answer_fn=fake_answer))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "hello" in body and "world" in body
    assert "event: sources" in body
    assert "https://w/x" in body


def test_ask_passes_game_filter():
    seen = {}

    def fake(question, **kw):
        seen.update(kw)
        return {"answer": "a", "sources": []}

    client = TestClient(create_app(answer_fn=fake))
    client.post("/ask", json={"question": "q", "game": "XC2"})
    assert seen.get("game_filter") == "XC2"


def test_empty_game_means_no_filter():
    seen = {}

    def fake(question, **kw):
        seen.update(kw)
        return {"answer": "a", "sources": []}

    client = TestClient(create_app(answer_fn=fake))
    client.post("/ask", json={"question": "q", "game": ""})
    assert seen.get("game_filter") is None


def test_ask_overrides_model_when_allowed():
    seen = {}

    def fake(question, **kw):
        seen["cfg"] = kw.get("cfg")
        return {"answer": "a", "sources": []}

    client = TestClient(create_app(answer_fn=fake, cfg={"gemini_model": "default-model"}))
    client.post("/ask", json={"question": "q", "model": "gemini-3.5-flash"})
    assert seen["cfg"]["gemini_model"] == "gemini-3.5-flash"


def test_ask_ignores_unknown_model():
    seen = {}

    def fake(question, **kw):
        seen["cfg"] = kw.get("cfg")
        return {"answer": "a", "sources": []}

    client = TestClient(create_app(answer_fn=fake, cfg={"gemini_model": "default-model"}))
    client.post("/ask", json={"question": "q", "model": "evil-model"})
    assert seen["cfg"]["gemini_model"] == "default-model"  # untrusted value ignored


def test_index_page_served():
    client = TestClient(create_app(answer_fn=fake_answer))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert 'id="game"' in r.text     # game selector present
    assert 'id="model"' in r.text    # model (Faster/Thinking) selector present
    assert "Faster" in r.text and "Thinking" in r.text
