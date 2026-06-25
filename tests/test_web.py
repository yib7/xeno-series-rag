"""Tests for the FastAPI + SSE web layer. Injects a fake answer function (no model/LLM)."""

import json

from fastapi.testclient import TestClient

from xeno_rag.web.app import create_app


def _reconstruct_answer(sse_body: str) -> str:
    """Rebuild the streamed answer text from the SSE body's data events."""
    text = ""
    for block in sse_body.split("\n\n"):
        block = block.strip()
        if block.startswith("data:") and not block.startswith("event:"):
            payload = block[len("data:"):].strip()
            try:
                text += json.loads(payload)
            except (ValueError, TypeError):
                text += payload  # tolerate a non-JSON (legacy) token
    return text


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


def test_ask_preserves_answer_structure():
    """Markdown structure (newlines, bullet lists) must survive streaming intact."""
    md = "Metal Face is a boss.\n\n- Colony 9 fight\n- Prison Island fight"

    def fake(question, **kw):
        return {"answer": md, "sources": []}

    client = TestClient(create_app(answer_fn=fake))
    r = client.post("/ask", json={"question": "q"})
    rebuilt = _reconstruct_answer(r.text)
    assert rebuilt == md                      # exact reconstruction, newlines and all
    assert "\n" in rebuilt
    assert "- Colony 9 fight" in rebuilt and "- Prison Island fight" in rebuilt


def test_ask_streams_event_protocol_from_stream_fn():
    """A streaming stream_fn (kind, payload) tuples map to SSE data/sources events, progressively."""
    def fake_stream(question, **kw):
        yield ("text", "Hello ")
        yield ("text", "world")
        yield ("sources", ["https://w/x"])

    client = TestClient(create_app(stream_fn=fake_stream))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200
    body = r.text
    assert _reconstruct_answer(body) == "Hello world"      # multiple text events concatenate
    assert "event: sources" in body and "https://w/x" in body


def test_ask_streams_error_event():
    def fake_stream(question, **kw):
        yield ("text", "partial answer")
        yield ("error", "the model is unavailable")

    client = TestClient(create_app(stream_fn=fake_stream))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "the model is unavailable" in r.text


def test_ask_passes_history_to_stream_fn():
    seen = {}

    def fake_stream(question, **kw):
        seen.update(kw)
        yield ("text", "a")
        yield ("sources", [])

    client = TestClient(create_app(stream_fn=fake_stream))
    hist = [{"question": "Who is Rex?", "answer": "The salvager protagonist."}]
    client.post("/ask", json={"question": "what about his weapon?", "history": hist})
    assert seen.get("history") == hist


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


def test_index_has_no_emojis():
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    for emoji in ("⚡", "🧠", "✨", "🎮", "🔥"):
        assert emoji not in body


def test_index_has_per_game_theming():
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    assert "data-game" in body                       # theme switches on selected game
    for code in ("XG", "XS1", "XS2", "XS3", "XC1", "XC2", "XC3", "XCX"):
        assert code in body                          # every game has a palette entry


def test_index_renders_markdown_client_side():
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    assert "renderMarkdown" in body                  # clean output, not raw **bold**


def test_static_art_is_served():
    """Per-game logo/key-art assets must be reachable under /static/art/."""
    client = TestClient(create_app(answer_fn=fake_answer))
    r = client.get("/static/art/xc3-logo.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_index_references_per_game_art():
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    assert "/static/art/" in body                    # references the art directory
    assert "art-wash" in body                         # background key-art wash layer
    assert "xc3-logo" in body                         # at least one sourced game logo wired in


ALL_CODES = ("xg", "xs1", "xs2", "xs3", "xc1", "xc2", "xc3", "xcx")


def test_index_wires_full_per_game_art_set():
    """Every game now ships a real logo + key-art banner — all 16 must be wired into the UI."""
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    for code in ALL_CODES:
        assert f"{code}-logo.png" in body, f"missing logo for {code}"
        assert f"{code}-bg.jpg" in body, f"missing key-art wash for {code}"


def test_static_serves_full_optimized_art_set():
    """The optimizer must have produced a reachable, image/* logo + bg for every game."""
    client = TestClient(create_app(answer_fn=fake_answer))
    for code in ALL_CODES:
        for asset in (f"{code}-logo.png", f"{code}-bg.jpg"):
            r = client.get(f"/static/art/{asset}")
            assert r.status_code == 200, f"{asset} not served"
            assert r.headers["content-type"].startswith("image/")


def test_frontend_assets_are_revalidated_not_cached():
    """Frontend code assets must carry ``Cache-Control: no-cache`` so a render.js / index.html update
    is never masked by a stale browser cache. This is the root cause of the recurring "source bubbles
    all look the same" report: the backend streamed tier'd sources, but the browser kept running a
    pre-tier render.js it had heuristically cached (Starlette's StaticFiles sets only ETag /
    Last-Modified, no Cache-Control). ``no-cache`` still permits fast 304 revalidation."""
    client = TestClient(create_app(answer_fn=fake_answer))
    for path in ("/", "/static/render.js", "/static/index.html"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        cc = r.headers.get("cache-control", "")
        assert "no-cache" in cc, f"{path} served without no-cache (Cache-Control={cc!r})"


def test_index_loads_fixed_fonts():
    """Two fixed faces (no jarring per-game switching): Cinzel = UI chrome, Spectral = chat/answers."""
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    assert "fonts.googleapis.com" in body            # web fonts loaded
    assert "--font-display" in body and "--font-read" in body   # UI vs reading font variables
    for font in ("Cinzel", "Spectral"):              # the two faces actually used
        assert font in body, f"font {font} not wired in"
    # per-game font switching was removed (it was jarring) -> the old game-specific faces are gone
    for font in ("Orbitron", "Fredoka", "Marcellus"):
        assert font not in body, f"stale per-game font {font} still present"
