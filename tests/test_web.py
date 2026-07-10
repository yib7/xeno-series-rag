"""Tests for the FastAPI + SSE web layer. Injects a fake answer function (no model/LLM)."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from xeno_rag.web.app import STATIC, create_app

# Per-game art (logos / key art) is copyrighted and gitignored: it is fetched locally by
# scripts/fetch_art.py and never committed (see static/art/README.md). The two asset-serving tests
# below therefore only have files to serve on a maintainer's checkout where the art was fetched; on a
# clean clone or in CI the binaries are absent by design, so those tests skip instead of failing. The
# xc3 logo is the sentinel for "art has been populated in this checkout."
_ART_PRESENT = (STATIC / "art" / "xc3-logo.png").exists()
_needs_art = pytest.mark.skipif(
    not _ART_PRESENT,
    reason="per-game art is gitignored (fetch locally via scripts/fetch_art.py); absent on a clean checkout / CI",
)


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


def test_ask_rejects_malformed_history_item():
    """A non-conforming history item (not {question, answer} of strings) must be rejected by pydantic
    validation as a 422, never reach rag's dict ``.get(...)`` and blow up as a 500/AttributeError."""
    client = TestClient(create_app(answer_fn=fake_answer))
    for bad in (["notadict"], [{"question": 123, "answer": "x"}], [{"answer": "no question"}]):
        r = client.post("/ask", json={"question": "hi", "history": bad})
        assert r.status_code == 422, f"expected 422 for {bad!r}, got {r.status_code}"


def test_ask_rejects_over_cap_history():
    """History is hard-capped at MAX_HISTORY_TURNS turns (mirrors rag.py's 6-turn window). An
    over-cap payload is rejected (422) so a crafted client cannot inflate prompt cost; legit clients
    self-cap at 6 and never hit this."""
    client = TestClient(create_app(answer_fn=fake_answer))
    hist = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    r = client.post("/ask", json={"question": "hi", "history": hist})
    assert r.status_code == 422


def test_ask_rejects_over_length_question():
    """The question string itself is capped (2000 chars): the rate limiter bounds request COUNT, so
    without a size cap a single request could still carry megabytes straight into a paid model call.
    Over-length -> 422 at the pydantic boundary, before any retrieval or model work."""
    client = TestClient(create_app(answer_fn=fake_answer))
    r = client.post("/ask", json={"question": "q" * 2001})
    assert r.status_code == 422
    assert client.post("/ask", json={"question": "q" * 2000}).status_code == 200


def test_ask_rejects_over_length_history_strings():
    """History turns are size-capped too (question 2000 / answer 20000): rag's `_history_block`
    passes prior answers whole into the prompt, so an uncapped history string is the same prompt-cost
    hole as an uncapped question, just one level down."""
    client = TestClient(create_app(answer_fn=fake_answer))
    over_q = [{"question": "q" * 2001, "answer": "a"}]
    over_a = [{"question": "q", "answer": "a" * 20001}]
    for bad in (over_q, over_a):
        r = client.post("/ask", json={"question": "hi", "history": bad})
        assert r.status_code == 422, f"expected 422 for over-length history string, got {r.status_code}"
    at_cap = [{"question": "q" * 2000, "answer": "a" * 20000}]
    assert client.post("/ask", json={"question": "hi", "history": at_cap}).status_code == 200


def test_ask_accepts_valid_history():
    """A couple of well-formed turns still flow through end-to-end (multi-turn preserved), and the
    injected stream fn receives them as plain dicts."""
    seen = {}

    def fake_stream(question, **kw):
        seen.update(kw)
        yield ("text", "a")
        yield ("sources", [])

    client = TestClient(create_app(stream_fn=fake_stream))
    hist = [
        {"question": "Who is Rex?", "answer": "The salvager protagonist."},
        {"question": "His weapon?", "answer": "The Aegis sword."},
    ]
    r = client.post("/ask", json={"question": "and Pyra?", "history": hist})
    assert r.status_code == 200
    assert seen.get("history") == hist          # received as plain dicts, rag's .get(...) still works


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


def test_ask_overrides_model_with_scholar_pro():
    """The top-tier "Scholar" model id is on the allowlist and passes through to the answer fn."""
    seen = {}

    def fake(question, **kw):
        seen["cfg"] = kw.get("cfg")
        return {"answer": "a", "sources": []}

    client = TestClient(create_app(answer_fn=fake, cfg={"gemini_model": "default-model"}))
    client.post("/ask", json={"question": "q", "model": "gemini-3.1-pro-preview"})
    assert seen["cfg"]["gemini_model"] == "gemini-3.1-pro-preview"


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
    assert 'id="model"' in r.text    # model (Fast/Thinking/Scholar) selector present
    assert "Fast" in r.text and "Thinking" in r.text and "Scholar" in r.text


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


@_needs_art
def test_static_art_is_served():
    """Per-game logo/key-art assets must be reachable under /static/art/ (when fetched locally)."""
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


@_needs_art
def test_static_serves_full_optimized_art_set():
    """The optimizer must have produced a reachable, image/* logo + bg for every game (when fetched)."""
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
    """Two fixed faces (no jarring per-game switching): Cinzel = UI chrome, Spectral = chat/answers.
    Self-hosted from /static/fonts/ (P2-11) — no Google CDN reference may remain: offline (the
    local-first promise) CDN faces never load, and every page view would leak to a third party."""
    client = TestClient(create_app(answer_fn=fake_answer))
    body = client.get("/").text
    assert "fonts.googleapis.com" not in body and "fonts.gstatic.com" not in body
    assert "@font-face" in body                      # self-hosted faces declared inline
    assert "/static/fonts/cinzel-latin-wght.woff2" in body
    for w in (400, 500, 600, 700):
        assert f"/static/fonts/spectral-latin-{w}.woff2" in body
    assert "--font-display" in body and "--font-read" in body   # UI vs reading font variables
    # the referenced faces are actually served (not a dangling url() after a bad move/rename)
    r = client.get("/static/fonts/cinzel-latin-wght.woff2")
    assert r.status_code == 200 and r.content[:4] == b"wOF2"
    for font in ("Cinzel", "Spectral"):              # the two faces actually used
        assert font in body, f"font {font} not wired in"
    # per-game font switching was removed (it was jarring) -> the old game-specific faces are gone
    for font in ("Orbitron", "Fredoka", "Marcellus"):
        assert font not in body, f"stale per-game font {font} still present"


# ---- startup warmup (XENO_WARM lifespan hook) ----

def _patch_warm_loaders(monkeypatch, called):
    """Replace the four heavy singleton loaders with recorders (no model / store is ever touched)."""
    from xeno_rag import embed_index, retrieve

    monkeypatch.setattr(embed_index, "_get_embedder", lambda cfg: called.append("embedder"))
    monkeypatch.setattr(embed_index, "_get_client", lambda cfg: called.append("client"))
    monkeypatch.setattr(retrieve, "_get_bm25", lambda cfg: called.append("bm25"))
    monkeypatch.setattr(retrieve, "_get_reranker", lambda cfg: called.append("reranker"))


def test_lifespan_warms_singletons_when_flag_set(monkeypatch):
    """With XENO_WARM=1, the lifespan hook touches every retrieve-side singleton at startup so the
    first /ask hits only warm caches (the cold load — Qwen ~1.2GB + reranker + Chroma + BM25 — moves
    to boot). TestClient runs the lifespan only as a context manager."""
    called = []
    _patch_warm_loaders(monkeypatch, called)
    monkeypatch.setenv("XENO_WARM", "1")
    with TestClient(create_app(answer_fn=fake_answer, cfg={"use_reranker": True})):
        pass
    assert set(called) == {"embedder", "client", "bm25", "reranker"}


def test_lifespan_skips_warmup_by_default(monkeypatch):
    """Without the opt-in flag (default), startup must NOT load anything heavy — tests and dev
    restarts stay fast, and the first /ask pays the cold load as before."""
    called = []
    _patch_warm_loaders(monkeypatch, called)
    monkeypatch.delenv("XENO_WARM", raising=False)
    with TestClient(create_app(answer_fn=fake_answer)):
        pass
    assert called == []


def test_lifespan_warmup_respects_use_reranker_off(monkeypatch):
    """A config with the reranker disabled must not load the cross-encoder during warmup."""
    called = []
    _patch_warm_loaders(monkeypatch, called)
    monkeypatch.setenv("XENO_WARM", "1")
    with TestClient(create_app(answer_fn=fake_answer, cfg={"use_reranker": False})):
        pass
    assert "reranker" not in called and "embedder" in called


def test_lifespan_warmup_failure_does_not_block_boot(monkeypatch):
    """Warmup is an optimization: a failing loader (e.g. missing store) logs and continues, and the
    app still serves requests."""
    from xeno_rag import embed_index

    def boom(cfg):
        raise RuntimeError("store missing")

    monkeypatch.setattr(embed_index, "_get_embedder", boom)
    monkeypatch.setenv("XENO_WARM", "1")
    with TestClient(create_app(answer_fn=fake_answer)) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 200


def test_client_key_ignores_forwarded_for_when_proxy_not_trusted():
    """Default trust model (``trust_proxy=False``, matching a directly-exposed deployment): XFF is a
    plain client-supplied header, so a direct caller could set a fresh random value on every request
    to mint a new bucket each time and defeat the limiter entirely. It must be ignored, and a peer-less
    request must fall through to unidentifiable (None) rather than trusting the header."""
    from xeno_rag.web.app import _client_key

    class FakeReq:
        def __init__(self, client, headers):
            self.client = client
            self.headers = headers

    r = FakeReq(None, {"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    assert _client_key(r) is None                      # trust_proxy defaults to False
    assert _client_key(r, trust_proxy=False) is None    # explicit off: XFF still ignored


def test_client_key_uses_forwarded_for_when_proxy_trusted():
    """When ``trust_proxy=True`` (operator has confirmed this process sits behind a proxy that sets
    XFF itself and is the only path in), the first X-Forwarded-For hop is used so distinct proxied
    clients stay distinct instead of all collapsing into one shared bucket."""
    from xeno_rag.web.app import _client_key

    class FakeReq:
        def __init__(self, client, headers):
            self.client = client
            self.headers = headers

    r1 = FakeReq(None, {"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    r2 = FakeReq(None, {"x-forwarded-for": "198.51.100.4"})
    assert _client_key(r1, trust_proxy=True) == "203.0.113.7"
    assert _client_key(r2, trust_proxy=True) == "198.51.100.4"
    assert _client_key(r1, trust_proxy=True) != _client_key(r2, trust_proxy=True)


def test_client_key_prefers_forwarded_for_over_real_peer_when_trusted():
    """The deployment that actually motivates ``trust_proxy``: uvicorn behind a TCP reverse proxy.
    There the socket peer is ALWAYS populated (it is the proxy's own address, e.g. 127.0.0.1), so if
    the peer took precedence the XFF branch would be dead code and every proxied user would collapse
    into the proxy's single rate-limit bucket. With trust enabled and XFF present, the first XFF hop
    must win over a non-None peer; the peer is only the fallback when XFF is absent."""
    from xeno_rag.web.app import _client_key

    class FakeAddr:
        host = "127.0.0.1"  # the proxy's address, seen as the direct peer

    class FakeReq:
        def __init__(self, client, headers):
            self.client = client
            self.headers = headers

    r = FakeReq(FakeAddr(), {"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    assert _client_key(r, trust_proxy=True) == "203.0.113.7"
    # Trust on but no XFF (or a blank one): fall back to the direct peer, not None.
    assert _client_key(FakeReq(FakeAddr(), {}), trust_proxy=True) == "127.0.0.1"
    assert _client_key(FakeReq(FakeAddr(), {"x-forwarded-for": "  "}), trust_proxy=True) == "127.0.0.1"
    # Trust off: the header is ignored even when present; the peer keys the bucket.
    assert _client_key(r, trust_proxy=False) == "127.0.0.1"
    assert _client_key(r) == "127.0.0.1"


def test_ask_keys_rate_limit_by_forwarded_for_behind_real_peer():
    """End-to-end through /ask: the default TestClient supplies a real (non-None) peer address, the
    situation of every TCP proxy deployment. With trust_proxy=True and rate_limit_max=1, two requests
    carrying DIFFERENT XFF clients must both pass (distinct buckets), and repeating one of them must
    429 (same bucket) — proving the limiter keys on XFF, not on the shared peer address."""
    app = create_app(answer_fn=fake_answer, rate_limit_max=1, rate_limit_window_s=60,
                     trust_proxy=True)
    client = TestClient(app)  # default client: a fixed sentinel peer, like a proxy's address
    a = {"x-forwarded-for": "203.0.113.7"}
    b = {"x-forwarded-for": "198.51.100.4"}
    assert client.post("/ask", json={"question": "q"}, headers=a).status_code == 200
    assert client.post("/ask", json={"question": "q"}, headers=b).status_code == 200
    assert client.post("/ask", json={"question": "q"}, headers=a).status_code == 429


def test_client_key_unidentifiable_regardless_of_trust_flag():
    """No peer AND no forwarded header: cannot identify the caller -> None (endpoint rejects), whether
    or not proxy trust is enabled."""
    from xeno_rag.web.app import _client_key

    class FakeReq:
        def __init__(self, client, headers):
            self.client = client
            self.headers = headers

    assert _client_key(FakeReq(None, {})) is None
    assert _client_key(FakeReq(None, {}), trust_proxy=True) is None
    assert _client_key(FakeReq(None, {}), trust_proxy=False) is None


def test_ask_rejects_when_client_unidentifiable():
    """If a request has no direct peer and proxy trust is off (default), the endpoint can't rate
    limit per-client, so it rejects with 400 rather than pooling everyone into a shared bucket.

    This drives the real ASGI wiring end-to-end rather than mocking `_client_key`: Starlette's
    TestClient normally always sets `request.client` to a fixed sentinel address, but it accepts a
    `client=None` override that becomes the literal ASGI scope `client` key, so `request.client is
    None` is reached naturally (verified: FastAPI's `Request.client` is `None` under this scope, no
    header sent). No `X-Forwarded-For` header is sent, so `_client_key` really returns None."""
    client = TestClient(create_app(answer_fn=fake_answer), client=None)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 400


def test_ask_rejects_when_client_unidentifiable_and_xff_present_but_untrusted():
    """Same peer-less request as above, but this time WITH an X-Forwarded-For header attached and
    proxy trust left at its default (off). The header must be ignored end-to-end through the real
    endpoint (not just the `_client_key` helper), so the request still 400s instead of being keyed off
    an unauthenticated, spoofable header."""
    client = TestClient(create_app(answer_fn=fake_answer), client=None)
    r = client.post("/ask", json={"question": "hi"}, headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 400


def test_ask_uses_forwarded_for_when_proxy_trusted():
    """With `trust_proxy=True` wired through `create_app`, a peer-less request WITH an XFF header is
    accepted (keyed by the forwarded address) instead of 400ing — the flag actually reaches the /ask
    endpoint, not just the helper function."""
    client = TestClient(
        create_app(answer_fn=fake_answer, trust_proxy=True), client=None
    )
    r = client.post("/ask", json={"question": "hi"}, headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 200


def test_ask_rejects_when_proxy_trusted_but_nothing_identifiable():
    """`trust_proxy=True` but neither a peer nor an XFF header is present: still wholly unidentifiable,
    so still 400 (enabling the flag doesn't relax the "must identify somehow" requirement)."""
    client = TestClient(create_app(answer_fn=fake_answer, trust_proxy=True), client=None)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 400


def test_create_app_trust_proxy_defaults_from_env(monkeypatch):
    """`trust_proxy=None` (the `create_app` default) resolves from the `XENO_TRUST_PROXY` env var, so
    an operator can enable proxy trust via deployment config instead of code changes."""
    monkeypatch.setenv("XENO_TRUST_PROXY", "1")
    client = TestClient(create_app(answer_fn=fake_answer), client=None)
    r = client.post("/ask", json={"question": "hi"}, headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 200

    monkeypatch.setenv("XENO_TRUST_PROXY", "0")
    client = TestClient(create_app(answer_fn=fake_answer), client=None)
    r = client.post("/ask", json={"question": "hi"}, headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 400


def test_ask_rate_limited_after_threshold():
    """/ask triggers paid Gemini calls + CPU reranking, so it is rate-limited per client as abuse
    protection. Past the window's limit, further requests get a 429 instead of running."""
    app = create_app(answer_fn=fake_answer, rate_limit_max=2, rate_limit_window_s=60)
    client = TestClient(app)
    assert client.post("/ask", json={"question": "q1"}).status_code == 200
    assert client.post("/ask", json={"question": "q2"}).status_code == 200
    assert client.post("/ask", json={"question": "q3"}).status_code == 429


def test_ask_rate_limit_disabled_when_max_none():
    """rate_limit_max=None disables the limiter (trusted single-user deployments can opt out)."""
    app = create_app(answer_fn=fake_answer, rate_limit_max=None)
    client = TestClient(app)
    for _ in range(5):
        assert client.post("/ask", json={"question": "q"}).status_code == 200


def test_rate_limiter_evicts_idle_hosts(monkeypatch):
    """The per-host `hits` map must stay bounded: a one-time visitor whose window has expired is
    never revisited by `allow()`, so without a sweep its deque lingers forever. If the server is
    exposed, unbounded distinct source IPs would grow memory without bound. A periodic sweep (fired
    once `window_s` has elapsed since the last one) must trim and drop expired hosts, while a host
    still active within the window is retained."""
    from xeno_rag.web import app as app_mod

    clock = {"now": 1000.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["now"])

    limiter = app_mod._make_rate_limiter(max_requests=5, window_s=60)

    # Two distinct hosts connect once each at t=1000; both are recorded.
    assert limiter("1.1.1.1") is True
    assert limiter("2.2.2.2") is True
    assert set(limiter.hits) == {"1.1.1.1", "2.2.2.2"}

    # Advance past the window AND past the sweep interval, then a *new* host connects. This is the
    # only thing that revisits the map, so the sweep must run here and evict the two idle hosts.
    clock["now"] = 1000.0 + 60 + 1
    assert limiter("3.3.3.3") is True

    assert "1.1.1.1" not in limiter.hits, "expired idle host was not evicted"
    assert "2.2.2.2" not in limiter.hits, "expired idle host was not evicted"
    assert "3.3.3.3" in limiter.hits, "the currently-active host must be retained"


def test_rate_limiter_counts_correctly_under_concurrent_threads(monkeypatch):
    """`allow()` runs concurrently on FastAPI's threadpool, and the shared `hits` map is mutated by
    both the periodic sweep (`del hits[k]`) and per-key appends. Unsynchronized, a sweep can drop a
    deque between another thread's `hits[key]` lookup and its append (the hit lands on an orphaned
    deque and is forgotten), and two threads can both pass the `len(dq) >= max` check before either
    appends — admitting more than `max_requests`. This drives two threads through the sweep window
    against one shared key and asserts exact counting: precisely `max_requests` admissions, all of
    them recorded on the live deque."""
    from xeno_rag.web import app as app_mod

    clock = {"now": 100.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["now"])

    limiter = app_mod._make_rate_limiter(max_requests=64, window_s=60)
    # Prime an idle key, then cross the sweep boundary so the very first concurrent call fires the
    # sweep while the other thread is appending — the exact interleaving the lock must serialize.
    assert limiter("idle") is True
    clock["now"] += 61

    admitted = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        admitted.append(sum(1 for _ in range(64) if limiter("shared")))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 128 concurrent attempts against a cap of 64: exactly 64 admitted, none lost or extra, and the
    # live deque holds exactly the admitted hits (nothing recorded on a swept-away orphan).
    assert sum(admitted) == 64
    assert len(limiter.hits["shared"]) == 64


def test_rate_limiter_keeps_active_host_across_sweep(monkeypatch):
    """A host that keeps making requests inside the window must survive a sweep and still be rate
    limited correctly — the sweep bounds memory without dropping live state."""
    from xeno_rag.web import app as app_mod

    clock = {"now": 5000.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["now"])

    limiter = app_mod._make_rate_limiter(max_requests=3, window_s=60)

    # Idle host connects once and then goes away.
    assert limiter("idle") is True
    # Active host makes a request now...
    assert limiter("active") is True

    # ...advance past the sweep interval; the active host makes another request (triggering the
    # sweep). The idle host is evicted; the active host keeps its live timestamps and window.
    clock["now"] = 5000.0 + 61
    assert limiter("active") is True
    assert "idle" not in limiter.hits
    assert "active" in limiter.hits
    # Its window still enforces the cap: two more (total 3 in this window) then a 429-equivalent.
    assert limiter("active") is True
    assert limiter("active") is True
    assert limiter("active") is False
