"""FastAPI app: an SSE-streaming /ask endpoint and a static frontend with a game selector.

`create_app` takes an injectable `answer_fn` (defaults to rag.answer) so tests can run without a
model or live LLM. The answer is produced grounded, then streamed token-by-token over SSE, followed
by a final `sources` event carrying the cited URLs.
"""

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import load_config

STATIC = Path(__file__).parent / "static"

# Mirrors rag.py's history window (`_history_block` renders `history[-6:]`): only the last 6 turns
# ever reach the prompt, so a payload with more is either a bug or an attempt to inflate prompt cost.
MAX_HISTORY_TURNS = 6


def _make_rate_limiter(max_requests, window_s):
    """A small in-process sliding-window limiter keyed by client host. Returns ``allow(key) -> bool``.

    The /ask endpoint fans out to the paid Gemini API and the CPU cross-encoder, so an unbounded
    caller (a runaway script, or the server accidentally exposed beyond localhost) could burn API
    credits and pin a core. This caps requests per client without any external dependency. Passing
    ``max_requests=None`` disables it for a trusted single-user deployment."""
    if not max_requests or max_requests <= 0:
        return lambda key: True
    hits = defaultdict(deque)

    def allow(key):
        now = time.monotonic()
        dq = hits[key]
        while dq and dq[0] <= now - window_s:
            dq.popleft()
        if len(dq) >= max_requests:
            return False
        dq.append(now)
        return True

    return allow

# User-facing "Fast" / "Thinking" / "Scholar" map to these Gemini models. Only these are accepted
# from the client (an allowlist — never pass an arbitrary model string through to the API). "Scholar"
# (pro-preview) is the heavy, large-scope tier; its retrieval depth lives in config's answer_styles.
FAST_MODEL = "gemini-3.1-flash-lite"
THINKING_MODEL = "gemini-3.5-flash"
SCHOLAR_MODEL = "gemini-3.1-pro-preview"
ALLOWED_MODELS = {FAST_MODEL, THINKING_MODEL, SCHOLAR_MODEL}


class AskTurn(BaseModel):
    """One prior conversation turn. Typed (both fields required strings) so malformed items are
    rejected at the API boundary (422) instead of reaching rag.py's dict ``.get(...)`` and raising an
    AttributeError. Extra keys are ignored (pydantic default)."""
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    game: Optional[str] = None
    model: Optional[str] = None
    # Prior turns for follow-up context. Item-schema'd (AskTurn) and hard-capped at MAX_HISTORY_TURNS
    # to reject malformed items and bound prompt cost; the JS client self-caps at 6 so never hits it.
    history: Optional[list[AskTurn]] = Field(default=None, max_length=MAX_HISTORY_TURNS)


def _adapt_answer_fn(answer_fn):
    """Wrap a non-streaming ``answer_fn`` (-> {answer, sources}) into the (kind, payload) event
    protocol, so tests can still inject a plain function while the real app streams token-by-token."""
    def stream_fn(question, **kw):
        res = answer_fn(question, **kw)
        text = res.get("answer") or ""
        for i in range(0, len(text), 24):
            yield ("text", text[i:i + 24])
        yield ("sources", res.get("sources") or [])
    return stream_fn


def create_app(answer_fn=None, stream_fn=None, cfg=None,
               rate_limit_max=30, rate_limit_window_s=60) -> FastAPI:
    if stream_fn is None:
        if answer_fn is not None:
            stream_fn = _adapt_answer_fn(answer_fn)
        else:
            from .. import rag
            stream_fn = rag.answer_stream

    allow_request = _make_rate_limiter(rate_limit_max, rate_limit_window_s)

    app = FastAPI(title="Xeno Series Wiki RAG")

    @app.middleware("http")
    async def _revalidate_frontend(request, call_next):
        """Force the browser to revalidate the frontend code on every load.

        Starlette's StaticFiles sends only ETag / Last-Modified (no Cache-Control), so browsers apply
        *heuristic* freshness and can serve a stale render.js / index.html without revalidating — which
        silently masks frontend updates (e.g. the source-bubble size tiers: the backend streamed the
        tier data, but the browser kept running a pre-tier render.js). ``no-cache`` keeps the cache but
        requires a conditional request each load, so a 304 is returned when unchanged (fast) and fresh
        bytes the moment a file changes. Only the frontend code/assets are tagged; /ask is untouched."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # Serve per-game logos / key-art (and any other static assets) under /static/.
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        client_key = request.client.host if request.client else "unknown"
        if not allow_request(client_key):
            return JSONResponse(
                {"error": "Rate limit exceeded. Please wait a moment and try again."},
                status_code=429,
            )
        game_filter = req.game or None
        # rag.py consumes history via dict ``.get("question")``/``.get("answer")``, so hand it plain
        # dicts, not AskTurn objects (keeps rag.py unchanged and dict-based).
        history = [t.model_dump() for t in req.history] if req.history else None
        effective_cfg = cfg
        if req.model in ALLOWED_MODELS:
            base = cfg if cfg is not None else load_config()
            effective_cfg = {**base, "gemini_model": req.model}

        def event_stream():
            # Real streaming: tokens flow as the model produces them. Each text delta is JSON-encoded
            # so newlines / markdown survive the SSE transport (a raw newline is an event boundary);
            # the client concatenates the decoded slices and renders markdown. Sources arrive last;
            # any failure arrives as an `error` event so the connection never just drops.
            try:
                for kind, payload in stream_fn(req.question, cfg=effective_cfg,
                                               game_filter=game_filter, history=history):
                    if kind == "text":
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif kind == "sources":
                        yield f"event: sources\ndata: {json.dumps(payload)}\n\n"
                    elif kind == "error":
                        yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            except Exception:  # noqa: BLE001 - last-resort guard so the stream always closes cleanly
                yield f"event: error\ndata: {json.dumps('Unexpected server error. Please try again.')}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
