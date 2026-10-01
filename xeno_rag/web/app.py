"""FastAPI app: an SSE-streaming /ask endpoint and a static frontend with a game selector.

`create_app` takes an injectable `answer_fn` (defaults to rag.answer) so tests can run without a
model or live LLM. The answer is produced grounded, then streamed token-by-token over SSE, followed
by a final `sources` event carrying the cited URLs.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from ..config import load_config
from ..parse_wikitext import _BASE_GAMES

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# Sentinel marking the sync stream_fn generator's exhaustion when driven via next(it, sentinel)
# from the async event stream (StopIteration cannot cross a coroutine boundary).
_STREAM_DONE = object()

# Env-var flag (not YAML config): whether to trust `X-Forwarded-For` for rate-limit keying. Off by
# default, matching a directly-exposed deployment. Set only when this process sits behind a proxy
# that itself sets/overwrites XFF and is the sole path in. See `_client_key`'s docstring for why an
# unconditional trust would reopen the exact abuse the rate limiter exists to stop.
_TRUST_PROXY_ENV = "XENO_TRUST_PROXY"

# Env var: comma-separated Host header names the server answers to (a port is ignored; "*" turns the
# check off). Default is loopback only, which stops DNS rebinding: a web page on an attacker's domain
# whose name resolves to 127.0.0.1 can otherwise call this unauthenticated, credit-spending API from
# the visitor's browser and read the reply. Set it when serving on a LAN name or behind a proxy that
# forwards the original Host.
_ALLOWED_HOSTS_ENV = "XENO_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "testserver")

# Env-var flag: warm the heavy retrieval singletons at startup (lifespan) instead of inside the
# first /ask. Off by default so tests, dev restarts, and retrieval-free usage stay fast. The cold
# load is the Qwen embedder (~1.2GB), the reranker cross-encoder, the Chroma store open, and the
# BM25 sqlite open, which otherwise all land on the first question's latency.
_WARM_ENV = "XENO_WARM"


def _env_flag_enabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _warm_singletons(cfg=None):
    """Touch every retrieve-side cached singleton so the first /ask doesn't pay the cold load.

    Uses the same `_get_*` accessors the request path uses, so the process-wide caches (and their
    double-checked locks) are populated exactly once and the request path later hits only the cached
    fast path. Imported lazily so that `create_app(answer_fn=...)` (every test) stays fast without
    pulling sentence-transformers / chromadb; the module-level `app` (production) does import them
    through `rag`, which is fine because uvicorn serves it right away."""
    from .. import embed_index, retrieve

    effective = cfg if cfg is not None else load_config()
    embed_index._get_embedder(effective)
    embed_index._get_client(effective)
    retrieve._get_bm25(effective)          # returns None (dense-only) if the index isn't built
    if effective.get("use_reranker", True):
        retrieve._get_reranker(effective)


# Mirrors rag.py's history window (`_history_block` renders `history[-6:]`): only the last 6 turns
# ever reach the prompt, so a payload with more is either a bug or an attempt to inflate prompt cost.
MAX_HISTORY_TURNS = 6

# Per-string size caps. The rate limiter bounds request COUNT, not SIZE: without these, a single
# request could carry megabytes of "question"/"history" straight into a billable model prompt
# (`_history_block` passes answers whole). Generous multiples of any real question / model answer.
MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 20000

# Hard cap on a request body, enforced before the JSON is parsed. The per-field caps above only run
# after the whole body is in memory, so without this a client could stream an arbitrarily large body.
# The legitimate worst case (6 history turns at the caps plus a question, every character written as
# a 6-byte JSON escape) is under 1 MiB; a real request is a few hundred bytes.
MAX_BODY_BYTES = 1024 * 1024

# Absolute filesystem paths (a Windows drive path, or a POSIX path with at least two segments). A
# SetupError message is written to be shown, but a few of them name where the config or store was
# looked for, which tells an HTTP client the server's directory layout.
_ABS_PATH = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"<>|*?]*"
    r"|(?<![\w/.:])/(?:[^\s/'\"<>|*?]+/)+[^\s'\"<>|*?]*"
)


def _public_error(message):
    """The text of an ``error`` event with any absolute filesystem path replaced by ``<path>``."""
    return _ABS_PATH.sub("<path>", message) if isinstance(message, str) else message


def _host_name(header):
    """The host part of a ``Host`` header value, lowercased, without the port (IPv6 literals keep
    their brackets)."""
    value = (header or "").strip().lower()
    if value.startswith("["):
        return value[:value.find("]") + 1] if "]" in value else value
    return value.split(":", 1)[0]


def _resolve_allowed_hosts(allowed_hosts):
    """Explicit argument, else ``XENO_ALLOWED_HOSTS``, else loopback only. ``None`` means any host."""
    if allowed_hosts is None:
        raw = os.environ.get(_ALLOWED_HOSTS_ENV, "").strip()
        allowed_hosts = [h for h in raw.split(",") if h.strip()] if raw else list(DEFAULT_ALLOWED_HOSTS)
    names = {h.strip().lower() for h in allowed_hosts}
    return None if "*" in names else names


class _HostGuardMiddleware:
    """Answer 400 to a request whose Host header isn't an allowed name (DNS-rebinding defence)."""

    def __init__(self, app, allowed):
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.allowed is not None:
            host = ""
            for name, value in scope.get("headers", ()):
                if name == b"host":
                    host = value.decode("latin-1")
            if _host_name(host) not in self.allowed:
                await JSONResponse({"error": "Invalid host header."}, status_code=400)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


_TOO_LARGE_BODY = json.dumps({"error": "Request body too large."}).encode()


class _BodyLimitMiddleware:
    """Reject a request body over ``max_bytes`` with 413, by ``Content-Length`` up front and by a
    running count for chunked bodies that declare none. Pure ASGI so the count sits under whatever
    reads the body, and a rejected request never reaches the JSON parser or the rate limiter.

    Overflow aborts the read by raising out of ``receive``. FastAPI's body parser wraps any such
    exception into its own 400, so once the limit has tripped this middleware swaps whatever response
    the app produces for the 413."""

    def __init__(self, app, max_bytes=MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = None
        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
        if declared is not None and declared > self.max_bytes:
            await JSONResponse({"error": "Request body too large."}, status_code=413)(scope, receive, send)
            return
        received = 0
        overflowed = False
        started = False

        async def counting_receive():
            nonlocal received, overflowed
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    overflowed = True
                    raise _BodyTooLarge
            return message

        async def replacing_send(message):
            nonlocal started
            if not overflowed:
                started = True
                await send(message)
            elif message["type"] == "http.response.start" and not started:
                started = True
                await send({"type": "http.response.start", "status": 413, "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_TOO_LARGE_BODY)).encode())]})
                await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})
            # any further message from the aborted response is dropped

        try:
            await self.app(scope, counting_receive, replacing_send)
        except _BodyTooLarge:
            if not started:
                await replacing_send({"type": "http.response.start", "status": 500, "headers": []})


def _client_key(request, trust_proxy=False):
    """Derive the per-client rate-limit key.

    Trust model: ``X-Forwarded-For`` is a plain client-supplied HTTP header: anyone who can reach
    this process directly can set it to an arbitrary, freshly-random value on every request, minting
    a new rate-limit bucket each time and defeating the limiter entirely (the abuse case this limiter
    exists to stop; see `_make_rate_limiter`'s docstring). It is therefore ONLY consulted when
    ``trust_proxy`` is explicitly enabled, which the caller should only do when this process sits
    behind a proxy (nginx / Cloudflare / etc.) that overwrites/sets XFF itself and is not reachable
    directly by untrusted clients (i.e. the proxy is the only path in), so the header can't be spoofed
    end-to-end. When ``trust_proxy`` is False (the default, safe for a directly-exposed deployment),
    XFF is ignored entirely and the direct peer (``request.client.host``) keys the bucket.

    Precedence when trust IS enabled: XFF first, peer as fallback. Behind any TCP proxy the socket
    peer is always populated with the PROXY's own address (uvicorn fills ``scope["client"]`` from the
    peername), so if the peer won, every proxied user would collapse into the proxy's single shared
    bucket and the flag would do nothing. Returns None only when the caller is wholly unidentifiable,
    which the endpoint treats as a 400 rather than pooling it into a shared key."""
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            # X-Forwarded-For is "client, proxy1, proxy2"; the first hop is the originating client.
            # NOTE: this is an unvalidated, attacker-influenceable bucketing key, not a verified
            # identity: fine for spreading load fairly across real proxied clients, not for any
            # security decision.
            first = fwd.split(",")[0].strip()
            if first:
                return first
    if request.client is not None:
        return request.client.host
    return None


def _store_health(cfg):
    """Cheap status of the vector store / collection: exists + chunk count.

    Deliberately never touches the embedder (the ~1.2GB cold load), only the Chroma client, which
    is the same cached open the request path uses. The directory is checked first so a missing store
    reports "missing" instead of PersistentClient silently creating an empty one."""
    paths = cfg.get("paths", {}) or {}
    path = paths.get("vectorstore")
    name = cfg.get("collection_name", "xeno_wiki")
    if not path or not os.path.isdir(path):
        return {"status": "missing", "collection": name, "chunks": 0}
    try:
        from .. import embed_index

        client = embed_index._get_client(cfg)
        col = client.get_collection(name)  # get_, never get_or_create_: /health must not create
        return {"status": "ok", "collection": name, "chunks": col.count()}
    except Exception as exc:  # noqa: BLE001 - /health degrades, never 500s
        # The route is unauthenticated: report the failure class only, keep the text (which can carry
        # filesystem paths) in the server log.
        log.warning("health: vector store check failed: %s: %s", type(exc).__name__, exc)
        return {"status": "error", "collection": name, "chunks": 0, "detail": type(exc).__name__}


def _bm25_health(cfg):
    """Cheap status of the BM25 sqlite index: present + row count + mtime (read-only connect)."""
    path = (cfg.get("paths", {}) or {}).get("bm25",
                                            os.path.join("data", "vectorstore", "bm25.sqlite3"))
    try:
        st = os.stat(path)
    except OSError:
        return {"status": "missing", "rows": 0}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT count(*) FROM meta").fetchone()[0]
        finally:
            con.close()
        return {"status": "ok", "rows": rows, "mtime": int(st.st_mtime)}
    except Exception as exc:  # noqa: BLE001 - a corrupt/foreign file reports, never crashes
        log.warning("health: bm25 check failed: %s: %s", type(exc).__name__, exc)
        return {"status": "error", "rows": 0, "detail": type(exc).__name__}


class _ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that always closes its body generator.

    When the client drops mid-stream the failed ``send`` raises out of ``__call__`` while the async
    generator is suspended at a ``yield``; nothing resumes or closes it, so its ``finally`` (which
    closes the paid Gemini stream) would only run at garbage collection. Closing it here makes the
    release prompt. Closing an already-finished generator is a no-op."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def _make_rate_limiter(max_requests, window_s):
    """A small in-process sliding-window limiter keyed by client host. Returns ``allow(key) -> bool``.

    The /ask endpoint fans out to the paid Gemini API and the CPU cross-encoder, so an unbounded
    caller (a runaway script, or the server accidentally exposed beyond localhost) could burn API
    credits and pin a core. This caps requests per client without any external dependency. Passing
    ``max_requests=None`` disables it for a trusted single-user deployment."""
    if not max_requests or max_requests <= 0:
        disabled = lambda key: True
        disabled.hits = {}
        return disabled
    hits = defaultdict(deque)
    last_sweep = time.monotonic()
    # allow() runs concurrently on FastAPI's threadpool. The lock serializes the whole body: without
    # it, the sweep's `del hits[k]` can race another thread between its `hits[key]` lookup and its
    # append (the hit lands on an orphaned deque and is forgotten), and two threads can both pass the
    # `len(dq) >= max_requests` check before either appends, admitting more than the cap. The
    # critical section is microseconds of dict/deque work, irrelevant next to model latency.
    lock = threading.Lock()

    def allow(key):
        nonlocal last_sweep
        with lock:
            now = time.monotonic()
            # Bound memory: an idle one-time visitor's deque is never revisited by allow() (that key
            # is only touched when *it* makes a request), so per-key self-deletion can't reclaim it.
            # Once a full window has elapsed since the last sweep, walk every key, trim expired hits,
            # and drop any deque that emptied out. This caps the map at the hosts active within one
            # window, no matter how many distinct hosts have ever connected.
            if now - last_sweep >= window_s:
                for k, kdq in list(hits.items()):
                    while kdq and kdq[0] <= now - window_s:
                        kdq.popleft()
                    if not kdq:
                        del hits[k]
                last_sweep = now
            dq = hits[key]
            while dq and dq[0] <= now - window_s:
                dq.popleft()
            if len(dq) >= max_requests:
                return False
            dq.append(now)
            return True

    # Expose the closure-local map for tests/introspection without changing the return contract
    # (allow is still a plain `key -> bool` callable).
    allow.hits = hits
    return allow


class AskTurn(BaseModel):
    """One prior conversation turn. Typed (both fields required strings) so malformed items are
    rejected at the API boundary (422) instead of reaching rag.py's dict ``.get(...)`` and raising an
    AttributeError. Extra keys are ignored (pydantic default). Both strings are length-capped so a
    crafted history payload can't inflate prompt cost past what the turn cap alone bounds."""
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    answer: str = Field(max_length=MAX_ANSWER_CHARS)


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    game: str | None = None
    # Prior turns for follow-up context. Item-schema'd (AskTurn) and hard-capped at MAX_HISTORY_TURNS
    # to reject malformed items and bound prompt cost; the JS client self-caps at 6 so never hits it.
    history: list[AskTurn] | None = Field(default=None, max_length=MAX_HISTORY_TURNS)

    @field_validator("game")
    @classmethod
    def _validate_game(cls, v):
        """Reject any game code outside the eight canonical base codes, plus None/"" (the "Xeno
        Series" = all-games option in the frontend selector). The game code is checked against the
        known game list (`_BASE_GAMES`): it used to be accepted as arbitrary text, so an unknown code
        silently disabled filtering AND reflected the raw string into the model prompt."""
        if v is None or v == "" or v in _BASE_GAMES:
            return v
        raise ValueError(f"unknown game code: {v!r}")


def _adapt_answer_fn(answer_fn):
    """Wrap a non-streaming ``answer_fn`` (-> {answer, sources}) into the (kind, payload) event
    protocol, so tests can still inject a plain function while the real app streams token-by-token."""
    def stream_fn(question, **kw):
        res = answer_fn(question, **kw)
        if res.get("tier"):
            yield ("tier", {"tier": res["tier"], "source": "auto"})
        text = res.get("answer") or ""
        for i in range(0, len(text), 24):
            yield ("text", text[i:i + 24])
        yield ("sources", res.get("sources") or [])
    return stream_fn


def create_app(answer_fn=None, stream_fn=None, cfg=None,
               rate_limit_max=30, rate_limit_window_s=60, trust_proxy=None, allowed_hosts=None) -> FastAPI:
    if stream_fn is None:
        if answer_fn is not None:
            stream_fn = _adapt_answer_fn(answer_fn)
        else:
            from .. import rag
            stream_fn = rag.answer_stream

    allow_request = _make_rate_limiter(rate_limit_max, rate_limit_window_s)
    # None (the default) means "not explicitly passed" -> resolve from the environment; an explicit
    # True/False from the caller always wins (lets tests force either branch deterministically).
    effective_trust_proxy = _env_flag_enabled(_TRUST_PROXY_ENV) if trust_proxy is None else trust_proxy

    @asynccontextmanager
    async def _lifespan(app):
        # Opt-in startup warmup (XENO_WARM=1): move the cold model/store load from the first /ask
        # to boot. The flag is read here (startup time), not at create_app time, so a deployment
        # can flip it without code changes and tests can monkeypatch the environment deterministically.
        if _env_flag_enabled(_WARM_ENV):
            log.info("XENO_WARM enabled: warming embedder / store / BM25 / reranker at startup...")
            try:
                _warm_singletons(cfg)
                log.info("warmup complete.")
            except Exception as exc:  # noqa: BLE001 - warmup is an optimization, never a boot blocker
                log.warning("startup warmup failed (continuing; first /ask will cold-load): %s", exc)
        yield

    app = FastAPI(title="Xeno Series Wiki RAG", lifespan=_lifespan)
    app.add_middleware(_BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(_HostGuardMiddleware, allowed=_resolve_allowed_hosts(allowed_hosts))

    @app.middleware("http")
    async def _revalidate_frontend(request, call_next):
        """Force the browser to revalidate the frontend code on every load.

        Starlette's StaticFiles sends only ETag / Last-Modified (no Cache-Control), so browsers apply
        *heuristic* freshness and can serve a stale render.js / index.html without revalidating, which
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

    @app.get("/health")
    def health():
        """Monitoring / smoke target: store + BM25 status and app version, via cheap checks only
        (a stat, a read-only sqlite count, the cached Chroma open, never the embedder). Always 200
        with a JSON body; "status" is "ok" only when both retrieval legs are serviceable."""
        from .. import __version__

        try:
            effective = cfg if cfg is not None else load_config()
        except Exception as exc:  # noqa: BLE001 - even a broken config must yield a readable body
            log.warning("health: config unusable: %s", exc)
            return {"status": "degraded", "version": __version__,
                    "error": f"config: {type(exc).__name__}", "store": {"status": "unknown"},
                    "bm25": {"status": "unknown"}}
        store = _store_health(effective)
        bm25 = _bm25_health(effective)
        ok = store["status"] == "ok" and bm25["status"] == "ok"
        return {"status": "ok" if ok else "degraded", "version": __version__,
                "store": store, "bm25": bm25}

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        client_key = _client_key(request, trust_proxy=effective_trust_proxy)
        if client_key is None:
            # Unidentifiable caller: no direct peer address, and either XFF trust is disabled or no
            # X-Forwarded-For header was sent. Can't rate limit per client, so reject rather than pool
            # everyone into one shared bucket. A proxy deployment must set XENO_TRUST_PROXY=1 (only
            # when this process is unreachable except through that proxy) and forward XFF.
            return JSONResponse(
                {"error": "Could not identify client for rate limiting (missing X-Forwarded-For)."},
                status_code=400,
            )
        if not allow_request(client_key):
            return JSONResponse(
                {"error": "Rate limit exceeded. Please wait a moment and try again."},
                status_code=429,
            )
        game_filter = req.game or None
        # rag.py consumes history via dict ``.get("question")``/``.get("answer")``, so hand it plain
        # dicts, not AskTurn objects (keeps rag.py unchanged and dict-based).
        history = [t.model_dump() for t in req.history] if req.history else None

        async def event_stream():
            # Real streaming: tokens flow as the model produces them. Each text delta is JSON-encoded
            # so newlines / markdown survive the SSE transport (a raw newline is an event boundary);
            # the client concatenates the decoded slices and renders markdown. Sources arrive last;
            # any failure arrives as an `error` event so the connection never just drops.
            #
            # Async on purpose: between chunks the client's disconnect state is checked, so an
            # abandoned answer (the user hit Stop / closed the tab) stops pulling from the paid
            # Gemini stream instead of burning tokens to the end. The sync stream_fn generator's
            # next() runs in the threadpool (it blocks on the model), keeping the event loop free.
            it = stream_fn(req.question, cfg=cfg,
                           game_filter=game_filter, history=history)
            try:
                while True:
                    if await request.is_disconnected():
                        log.info("client disconnected mid-stream; stopping generation early.")
                        break
                    # Sentinel instead of StopIteration: a StopIteration raised into a coroutine
                    # is a RuntimeError, so exhaustion is signalled by value.
                    item = await run_in_threadpool(next, it, _STREAM_DONE)
                    if item is _STREAM_DONE:
                        break
                    kind, payload = item
                    if kind == "text":
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif kind == "sources":
                        yield f"event: sources\ndata: {json.dumps(payload)}\n\n"
                    elif kind == "tier":
                        yield f"event: tier\ndata: {json.dumps(payload)}\n\n"
                    elif kind == "error":
                        yield f"event: error\ndata: {json.dumps(_public_error(payload))}\n\n"
            except Exception:
                log.exception("unexpected error while streaming /ask")
                yield f"event: error\ndata: {json.dumps('Unexpected server error. Please try again.')}\n\n"
            finally:
                # Close the generator so its finally/GeneratorExit path releases the model stream.
                close = getattr(it, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:  # noqa: BLE001, S110 - closing an abandoned stream is best-effort
                        pass

        return _ClosingStreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
