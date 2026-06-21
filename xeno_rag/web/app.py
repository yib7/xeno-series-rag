"""FastAPI app: an SSE-streaming /ask endpoint and a static frontend with a game selector.

`create_app` takes an injectable `answer_fn` (defaults to rag.answer) so tests can run without a
model or live LLM. The answer is produced grounded, then streamed token-by-token over SSE, followed
by a final `sources` event carrying the cited URLs.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import load_config

STATIC = Path(__file__).parent / "static"

# User-facing "Faster" vs "Thinking" maps to these Gemini models. Only these are accepted from the
# client (an allowlist — never pass an arbitrary model string through to the API).
FAST_MODEL = "gemini-3.1-flash-lite"
THINKING_MODEL = "gemini-3.5-flash"
ALLOWED_MODELS = {FAST_MODEL, THINKING_MODEL}


class AskRequest(BaseModel):
    question: str
    game: Optional[str] = None
    model: Optional[str] = None


def create_app(answer_fn=None, cfg=None) -> FastAPI:
    if answer_fn is None:
        from .. import rag
        answer_fn = rag.answer

    app = FastAPI(title="Xeno Series Wiki RAG")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/ask")
    def ask(req: AskRequest):
        game_filter = req.game or None
        effective_cfg = cfg
        if req.model in ALLOWED_MODELS:
            base = cfg if cfg is not None else load_config()
            effective_cfg = {**base, "gemini_model": req.model}
        result = answer_fn(req.question, cfg=effective_cfg, game_filter=game_filter)

        def event_stream():
            for token in result["answer"].split():
                yield f"data: {token}\n\n"
            yield f"event: sources\ndata: {json.dumps(result['sources'])}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
