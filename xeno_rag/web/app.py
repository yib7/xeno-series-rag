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

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    game: Optional[str] = None


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
        result = answer_fn(req.question, cfg=cfg, game_filter=game_filter)

        def event_stream():
            for token in result["answer"].split():
                yield f"data: {token}\n\n"
            yield f"event: sources\ndata: {json.dumps(result['sources'])}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
