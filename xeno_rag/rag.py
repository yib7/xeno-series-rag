"""Retrieval-augmented generation: retrieve grounded context, prompt an LLM, cite sources.

The LLM is pluggable. `GeminiClient` is the default (reads credentials from the environment at call
time and never embeds a key); `MockLLM` is used in tests. Generation is grounded: the system prompt
forbids inventing facts and requires citing sources.
"""

import os

from .config import load_config
from .embed_index import query

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the Xeno video game series "
    "(Xenogears, Xenosaga, Xenoblade Chronicles). "
    "Answer ONLY using the provided context. If the context does not contain the answer, say you "
    "do not know rather than guessing. Do not invent mechanics or numbers. "
    "Prefer the structured infobox entries for stats and numeric questions. "
    "Cite the source URLs you relied on at the end of your answer."
)


class MockLLM:
    """Deterministic stand-in for tests. Records the last call."""

    def __init__(self, canned: str = "MOCK ANSWER"):
        self.canned = canned
        self.last_system = None
        self.last_prompt = None

    def generate(self, system: str, prompt: str) -> str:
        self.last_system = system
        self.last_prompt = prompt
        return self.canned


class GeminiClient:
    """Gemini adapter using the supported `google-genai` SDK.

    Reads GOOGLE_API_KEY (or GEMINI_API_KEY) at call time. Raises if absent so the autonomous
    pipeline never silently makes a billable call without credentials.
    """

    def __init__(self, cfg: dict):
        self.model = cfg.get("gemini_model", "gemini-1.5-flash")

    def generate(self, system: str, prompt: str) -> str:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini credentials not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) to enable "
                "live answers."
            )
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return resp.text


def build_prompt(question: str, chunks):
    """Return (system, user) prompt strings grounding the answer in the retrieved chunks."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['title']} ({c['game']})] {c['text']}\nSource: {c['url']}")
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user


def _dedupe_sources(chunks):
    seen = set()
    sources = []
    for c in chunks:
        url = c.get("url")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


def answer(question: str, cfg: dict = None, game_filter: str = None, k: int = None,
           llm=None, embedder=None) -> dict:
    """Retrieve context, generate a grounded answer, and return {answer, sources}."""
    if cfg is None:
        cfg = load_config()
    chunks = query(question, cfg, k=k, game_filter=game_filter, embedder=embedder)
    system, user = build_prompt(question, chunks)
    if llm is None:
        llm = GeminiClient(cfg)
    text = llm.generate(system, user)
    return {"answer": text, "sources": _dedupe_sources(chunks)}
