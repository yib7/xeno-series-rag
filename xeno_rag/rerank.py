"""Cross-encoder reranker: re-scores (query, passage) pairs to promote the on-topic page.

A bi-encoder (Qwen cosine) and BM25 both judge relevance cheaply but coarsely, so the canonical
subject page can sit just below ancillary look-alikes (music tracks, weapon SKUs). A cross-encoder
reads the query and passage *together* and is far more accurate at ordering, used here as the final
layer over the fused candidate set. Small CPU model by default (``ms-marco-MiniLM-L-6-v2``) since the
machine is AMD/CPU-only; swap to ``BAAI/bge-reranker-base`` via config for higher quality.
"""

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, cfg: dict | None = None, model=None):
        self.model_name = (cfg or {}).get("rerank_model", DEFAULT_MODEL)
        self._model = model  # injectable for tests; model weights load lazily on first use

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, items, text_key: str = "text"):
        """Return ``items`` reordered best-first by cross-encoder relevance to ``query``. Stable for
        ties; returns the list unchanged when empty. Each returned item carries its relevance score on
        ``_score`` so the source list can size by relevance without re-scoring."""
        if not items:
            return items
        model = self._ensure()
        scores = model.predict([[query, it.get(text_key, "")] for it in items])
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        for i in order:
            items[i]["_score"] = float(scores[i])
        return [items[i] for i in order]
