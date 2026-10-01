"""Deterministic test doubles shared across the suite."""

import hashlib
import re

from xeno_rag.embed_index import _l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """A bag-of-words hashing embedder: the same interface as ``embed_index.Embedder`` (``encode`` for
    documents, ``embed_query`` for queries) with no model, no download, and no randomness.

    Each lowercase alphanumeric token is hashed (sha1, not the salted built-in ``hash``) into one of
    ``DIM`` buckets and the counts are L2-normalized, so texts that share words land close in cosine
    distance. That is enough for the retrieval tests, which assert on lexical overlap and on filter,
    cap, and fusion behaviour rather than on the quality of Qwen's semantic ranking. The real model is
    exercised once, in ``tests/test_embedder_model.py`` (marker ``model``, skipped by default).
    """

    DIM = 256

    def __init__(self, cfg=None):
        self.query_instruction = (cfg or {}).get("query_instruction", "")

    def _vector(self, text: str) -> list[float]:
        counts = [0.0] * self.DIM
        for token in _TOKEN.findall(text.lower()):
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.DIM
            counts[bucket] += 1.0
        return counts

    def encode(self, texts):
        return _l2_normalize([self._vector(t) for t in texts])

    def embed_query(self, text: str):
        return _l2_normalize([self._vector(text)])[0]
