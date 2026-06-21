"""Embed chunks and index them in ChromaDB.

The Embedder prefers DirectML (AMD GPU on Windows) via the ONNX backend and falls back to CPU if
DirectML / the ONNX runtime is unavailable, so embedding always completes. BGE models want a query
instruction prepended to *queries* only, not to stored documents.
"""

import json
import logging

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


class Embedder:
    def __init__(self, cfg: dict):
        self.model_name = cfg.get("embed_model", DEFAULT_MODEL)
        self.query_instruction = cfg.get("bge_query_instruction", "")
        self.model, self.backend = self._load(cfg.get("embed_device", "auto"))

    def _load(self, device: str):
        from sentence_transformers import SentenceTransformer

        if device in ("auto", "directml"):
            try:
                model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={"provider": "DmlExecutionProvider"},
                )
                log.info("Embedder using DirectML (ONNX) backend")
                return model, "directml"
            except Exception as exc:  # noqa: BLE001 - any failure -> CPU fallback
                log.warning("DirectML embedding unavailable (%s); falling back to CPU", exc)
        return SentenceTransformer(self.model_name, device="cpu"), "cpu"

    def encode(self, texts):
        """Embed documents (no query instruction). Returns a list of float lists."""
        embs = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return embs.tolist()

    def embed_query(self, text: str):
        embs = self.model.encode(
            [self.query_instruction + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return embs[0].tolist()


def _collection(cfg: dict, client=None):
    if client is None:
        client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=_CHROMA_SETTINGS)
    return client.get_or_create_collection(
        name=cfg.get("collection_name", "xeno_wiki"),
        metadata={"hnsw:space": "cosine"},
    )


def _metadata(chunk: dict) -> dict:
    return {
        "pageid": chunk["pageid"],
        "title": chunk["title"],
        "game": chunk["game"],
        "heading": chunk["heading"],
        "url": chunk["url"],
    }


def build_index(chunks, cfg: dict, embedder=None, client=None, batch_size: int = 256) -> int:
    """Embed chunks and add them to the persistent ChromaDB collection. Returns the collection count."""
    if embedder is None:
        embedder = Embedder(cfg)
    collection = _collection(cfg, client)

    batch = []

    def flush():
        if not batch:
            return
        ids = [c["chunk_id"] for c in batch]
        existing = set(collection.get(ids=ids)["ids"])
        new = [c for c in batch if c["chunk_id"] not in existing]
        if new:
            collection.add(
                ids=[c["chunk_id"] for c in new],
                embeddings=embedder.encode([c["text"] for c in new]),
                metadatas=[_metadata(c) for c in new],
                documents=[c["text"] for c in new],
            )
        batch.clear()

    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush()
    flush()
    return collection.count()


def _iter_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(cfg: dict, embedder=None) -> int:
    """Embed all chunks from cfg['paths']['chunks'] into the persistent store. Returns the count."""
    return build_index(_iter_chunks(cfg["paths"]["chunks"]), cfg, embedder=embedder)


def query(text: str, cfg: dict, k: int = None, game_filter: str = None, embedder=None, client=None):
    """Retrieve the top-k most similar chunks. Returns a list of result dicts."""
    if embedder is None:
        embedder = Embedder(cfg)
    if k is None:
        k = cfg.get("top_k", 8)
    collection = _collection(cfg, client)
    res = collection.query(
        query_embeddings=[embedder.embed_query(text)],
        n_results=k,
        where={"game": game_filter} if game_filter else None,
    )
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, cid in enumerate(ids):
        meta = metas[i] or {}
        out.append({
            "chunk_id": cid,
            "text": docs[i],
            "distance": dists[i] if i < len(dists) else None,
            **meta,
        })
    return out
