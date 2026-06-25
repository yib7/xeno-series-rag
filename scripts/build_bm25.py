"""Build the BM25 (SQLite-FTS5) lexical index from the LIVE ChromaDB collection, with a smoke check.

Thin wrapper around `Bm25Index.from_collection` (also runnable via `python -m xeno_rag.pipeline bm25`).
Sourcing text + game from the collection guarantees the BM25 game column matches the current
re-tagged metadata exactly.
"""
from xeno_rag.config import load_config
from xeno_rag.bm25_index import Bm25Index


def main():
    cfg = load_config()
    idx = Bm25Index.from_collection(cfg)
    print(f"BM25 index built: {idx.count} chunks -> {idx.path}")
    for q, gf in [("mimeosome", "XCX"), ("KOS-MOS android", "XS1"), ("Monado", "XC1")]:
        print(f"  [{gf}] {q!r} -> {idx.search(q, n=3, game_filter=gf)}")


if __name__ == "__main__":
    main()
