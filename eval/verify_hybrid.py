"""Verify the hybrid (dense + BM25 + rerank) retriever on the questions that exposed recall misses,
plus regression checks. Uses the real config (use_bm25/use_reranker = true)."""
import time
from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve

cfg = load_config()
print(f"use_bm25={cfg.get('use_bm25')} use_reranker={cfg.get('use_reranker')} model={cfg.get('rerank_model')}")
emb = Embedder(cfg)

CASES = [
    ("XCX", "What are mimeosomes?", "Mimeosome"),          # the headline recall failure
    ("XCX", "What are Skells?", "Skell"),                  # was shallow
    ("XS1", "How does the Boost system work in battle?", None),  # was Boost-Quiz confusion
    ("XS1", "Who is KOS-MOS?", "KOS-MOS"),                 # tagging-fixed; confirm still good
    ("XC1", "Who is Shulk?", "Shulk"),                     # regression check
    ("XC2", "What are Blades and Drivers?", "Blade (XC2)"),# regression check
    ("XC3", "What is a flame clock?", "Flame Clock"),      # regression check
]

for game, q, want in CASES:
    t0 = time.time()
    res = retrieve(q, cfg, game_filter=game, embedder=emb)
    dt = time.time() - t0
    titles = [r["title"] for r in res]
    hit = (want is None) or any((want or "").lower() == (t or "").lower() for t in titles)
    print(f"\n[{game}] {q}   ({dt:.1f}s)   canonical={want!r} present={hit}")
    print(f"   titles: {titles}")
