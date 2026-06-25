"""Spot re-run (live Gemini) after the 'XS' umbrella re-tag: confirm the XG Ether answer is cleaner
and Xenosaga answers do not regress. A handful of authorized flash-lite calls."""
import time

from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve
from xeno_rag.rag import build_prompt, GeminiClient

cfg = load_config()
emb = Embedder(cfg)
llm = GeminiClient(cfg)

PAIRS = [
    ("XG", "How does the Ether system work in Xenogears?"),                 # #4 fix target
    ("XS2", "What is a Zohar Emulator and how many Zohar Emulators are there?"),  # #13 regression check
    ("XS3", "What are the Testaments and who are they?"),                   # #16 regression check
    ("XS1", "Who is KOS-MOS?"),                                             # round-1 #6 (KOS-MOS = series) still ok
]

for g, q in PAIRS:
    chunks = retrieve(q, cfg, game_filter=g, embedder=emb)
    system, user = build_prompt(q, chunks)
    t0 = time.time()
    ans = llm.generate(system, user)
    dt = round(time.time() - t0, 1)
    tags = {}
    for c in chunks:
        tags[c.get("game")] = tags.get(c.get("game"), 0) + 1
    print("=" * 90)
    print(f"[{g}] {q}   ({dt}s)  tags={tags}")
    print(f"titles: {[c.get('title') for c in chunks]}")
    print(ans)
    print()
