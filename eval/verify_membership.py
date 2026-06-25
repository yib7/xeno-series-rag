"""Verify multi-tag membership through the real hybrid retrieval path (no LLM).

For each cross-appearance page, query its name under EVERY base-game filter and assert the page is
retrievable under exactly its membership games and excluded everywhere else.
"""
from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve

cfg = load_config()
emb = Embedder(cfg)
GAMES = ["XG", "XS1", "XS2", "XS3", "XC1", "XC2", "XC3", "XCX"]

# (query, exact page title to look for, expected membership games)
CASES = [
    ("Who is KOS-MOS?", "KOS-MOS", {"XS1", "XS2", "XS3", "XC2"}),
    ("Who is T-elos?", "T-elos", {"XS3", "XC2"}),
    ("Who is Elma?", "Elma", {"XCX", "XC2"}),
    ("Who is Shulk?", "Shulk", {"XC1", "XC2", "XC3"}),
    ("Who is Shion Uzuki?", "Shion", {"XS1", "XS2", "XS3"}),
    ("Who is Pyra?", "Pyra", {"XC2"}),
    ("Who is Metal Face?", "Metal Face", {"XC1"}),
]

all_ok = True
for q, title, expected in CASES:
    present = set()
    for g in GAMES:
        chunks = retrieve(q, cfg, game_filter=g, embedder=emb)
        if any(c.get("title") == title for c in chunks):
            present.add(g)
    ok = present == expected
    all_ok &= ok
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {title:12s} present under {sorted(present)}  expected {sorted(expected)}")
    if not ok:
        print(f"        leaked into: {sorted(present - expected)}   missing from: {sorted(expected - present)}")

print("\nALL MEMBERSHIP CASES PASS" if all_ok else "\nSOME CASES FAILED")
