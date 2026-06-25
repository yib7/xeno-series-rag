"""Post-fix verification: re-run the affected questions and confirm the canonical page now appears."""
import time
from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder, query
from xeno_rag.rag import build_prompt, GeminiClient

cfg = load_config()
emb = Embedder(cfg)
llm = GeminiClient(cfg)

CASES = [
    ("XS1", "Who is KOS-MOS?", "KOS-MOS"),          # was broken (page tagged XC2)
    ("XS3", "Who is T-elos?", "T-elos"),            # was broken (page tagged XC2)
    ("XS1", "What is the Zohar in Xenosaga?", "Zohar (XS)"),  # was broken (Zohar(XS) tagged XG)
    ("XS2", "Who is Shion Uzuki?", "Shion"),        # bonus cross-over fix
    ("XCX", "What are mimeosomes?", "Mimeosome"),   # recall bug — expected STILL weak (control)
    ("XC1", "Who is Shulk?", "Shulk"),              # regression check
    ("XC2", "Who is Rex?", "Rex"),                  # regression check
]

for game, q, want in CASES:
    chunks = query(q, cfg, game_filter=game, embedder=emb)
    titles = [c.get("title") for c in chunks]
    tags = sorted({c.get("game") for c in chunks})
    hit = any((want or "").lower() == (t or "").lower() for t in titles)
    sys_, usr = build_prompt(q, chunks)
    ans = llm.generate(sys_, usr)
    print("=" * 90)
    print(f"[{game}] {q}")
    print(f"  canonical '{want}' retrieved? {hit}   chunk tags={tags}")
    print(f"  titles: {titles}")
    print(f"  ANSWER: {ans[:600]}")
    time.sleep(1)
