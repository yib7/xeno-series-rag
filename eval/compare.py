"""Before/after diff of two 40-question runs: which questions changed retrieval/answer.

Usage: python eval/compare.py [before.json] [after.json]  (defaults: results_before.json, results.json)
"""
import json
import sys
from pathlib import Path

bpath = sys.argv[1] if len(sys.argv) > 1 else "eval/results_before.json"
apath = sys.argv[2] if len(sys.argv) > 2 else "eval/results.json"
before = {r["idx"]: r for r in json.loads(Path(bpath).read_text(encoding="utf-8"))}
after = {r["idx"]: r for r in json.loads(Path(apath).read_text(encoding="utf-8"))}

changed = 0
for idx in sorted(before):
    b, a = before[idx], after[idx]
    bt = set(b["chunks"][i]["title"] for i in range(len(b["chunks"])))
    at = set(a["chunks"][i]["title"] for i in range(len(a["chunks"])))
    btags, atags = b["tag_counts"], a["tag_counts"]
    new_titles = at - bt
    lost_titles = bt - at
    if new_titles or lost_titles or btags != atags:
        changed += 1
        print(f"\n[{idx:2d}] {a['game_filter']}  {a['question']}")
        print(f"     tags: {btags}  ->  {atags}")
        if new_titles:
            print(f"     + now retrieved: {sorted(new_titles)}")
        if lost_titles:
            print(f"     - no longer:     {sorted(lost_titles)}")
        # first sentence of each answer to eyeball quality shift
        b1 = (b["answer"] or "").strip().split("\n")[0][:160]
        a1 = (a["answer"] or "").strip().split("\n")[0][:160]
        if b1 != a1:
            print(f"     answer was: {b1}")
            print(f"     answer now: {a1}")

print(f"\n{changed}/40 questions changed retrieval or tags after the re-tag.")
