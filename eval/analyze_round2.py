"""Analyze eval/results_round2.json: tagging mismatches, retrieval gaps, compact answer dump.

Same logic as analyze.py but for the round-2 (niche/complex) set, and prints the full answer so
graders can judge correctness — not just retrieval. Run with PYTHONIOENCODING=utf-8.
"""
import json
import re
from pathlib import Path

results = json.loads(Path("eval/results_round2.json").read_text(encoding="utf-8"))

SUFFIX = [
    (re.compile(r"\(XG\)|\(XG[, )]"), "XG"),
    (re.compile(r"\(XCX\)|\(XCX[, )]"), "XCX"),
    (re.compile(r"\(XC1\)|\(XCDE\)|\(XC1DE\)"), "XC1"),
    (re.compile(r"\(XC2\)|\(XC2T\)"), "XC2"),
    (re.compile(r"\(XC3\)|\(XC3FR\)"), "XC3"),
]
XS_HINT = re.compile(r"\(XS\)|\(XS1\)|\(XS2\)|\(XS3\)|\(XSI\)|\(XSII\)|\(XSIII\)|\(XS1&2\)|/Gameplay \(XS")


def title_hint(title):
    for rx, g in SUFFIX:
        if rx.search(title):
            return g
    if XS_HINT.search(title):
        return "XS*"
    return None


for r in results:
    g = r["game_filter"]
    titles = [c["title"] for c in r["chunks"]]
    mismatches = []
    for c in r["chunks"]:
        h = title_hint(c["title"])
        if h is None:
            continue
        tag = c["game"]
        if h == "XS*":
            if tag in {"XG", "XC1", "XC2", "XC3", "XCX"}:
                mismatches.append(f"{c['title']!r} hint=Xenosaga tag={tag}")
        elif h != tag and tag != "series":
            mismatches.append(f"{c['title']!r} hint={h} tag={tag}")
    print(f"\n{'='*100}")
    print(f"[{r['idx']:2d}] filter={g}  Q: {r['question']}")
    print(f"     tags={r['tag_counts']}  foreign_base={r['foreign_base_tags']}  elapsed={r.get('elapsed_s')}s")
    if mismatches:
        print(f"     !! TAG-MISMATCH: " + " | ".join(mismatches))
    print(f"     titles: {titles}")
    ans = r["answer"] or "(NO ANSWER / ERROR: %s)" % r.get("error")
    print(f"     ANSWER: {ans}")
