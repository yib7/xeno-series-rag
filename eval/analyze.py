"""Analyze eval/results.json: surface tagging mismatches, retrieval gaps, and dump answers compactly."""
import json
import re
import sys
from pathlib import Path

# Map a title's disambiguator/suffix to the base game it implies (None if no clear hint).
SUFFIX = [
    (re.compile(r"\(XG\)|\(XG[, )]"), "XG"),
    (re.compile(r"\(XCX\)|\(XCX[, )]"), "XCX"),
    (re.compile(r"\(XC1\)|\(XCDE\)|\(XC1DE\)"), "XC1"),
    (re.compile(r"\(XC2\)|\(XC2T\)"), "XC2"),
    (re.compile(r"\(XC3\)|\(XC3FR\)"), "XC3"),
]
# Xenosaga sub-series hints (these span/short of base codes; "XS" alone = Xenosaga-wide).
XS_HINT = re.compile(r"\(XS\)|\(XS1\)|\(XS2\)|\(XS3\)|\(XSI\)|\(XSII\)|\(XSIII\)|\(XS1&2\)|/Gameplay \(XS")


def title_hint(title):
    for rx, g in SUFFIX:
        if rx.search(title):
            return g
    if XS_HINT.search(title):
        return "XS*"  # some Xenosaga entry
    return None


def subject_terms(q):
    """Rough main-subject tokens from a 'Who/What is X' question."""
    m = re.search(r"(?:who|what) (?:is|are) (?:the )?(.+?)\??$", q.lower())
    if not m:
        return []
    raw = m.group(1)
    raw = re.split(r"\band\b", raw)[0]
    return [w for w in re.findall(r"[a-z0-9\-]+", raw) if len(w) > 2]


def main():
    """Load eval/results.json (relative to the current working directory) and print the per-question
    tag-mismatch / retrieval-gap analysis. If the results file does not exist yet, print where it
    comes from and return, rather than raise."""
    results_path = Path("eval/results.json")
    if not results_path.exists():
        print("eval/results.json not found. Run `python -m eval.run_eval` to produce it.")
        return
    results = json.loads(results_path.read_text(encoding="utf-8"))
    sys.stdout.reconfigure(encoding="utf-8")   # answers hold non-ASCII text; a cp1252 console would raise

    for r in results:
        g = r["game_filter"]
        titles = [c["title"] for c in r["chunks"]]
        # tag mismatches: a chunk whose title clearly implies game X but is tagged a different base game
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
            elif h != tag and tag == "series" and h != "XS*":
                pass  # series is the safe fallback; not a hard mistag
        # main-subject retrieval: did any retrieved title contain the subject token(s)?
        terms = subject_terms(r["question"])
        main_hit = any(all(t in c["title"].lower() for t in terms) for c in r["chunks"]) if terms else None
        # exact main page (title == subject, ignoring disambiguators) present?
        print(f"\n{'='*100}")
        print(f"[{r['idx']:2d}] filter={g}  Q: {r['question']}")
        print(f"     tags={r['tag_counts']}  main_subject_page_hit={main_hit}  terms={terms}")
        if mismatches:
            print("     !! TAG-MISMATCH: " + " | ".join(mismatches))
        print(f"     titles: {titles}")
        ans = r["answer"] or "(no answer)"
        print(f"     ANSWER: {ans}")


if __name__ == "__main__":
    main()
