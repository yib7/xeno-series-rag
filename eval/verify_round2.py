"""Verify the round-2 'XS' umbrella fix through the real hybrid retrieval path (no LLM calls).

Checks:
  1. XG 'Ether' query no longer returns Xenosaga 'Ether (XS)' / 'Ether Amp (XS1&2)' (the leak).
  2. XS1 'Ether' query STILL returns 'Ether (XS)' (now tagged XS, admitted by the XS filter).
  3. No XS-tagged chunk appears under ANY non-Xenosaga filter (XG / XC1 / XCX) across probe queries.
  4. A 'series' page with a real Xenoblade cameo (KOS-MOS) still reaches the XC2 filter.
"""
from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve

cfg = load_config()
emb = Embedder(cfg)


def titles_tags(q, g):
    chunks = retrieve(q, cfg, game_filter=g, embedder=emb)
    return [(c.get("title"), c.get("game")) for c in chunks]


def show(q, g):
    rows = titles_tags(q, g)
    print(f"\n[{g}] {q!r}")
    for t, tag in rows:
        print(f"   ({tag}) {t}")
    return rows


print("=" * 80)
xg_ether = show("How does the Ether system work in Xenogears?", "XG")
xs_leak = [t for t, tag in xg_ether if tag == "XS"]
ether_xs_in_xg = [t for t, _ in xg_ether if t in ("Ether (XS)", "Ether Amp (XS1&2)")]
print(f"  -> XS-tagged chunks under XG: {xs_leak}")
print(f"  -> Ether(XS)/Ether Amp under XG: {ether_xs_in_xg}  (want: none)")

print("=" * 80)
xs1_ether = show("What is Ether in Xenosaga?", "XS1")
ether_xs_in_xs1 = [t for t, _ in xs1_ether if t in ("Ether (XS)", "Ether Amp (XS1&2)")]
print(f"  -> Ether(XS)/Ether Amp under XS1: {ether_xs_in_xs1}  (want: present)")

print("=" * 80)
# Probe several non-Xenosaga filters for any XS leak
probes = [
    ("What are the Gnosis?", "XG"),
    ("Who is KOS-MOS?", "XC1"),
    ("What is the Zohar?", "XCX"),
    ("Tell me about Shion", "XC2"),
]
any_xs_leak = []
for q, g in probes:
    rows = titles_tags(q, g)
    leaks = [(t, tag) for t, tag in rows if tag == "XS"]
    print(f"\n[{g}] {q!r}  XS-leak={leaks}")
    any_xs_leak += leaks

print("=" * 80)
kos = titles_tags("Who is KOS-MOS?", "XC2")
kos_hit = [t for t, tag in kos if "KOS-MOS" in (t or "")]
print(f"\n[XC2] 'KOS-MOS' query -> KOS-MOS pages present (series still reaches XC2): {kos_hit[:3]}")

print("\n" + "=" * 80)
print("SUMMARY")
print(f"  XG Ether leak fixed (no Ether(XS) under XG): {not ether_xs_in_xg}")
print(f"  XS1 still sees Ether(XS):                    {bool(ether_xs_in_xs1)}")
print(f"  No XS leak into non-Xenosaga probes:         {not any_xs_leak}")
print(f"  KOS-MOS (series) still reaches XC2:          {bool(kos_hit)}")
