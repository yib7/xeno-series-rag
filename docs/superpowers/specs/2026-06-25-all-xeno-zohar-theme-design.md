# All-Xeno "Zohar" Theme — Design

**Date:** 2026-06-25
**Scope:** Give the "All games" filter (`data-game="all"`) its own distinct visual identity, matching
the polish of the per-game themes, using the Zohar — the one icon that recurs across all three
sub-series (Zohar in Xenogears/Xenosaga, Conduit/Core Crystals in Xenoblade).

## Problem

The per-game themes each get a real logo, key-art wash, box-art palette, and a matched display font.
The "All games" state is the un-themed fallback: a plain `<h1>Xeno Series Wiki RAG</h1>`, no wash, a
generic indigo/purple palette, and a system sans font. It looks unfinished next to every game.

## Approved decisions

- **Crisp custom SVG** for the Zohar mark (not a sourced raster) — razor-sharp at any size, themes
  with the palette vars, no scraping/optimize step.
- **Subtle** treatment — the Zohar glyph in the header + a faint cosmic halo, NOT a large monolith
  dominating the background.

## Design

All changes live in `xeno_rag/web/static/index.html`. No backend, no new image assets, no `render.js`
change.

1. **Brand mark (the centerpiece).** An inline SVG Zohar glyph — gold cross-shaped monolith, glowing
   turquoise core "eye", thin radiating etchings — paired with a "Xeno Series" wordmark. Rendered in
   the header for `data-game="all"`, replacing the plain `<h1>`. The SVG fills use `var(--accent)`
   (gold) and `var(--accent-2)` (turquoise) so it stays consistent with the palette. A soft CSS glow
   (`drop-shadow`) gives it presence, mirroring the per-game logo drop-shadow — no PNG `whiten`/`glow`
   path needed since it's vector.

2. **Palette (all/base theme).** Override the base accent vars to the Zohar pairing:
   `--accent: #e7c25e` (gold), `--accent-2: #34d8c8` (turquoise), `--accent-ink: #1a1206` (dark ink on
   the gold/cyan pills). Because the badge, accent bar, `Ask` button, links, focus rings, source chips,
   and example cards all already reference these vars, the new palette propagates automatically. This
   gold+turquoise pairing is used by no single game, so "All Xeno" reads as distinct, not a clone.

3. **Display font.** Add `Marcellus SC` (fallback `Marcellus`, then serif) to the existing Google
   Fonts import and set it as `--font-display` for the all/base theme. An elegant classical/engraved
   face that feels like it sits *above* the individual games (the Zohar's Gnostic/mythic tone), and
   collides with no per-game font. Applied only to brand/badge/field-labels/answer headings per the
   existing pattern; body text stays a readable sans.

4. **Cosmic wash (subtle).** Extend the all/base background with a layered gold→turquoise radial halo
   plus a faint, low-opacity CSS starfield (tiled radial dots) — pure CSS, no raster. The per-game
   `.art-wash` image mechanism is untouched and still off for the all state.

## Integration points

- `index.html` CSS: base / `html[data-game="all"]` vars (palette + `--font-display`), the cosmic
  background layers, and a `.brand-zohar` sizing/glow rule.
- `index.html` JS `setBrand()`: the `!g` (all) branch builds the Zohar SVG + wordmark instead of the
  plain `<h1>`. `applyTheme()` is otherwise unchanged (it already sets `data-game="all"`).
- `index.html` `<head>`: append `Marcellus SC` to the font import URL.

## Scope / non-goals

- Only the "All games" (`data-game="all"`) state. Per-game themes are untouched.
- No changes to retrieval, answers, or any backend/Python.
- The Xenosaga "XS" umbrella tag is not currently a UI selector option, so it is out of scope.

## Verification

- Run the app, select "All games": Zohar glyph renders crisp, gold/turquoise palette applied across
  badge/button/accent-bar/links, `Marcellus SC` loads, halo is subtle and prose stays legible.
- Select a couple of games (e.g. XC1, XG): confirm no regression — per-game logo, wash, palette, font
  still apply.
- Per the project's UI-verification note: judge via computed styles on the live server and an offline
  render; in-harness screenshots time out.
