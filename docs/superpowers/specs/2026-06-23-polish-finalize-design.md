# Polish / Finalize Cycle — Design

**Date:** 2026-06-23
**Goal:** Harden the shipped Xeno-RAG chatbot — fix UI/RAG/prompt bugs surfaced by close inspection,
add the test coverage that would have caught them, and layer quality + interactive features
(streaming, tables, multi-turn, source previews) onto the working app.

**Scope chosen by user:** "Quality + bigger features" + ALL optional extras.

---

## Investigation findings (evidence-backed)

### Bugs (reproduced)

1. **[SEVERE] Markdown renderer renders every number as "undefined"** —
   `xeno_rag/web/static/index.html` `inline()` restore step:
   `s.replace(/(\d+)/g, (_, i) => stash[+i])` matches *prose* digits, not just its placeholder
   tokens. Reproduced with Node: `"deals 100 damage"` → `"deals undefined damage"`; even a string
   with no code/links (`"just 100 and 250"`) → `"just undefined and undefined"`. Hits virtually every
   answer on a stats wiki. Survived because the eval harness exercises `rag.answer` (Python) directly
   and never the browser renderer — **the frontend JS has zero tests.**

2. **[MED] Unhandled LLM-response failure** — `rag.py` `GeminiClient.generate` returns `resp.text`,
   which raises (or is `None`) when Gemini blocks a response or a "Thinking" model returns only
   thought parts → unhandled 500 → UI shows a bare "Error: HTTP 500".

3. **[MED] No server-side question validation** — `web/app.py` `/ask` accepts empty/whitespace
   questions (only the browser guards); a direct call retrieves on `""` and prompts the LLM with
   nothing.

4. **[LOW] Stale default model** — `rag.py` hardcodes the retired `gemini-1.5-flash` as the
   `GeminiClient` fallback.

5. **[LOW] Ungraceful error path** — retrieval/generation runs *before* the SSE stream opens, so
   failures surface as raw HTTP errors instead of an in-pane message.

### Quality opportunities

- **Real token streaming**: the answer is fully generated, then fake-streamed in 24-char slices, so
  time-to-first-token = full generation time. `generate_content_stream` makes "Thinking…" genuinely
  progressive and avoids O(n^2) re-render of the whole answer per slice.
- **Markdown tables**: renderer can't render tables; stat-comparison answers come out as raw `|`
  pipes. Pair with a prompt nudge to tabulate multi-stat comparisons.
- **Prompt tuning**: never tells the model the active game-filter scope or to prefer tight Markdown.

### Features requested

- Conversation **thread** + **multi-turn follow-ups** (history-aware prompt + light query expansion).
- Clickable **example questions** (empty state).
- **Copy-answer** button per answer.
- **Source snippet previews** (richer source payload: url + title + game + snippet).

---

## Architecture decisions

- **Frontend testability:** extract the Markdown/render logic from inline `<script>` into
  `static/render.js` exposed UMD-style (browser global + CommonJS export). Test with Node's built-in
  runner (`node --test`, no npm install). A `tests/test_frontend_js.py` shells out to `node --test`
  (skips if Node absent) so the JS suite runs inside `pytest` and the "full suite green" checkpoint
  stays meaningful.
- **Renderer fix:** replace numeric placeholders with a non-colliding sentinel
  (`\x00<idx>\x00`) so prose digits are never mistaken for stash indices; add a `| … |` table parser.
- **Streaming:** add `generate_stream(system, prompt) -> Iterator[str]` to the LLM protocol (Gemini
  uses `generate_content_stream`; Mock yields slices). New `answer_stream(...)` retrieves, streams
  tokens, then yields a final sources payload. `answer()` stays (consumes the stream) for CLI/tests.
  `/ask` streams real tokens wrapped in try/except → a friendly `error` SSE event on failure.
- **Richer sources:** sources become `list[{url, title, game, snippet}]` (deduped by url). CLI prints
  title + url. Frontend chips expand/hover to show the snippet.
- **Multi-turn:** stateless — the client sends `history: [{question, answer}, …]`; the prompt includes
  the short prior context; retrieval query = previous user turn + current question (pronoun
  disambiguation) — no extra LLM call, stays grounded. "New chat" clears client history.
- **Execution:** INLINE phases (per the project's established pattern; see DECISIONS 2026-06-21
  exec), TDD + verification each phase. No git (not a repo here); reversibility via how-to-undo logs
  and small diffs.

## Out of scope (→ BACKLOG)
- Answer caching / persistence; server-side session storage; auth; rate-limiting; re-embed or
  re-tag of the corpus (this cycle is app-layer only).
