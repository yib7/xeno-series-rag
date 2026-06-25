# Polish / Finalize Cycle — Report (2026-06-23)

Cycle 2 of the Xeno-RAG project: a finalization pass that fixed inspected UI/RAG/prompt bugs, added
the missing frontend test coverage, and layered interactive features onto the working app. Inline
execution, TDD throughout. **Final state: 157 Python tests + 17 JS tests green; full interactive UI
verified end-to-end in a real browser with no billable LLM call.**

## Bugs fixed

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | **Severe** | The Markdown renderer replaced **every number in every answer** with the literal "undefined" (`s.replace(/(\d+)/g, …)` matched prose digits, not just placeholder tokens). `"deals 100 damage"` → `"deals undefined damage"`. | NUL-wrapped stash sentinel (`\x00<idx>\x00`); extracted the renderer to `render.js` with a Node test harness. |
| 2 | Med | `GeminiClient.generate` returned `resp.text`, which raises / is None on a safety block or a thinking-model thought-only response → unhandled 500. | `_extract_text(resp)` (text → candidate parts → ""), never raises; `answer()` substitutes a friendly fallback. |
| 3 | Med | `/ask` accepted empty/whitespace questions (only the browser guarded). | `answer()`/`answer_stream()` short-circuit with a friendly message before any retrieval/LLM call. |
| 4 | Low | Hardcoded retired `gemini-1.5-flash` as the default model. | `DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"`. |
| 5 | Low | Errors surfaced as bare HTTP 500s with no in-pane message. | Streamed `error` SSE event rendered in-pane per turn. |

**Why bug #1 survived to this cycle:** the evaluation harness drives `rag.answer` (Python) directly and
never the browser renderer — the frontend JS had **zero automated tests**. That gap is now closed
(`tests/js/render.test.mjs` via `node --test`, wrapped by `tests/test_frontend_js.py` so it runs in the
`pytest` gate). This is the single most important durable outcome of the cycle.

## Quality + features shipped

- **Real token streaming** — `generate_stream`/`answer_stream` emit `(text|sources|error)` events;
  `/ask` streams tokens as the model produces them (was: generate-fully-then-fake-slice). Lower
  time-to-first-token; errors stream gracefully.
- **Markdown tables** — GFM table parsing in `render.js` + a prompt nudge to tabulate multi-stat
  comparisons. Stat answers now render as real `<table>`s instead of raw `|` pipes.
- **Prompt tuning** — concise-Markdown instruction + a game-scope line when a filter is active (helps
  resolve ambiguous names within the selected game).
- **Richer sources** — `{url, title, game, snippet}` payload; chips became cards with a game badge and a
  2-line snippet preview.
- **Conversation thread + multi-turn follow-ups** — per-ask turn blocks; stateless `history` sent with
  each ask; the prompt includes prior turns and retrieval expands with the previous question so
  pronoun follow-ups resolve. "New chat" resets.
- **Example questions** (clickable empty state), **copy-answer** button per turn, **aria-live** on the
  streamed answer.

## Verification

- **Python:** 157 passed (was 135 at cycle start; +22 across SP1–SP7), 1 live test deselected.
- **JS:** 17 `node --test` cases for the renderer, sources, turn blocks, and examples.
- **Inline app script:** `node --check` clean.
- **Live browser smoke** (preview server; screenshots time out in this env, so verified via
  `preview_eval` driving the real `ask()` with a stubbed `fetch` — no LLM cost): a numeric answer
  renders digits (not "undefined"), a Markdown table renders, streaming appends across 2 turns,
  sources + copy button render, a follow-up sends the prior turn as `history`, and "New chat" clears
  the thread/history and restores the empty state.

## Notes / follow-ups

- The full *live* end-to-end with Gemini was not run autonomously: live API spend is an autonomy
  hard-stop and was not re-authorized this cycle. The UI/SSE plumbing was instead verified with a
  stubbed stream; the Gemini adapter changes (`_extract_text`, `generate_stream`) are unit-tested
  against fakes.
- A stale dev server from a prior session still holds port 8000 with the *old* Python code; the new
  streaming features require restarting the server on the current code (`.claude/launch.json` now
  targets 8765 for the preview).
- App-layer only — the ChromaDB collection and BM25 index were untouched (no re-embed/re-tag).
