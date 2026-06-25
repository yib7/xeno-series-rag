# PLAN — Xeno RAG, Cycle 2: Polish / Finalize

> On autopilot. Resume point = the first unchecked box below. **Not a git repo here** — no
> branch/worktree isolation; reversibility via how-to-undo logs in `.autopilot/DECISIONS.md` + small
> diffs. Autonomy contract: `.autopilot/AUTONOMY.md` (hard-stops: secrets, real money). New ideas →
> `.autopilot/BACKLOG.md`. Shipped history → `.autopilot/MILESTONES.md`.
> Design: `docs/superpowers/specs/2026-06-23-polish-finalize-design.md`.

## Scope (frozen)

Harden the shipped chatbot and add interactive features. **User chose: "Quality + bigger features" +
ALL extras.** Fix 5 inspected bugs (severe Markdown number bug, LLM-response failure, no question
validation, stale model, ungraceful errors); add the missing **frontend JS test harness** + backend
edge-case tests; real **token streaming**; **Markdown tables**; **prompt tuning**; richer **source
snippet previews**; conversation **thread** + **multi-turn follow-ups**; **example questions**;
**copy-answer**; **accessibility (aria-live)**. **OUT of scope:** corpus re-embed/re-tag, caching,
auth, server-side sessions (→ BACKLOG).

## Global constraints (every phase inherits)
- **Python 3.12** venv; run tools via `.venv\Scripts\python.exe`. Set `PYTHONIOENCODING=utf-8` on
  Windows to avoid cp1252 crashes.
- **No live LLM calls in tests** — MockLLM only. Live Gemini path reads creds from gitignored `.env`.
- **App-layer only** — do NOT re-embed or re-tag the corpus; the ChromaDB collection + BM25 index are
  untouched. The web server may hold `bm25.sqlite3` open; stop it before any rebuild (not needed here).
- TDD: failing test first, watch fail, minimal impl, watch pass. Keep the full suite green each phase.
- JS tests run via `node --test` (Node v24 present) and are wrapped by `tests/test_frontend_js.py`.
- No emojis in UI or output. Every answer keeps its CC-BY-SA source links.

---

## SP1 — Frontend test harness + Markdown renderer fix + tables

**Checkpoint:** `node --test tests/js/` green AND `.venv\Scripts\python.exe -m pytest tests/test_frontend_js.py -q`
green. A test reproduces the old "100 -> undefined" bug and now asserts "100" survives; a table test
renders `<table>`; existing inline/link/code behavior preserved. App still loads (`render.js` wired).

- [x] Extract `escapeHtml`/`inline`/`renderMarkdown` from `index.html` into `static/render.js`
  (UMD: browser global + `module.exports`). Wired `index.html` via `<script src="/static/render.js">`;
  normalized `index.html` to LF (was CRLF) so future Edits apply cleanly.
- [x] Wrote failing JS tests (`tests/js/render.test.mjs`, 9 tests) — 6 red (numbers→undefined, tables).
- [x] Fixed `inline()` stash restore to NUL-wrapped sentinel (`\x00<idx>\x00`). Added GFM table parsing
  to `renderMarkdown` + table CSS. **9/9 JS green.**
- [x] Added `tests/test_frontend_js.py` (shells out to `node --test`, skips if no Node).
- [x] Verified: `node --test` 9/9; full suite **136 passed** (135 + JS wrapper), live deselected.

## SP2 — Backend robustness (bugs #2/#3/#4) + edge-case tests

**Checkpoint:** `pytest tests/test_rag.py tests/test_web.py -q` green incl. new tests: empty/whitespace
question returns a friendly message without calling the LLM; a MockLLM that raises / returns None →
graceful fallback answer (no exception); no stale `gemini-1.5-flash` literal remains.

- [x] Tests (MockLLM): empty question → friendly no-question message, LLM not invoked; `_extract_text`
  handles `None`/raising `.text`/candidate parts; empty LLM response → graceful fallback; default model
  is not the retired `gemini-1.5-flash`. (7 new tests.)
- [x] Implemented: empty-question guard + empty-response fallback in `answer()`; `_extract_text(resp)`
  used by `GeminiClient.generate`; `DEFAULT_GEMINI_MODEL` constant replaces the `1.5-flash` literal.
- [x] Verified: `tests/test_rag.py tests/test_web.py` green; full suite **143 passed**.

## SP3 — Real token streaming + graceful streamed errors (bug #5)

**Checkpoint:** `pytest tests/test_rag.py tests/test_web.py -q` green: streaming adapter yields
multiple chunks (Mock), `answer_stream` yields tokens then a sources payload; `/ask` `TestClient`
receives >1 `data:` events progressively + a `sources` event; an LLM failure streams an `error` event
(no 500). `answer()` still returns the full string for the CLI.

- [x] Tests: `MockLLM.generate_stream` yields slices; `answer_stream` yields `("text",…)` then
  `("sources",…)`; empty Q short-circuits; forced failure → `("error",…)`; empty model response →
  fallback; `/ask` maps events to SSE (text/sources/error). (7 new tests.)
- [x] Implemented `generate_stream` on Mock + Gemini (`generate_content_stream`, safe per-chunk text);
  `answer_stream` (event tuples, try/except → error event); `web/app.py` consumes events + adapter so
  injected non-streaming `answer_fn` still works; frontend `ask()` handles `event: error` in-pane.
- [x] Verified: `tests/test_rag.py tests/test_web.py` green; full suite **150 passed**.

## SP4 — Prompt tuning (game-context + structure/tables nudge)

**Checkpoint:** `pytest tests/test_rag.py -q` green: prompt includes the active game scope when a
filter is set, and instructs concise Markdown + tables for multi-stat comparisons; ungrounded-invention
guard retained.

- [x] Tests: `build_prompt(…, game_filter="XC2")` adds a scope line (none when unfiltered); system
  prompt requests Markdown tables for multi-stat comparisons + retains grounding guard. (2 new tests.)
- [x] Implemented: extended `SYSTEM_PROMPT` (concise Markdown + tables nudge); `GAME_NAMES` map;
  scope line in `build_prompt(game_filter=…)`; threaded `game_filter` through `answer`/`answer_stream`.
- [x] Verified: `tests/test_rag.py` green; full suite **152 passed**.

## SP5 — Richer source payload + snippet previews

**Checkpoint:** `pytest tests/test_rag.py tests/test_cli.py tests/test_web.py -q` green: sources are
`list[{url,title,game,snippet}]` deduped by url; CLI prints `title — url`; `/ask` `sources` event
carries the dicts. JS test: a source chip renders the title and exposes the snippet.

- [x] Tests: `_dedupe_sources` returns deduped `{url,title,game,snippet}` dicts (breadcrumb stripped,
  truncated); existing source assertions updated to the dict shape; JS `sourcesHtml` renders
  title+game+snippet, tolerates legacy string urls, escapes injection. (1 PY + 4 JS new.)
- [x] Implemented: `_snippet` + rich `_dedupe_sources` in `rag.py`; `sourcesHtml`/`sourceName`/
  `escapeAttr` in `render.js`; CLI prints `title — url` (string-tolerant); `renderSources` uses
  `sourcesHtml`; chip restyled to a card with a 2-line snippet preview + game badge.
- [x] Verified: JS 13/13; full suite **153 passed**.

## SP6 — Frontend UX: conversation thread, aria-live, examples, copy, friendly errors

**Checkpoint:** `node --test tests/js/` + `pytest tests/test_frontend_js.py -q` green: each ask appends
a Q+answer **block** to a thread (not replacing); `renderAnswerBlock` builds Q, answer, sources, copy
button; example-question click fills the box; answer region has `aria-live="polite"`; error renders an
in-pane block. Manual load check: thread + examples + copy work.

- [x] JS tests: `answerBlockHtml({question,answerHtml,sources})` builds a turn (question, aria-live
  answer, copy button, sources) + escapes the question; `examplesHtml` builds clickable items. (6 new.)
- [x] Implemented: restructured `index.html` into `#examples` empty-state + `#thread`; per-ask turn
  blocks via `answerBlockHtml`; streaming appends into the current block's `.answer`; copy-answer
  (Clipboard API, event-delegated), clickable examples, `aria-live`, in-pane error; CSS `#answer`→
  `.answer`, new thread/turn/examples/copy styles.
- [x] Verified: JS 17/17; full suite **153 passed**; inline script `node --check` clean; **live browser
  smoke (preview_eval on port 8765): render.js loaded, 4 examples, "124 HP" renders (no "undefined"),
  Markdown table renders, turn block + copy + source card assemble, question escaped.**

## SP7 — Multi-turn follow-up context

**Checkpoint:** `pytest tests/test_rag.py tests/test_web.py -q` + `node --test tests/js/` green:
`answer_stream`/`answer` accept `history`; prompt includes prior turns; retrieval query expands with
the previous user turn; `/ask` accepts `history`; frontend sends accumulated history + has a "New chat"
reset. A follow-up test ("what is her element?" after a Pyra turn) shows history reached the prompt.

- [x] Tests: `build_prompt(…history=[…])` includes prior Q/A (none when empty); `_retrieval_query`
  prepends the previous question; `answer_stream` threads history into the prompt; `/ask` round-trips
  `history`. (3 PY rag + 1 PY web.)
- [x] Implemented: `history` through `answer`/`answer_stream`/`build_prompt` (+ `_history_block`) and
  retrieval expansion (`_retrieval_query`); `AskRequest.history`; frontend `history` array sent on ask
  + accumulated per turn, "New chat" button resets thread/history/empty-state.
- [x] Verified: full suite **157 passed**; inline script `node --check` clean; live (preview reload):
  New-chat button present+hidden, examples visible, builders loaded. Full interactive flow → SP8 mock smoke.

## SP8 — Final verification + write-up

**Checkpoint:** full suite `.venv\Scripts\python.exe -m pytest -q` green AND `node --test tests/js/`
green; manual smoke of the live UI (numbers render, stream flows, follow-up works, copy/examples/source
previews work). `docs/eval/2026-06-23-polish-finalize.md` written; MILESTONES appended; DECISIONS current.

- [x] Full suites: **157 Python passed** (1 live deselected) + **17 JS** (`node --test`).
- [x] Live smoke via `preview_eval` driving the real `ask()` with a stubbed `fetch` (no LLM cost):
  numbers render (no "undefined"), Markdown table renders, streaming appends across 2 turns,
  sources + copy button render, follow-up sends prior turn as `history`, "New chat" resets.
- [x] Wrote `docs/eval/2026-06-23-polish-finalize.md`.
- [x] Appended cycle 2 to MILESTONES + refreshed current-state snapshot. Teardown: no SDD scratch /
  `.done` files to sweep (inline run). No git ops (project operates git-free per standing constraint).

## Blocked (filled in during the run)

-
