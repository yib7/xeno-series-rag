# Decisions — assumptions & reversible calls made on autopilot

Claude logs here during the run so it never stalls waiting on you. Reversible calls it already
made go under **Resolved** (with how to undo). The rare non-blocking question it wants you to
weigh in on goes under **Open** — you answer all of those in a single pass when you come back,
while it keeps running the sensible default meanwhile.

Format: `[date] <phase> — <decision/question> — <why> — <how to undo>`

## Open (need your answer)
-

## Resolved
- [2026-06-21] setup — Init git repo + branch `autopilot/xeno-rag-cycle1` — project was not under version control — **how to undo:** `rm -rf .git`
- [2026-06-21] setup — Pin venv to Python 3.12 (3.13/3.14 also installed) — best ML wheel availability (torch/chromadb) — **how to undo:** recreate venv on another interpreter
