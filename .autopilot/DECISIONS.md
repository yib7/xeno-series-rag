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
- [2026-06-21] scope — Cycle 1 = full live scrape + full embed of all ~36k articles (user choice) — user authorized the heavy outward-facing pull explicitly — **how to undo:** delete `data/` and re-run on a sample
- [2026-06-21] scope — Generation LLM = Gemini (Vertex/GCP) as default adapter, built provider-agnostic, tested with a MOCK LLM — user has GCP credits; live calls need user creds+credits (hard-stop) so never invoked autonomously — **how to undo:** swap adapter in config
- [2026-06-21] scope — Embedding backend = ONNX+DirectML on the AMD Radeon with automatic CPU fallback (user chose fully-local, no GPU rental) — free, no accounts; fallback guarantees the embed always completes — **how to undo:** set embedder device to `cpu` in config
- [2026-06-21] scope — Interface = CLI + thin FastAPI/SSE web UI with game selector this cycle — user choice — **how to undo:** ship CLI only
- [2026-06-21] scope — Vector store = ChromaDB persistent, cosine space (plan default, no Docker dep) — simplest local file-based store — **how to undo:** swap to Qdrant
