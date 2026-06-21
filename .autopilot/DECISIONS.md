# Decisions — assumptions & reversible calls made on autopilot

Claude logs here during the run so it never stalls waiting on you. Reversible calls it already
made go under **Resolved** (with how to undo). The rare non-blocking question it wants you to
weigh in on goes under **Open** — you answer all of those in a single pass when you come back,
while it keeps running the sensible default meanwhile.

Format: `[date] <phase> — <decision/question> — <why> — <how to undo>`

## Open (need your answer)
-

## Resolved
- [2026-06-21] SP8/live — User provided a Gemini API key (new `AQ.` AI-Studio format) → stored in **gitignored `.env`** (never committed/printed); added `load_env()` so CLI/web auto-load it. Verified live: key valid, 55 models. **how to undo:** delete `.env`, rotate key
- [2026-06-21] SP8/live — `gemini_model` `gemini-1.5-flash` → **`gemini-2.5-flash`** (1.5 retired; 2.5-flash returns OK via `models.list`) — **how to undo:** set any current model from `models.list()`
- [2026-06-21] SP10 — Full embed runs on **CPU** (not DirectML) — `onnxruntime-directml` collides with the CPU `onnxruntime 1.27.0` that chromadb requires, and `torch-directml` needs an older torch than 2.12.1; installing either risks breaking the working stack. This is the designed DirectML→CPU fallback. Verified `auto`→cpu yields 768-dim vectors. **how to undo:** enable DirectML in a separate isolated env with matching onnxruntime/torch and set `embed_device: directml`
- [2026-06-21] SP10 — Harvest complete: **36,181** ns=0 non-redirect titles → `data/raw/titles.jsonl`
- [2026-06-21] SP2 — Use supported `google-genai` SDK for the Gemini adapter, not the now-deprecated `google-generativeai` (install warns support has ended) — build SP8 on a maintained package — **how to undo:** revert pyproject + adapter import to `google.generativeai`
- [2026-06-21] SP1 recon — Live API confirmed: `articles=36144, pages=132982` (matches plan's ~36k) — API path validated for the full pull
- [2026-06-21] setup — Init git repo + branch `autopilot/xeno-rag-cycle1` — project was not under version control — **how to undo:** `rm -rf .git`
- [2026-06-21] setup — Pin venv to Python 3.12 (3.13/3.14 also installed) — best ML wheel availability (torch/chromadb) — **how to undo:** recreate venv on another interpreter
- [2026-06-21] scope — Cycle 1 = full live scrape + full embed of all ~36k articles (user choice) — user authorized the heavy outward-facing pull explicitly — **how to undo:** delete `data/` and re-run on a sample
- [2026-06-21] scope — Generation LLM = Gemini (Vertex/GCP) as default adapter, built provider-agnostic, tested with a MOCK LLM — user has GCP credits; live calls need user creds+credits (hard-stop) so never invoked autonomously — **how to undo:** swap adapter in config
- [2026-06-21] scope — Embedding backend = ONNX+DirectML on the AMD Radeon with automatic CPU fallback (user chose fully-local, no GPU rental) — free, no accounts; fallback guarantees the embed always completes — **how to undo:** set embedder device to `cpu` in config
- [2026-06-21] scope — Interface = CLI + thin FastAPI/SSE web UI with game selector this cycle — user choice — **how to undo:** ship CLI only
- [2026-06-21] scope — Vector store = ChromaDB persistent, cosine space (plan default, no Docker dep) — simplest local file-based store — **how to undo:** swap to Qdrant
- [2026-06-21] scope — Package layout = top-level `xeno_rag/` installed editable (vs plan's bare `src/`) — clean imports/tests on Windows without sys.path hacks — **how to undo:** move modules under `src/`
- [2026-06-21] exec — Run phases INLINE (not per-phase subagents) — sequential pipeline with shared venv/package/conventions; env flags spawning as the expensive path; autopilot blesses inline as correct — keeping TDD + code-review + verified checkpoints throughout — **how to undo:** dispatch subagents per remaining phase
