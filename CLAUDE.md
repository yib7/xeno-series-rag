# CLAUDE.md — Xeno Series Wiki RAG Chatbot

Local-first RAG chatbot over the Xeno Series Wiki (xenoserieswiki.org). See `xeno-rag-plan.md`
for the full build plan.

**Autopilot runs:** the autonomy contract is `.autopilot/AUTONOMY.md` (short) — quote its two
hard-stops in every subagent brief and reference the file for the rest. The live plan + resume
point is `.autopilot/PLAN.md` (first unchecked box). Shipped history: `.autopilot/MILESTONES.md`.

## Environment (this machine)
- Windows 11, PowerShell primary shell (Bash tool also available).
- Python: 3.12 / 3.13 / 3.14 installed via `py` launcher. **Use 3.12 for the project venv** (ML wheels).
- GPU: AMD Radeon RX 6600 XT — **no CUDA**. Local embeddings run CPU-only (free) or via ONNX/DirectML.

## Key constraints
- Data source is the MediaWiki API, not an HTML scraper. Carry a descriptive User-Agent, `maxlag=5`,
  serial requests, throttled. A full ~36k-article live pull is gated behind an explicit command.
- Content is CC-BY-SA: surface a source link with every answer; attribution notice in README.
