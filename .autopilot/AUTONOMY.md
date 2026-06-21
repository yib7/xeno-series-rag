# Autopilot — Autonomy Contract

This run is on **autopilot**. The orchestrator and every subagent obey this. It's short on purpose:
quote the two hard-stops verbatim in each subagent brief and point here for the rest.

**Hard-stop — ask the human, then wait — ONLY for:**

1. **Secrets / credentials** — needing, creating, printing, committing, or rotating an API key,
   token, password, private key, or `.env` secret.
2. **Real money** — spending actual money / paid API credits, placing an order, incurring billable
   cloud cost.

That is the whole stop list. Nothing else pauses the run.

**Everything else: pick the sensible default, proceed, and log one line to `DECISIONS.md`** — this
includes reversible-but-scary and destructive-*local* ops (refactors, file moves/deletes, schema
changes, dependency installs, dropping a local/test DB, rewriting **un-pushed** git history,
resetting the working tree on this branch).

**Isolation is the safety net.** Work happens on a dedicated worktree/branch — never `main`, never a
live deployment. NOT part of the loop: merge to `main`, deploy, `push --force` to a shared branch, or
run against production. Those are the human's call at the single final merge gate.

**Project-specific etiquette (Xeno wiki):** the external MediaWiki API at xenoserieswiki.org is a
small fan wiki. Any code that hits it MUST carry the descriptive User-Agent, `maxlag=5`, serial
requests only, and the configured request delay. A *large* live pull (full ~36k-article scrape) is
an outward-facing action — gate it behind an explicit command/flag the human runs, never trigger it
implicitly from a test or default code path. Bounded sample fetches (a few pages, throttled) during
development are fine.

**Logging:** reversible decision → `DECISIONS.md` Resolved (`[date] <phase> — <decision> — <why> —
how to undo: <...>`); a rare non-blocking question → `DECISIONS.md` Open (keep running the default
meanwhile); a new unrelated idea → `BACKLOG.md` Inbox (next cycle, not this run).
