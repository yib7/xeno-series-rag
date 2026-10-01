# Security

This is a local-first application. By default the web server binds to `127.0.0.1` (localhost) and is
meant to be run by a single user on their own machine. It holds two secrets: a Gemini API key, read
from a gitignored `.env` and sent only to Google's Gemini API; and an optional Jev router key,
`TYPESAFE_API_KEY`, read from the environment at call time and sent only to `api.typesafe.ai`. Neither
is ever logged.

## What the code does to stay safe

- **No client-chosen model:** the web `/ask` endpoint does not let a client pick a model at all. The
  answer tier comes from the router, and it is allowlisted on both ends: server-side, an unknown
  choice (`choice not in TIERS`) falls back to the configured fallback tier; client-side,
  `tierCaptionHtml` looks the tier up in a fixed label map and only ever puts that fixed label into
  `innerHTML`, never a raw string from the server.
- **No SSRF surface:** outbound requests go only to the configured wiki API base URL, fixed Wikimedia
  hosts (for art), and, when the Jev router is enabled, the fixed `api.typesafe.ai` URL from config.
  No request target is user-controlled. The routing call sends the question text, the previous
  question, and the game scope. When `router.answerability_check` is on, one or two more Jev calls
  (after rerank — a second one only if the first comes back `not_covered` and retrieval escalates to
  Scholar depth and checks again) additionally send up to `router.answerability_passages` (default 8)
  trimmed passages of already-public, CC BY-SA wiki text pulled from the retrieved chunks, alongside
  the question and, on a follow-up, the previous question — no other user data, and never the previous
  question's answer or any history beyond the one prior question. So a single `/ask` makes at most 3
  Jev calls (routing + two coverage checks): 0 with no key or `router.provider: fixed`; 1 for an
  off-topic question, a forced `--tier` (CLI and evals only) with a key, a failed routing call, or the check disabled;
  2 for a normal on-topic question (routing + one coverage check); 3 only when that check escalates.
  Set `router.provider: fixed` to disable all of them and keep routing fully offline.
- **Sanitized lexical search:** free-text questions are tokenized and each token is quoted before it
  reaches SQLite FTS5, so a question can never form a malformed or injected MATCH expression. All SQL
  uses bound parameters.
- **Structured metadata filter:** the game filter builds a structured ChromaDB `where` clause, not a
  query string, so it cannot be used for injection.
- **Safe error responses:** failures in the answer stream are returned as a generic message. Stack
  traces, internal paths, and secrets are never sent to the client.
- **Rate limiting:** `/ask` fans out to the paid Gemini API and a CPU cross-encoder, and, when the Jev
  router is enabled, a paid routing call too, so it is rate limited per client (a small in-process
  sliding window, configurable, default 30 requests/minute). This protects API credits and CPU if the
  server is ever exposed beyond localhost. It can be disabled for a trusted single-user deployment.
- **Proxy trust is opt-in.** Rate limiting keys on the direct peer address by default and ignores the
  client-supplied `X-Forwarded-For` header, since trusting it on a directly-exposed port would let a
  caller spoof a fresh bucket per request and defeat the limiter. Behind a reverse proxy that sets XFF
  itself and is the only path in, set the environment variable `XENO_TRUST_PROXY=1` so the first XFF
  hop is used as the key. Documented in `.env.example`.

## Dependency audit

`pip-audit` is run as part of the release checklist, over the committed lock (`requirements.txt`) and
over the resolved environment of a clean `pip install -e ".[dev]"`. Current status (2026-10-01):

- **Resolved by bumping.** The previous lock carried advisories in `aiohttp` (3.14.1, fixed in
  3.14.3), `anyio` (4.14.0, fixed in 4.14.2), `cryptography` (49.0.0, fixed in 50.0.0), `oauthlib`
  (3.3.1, fixed in 4.0.0), `soupsieve` (2.8.4, fixed in 2.9.0), `urllib3` (2.7.0, fixed in 2.8.0), and
  the installer itself (`pip`, fixed in 26.2). The lock now pins patched releases of all of them
  (`aiohttp 3.14.3`, `anyio 4.15.1`, `cryptography 50.0.2`, `oauthlib 4.0.0`, `soupsieve 2.10`,
  `urllib3 2.8.0`), and a fresh `pip-audit` reports none of them. The earlier setuptools, torch, and
  pyasn1 advisories stay resolved (`setuptools 84.0.0`, `torch 2.14.1`, `pyasn1 0.6.4`).
- **Legacy Google SDK removed from the lock.** `google-generativeai` and `google-api-python-client`
  (and their `google-ai-generativelanguage`, `httplib2`, `uritemplate`, `proto-plus`, `google-api-core`
  dependencies) were left over in the lock from before the move to `google-genai`. Nothing imports
  them and `pyproject.toml` never declared them, so they are gone from the environment and the lock.
- **chromadb (documented dead end, no fixed release):** `pip-audit` reports four advisories against
  `chromadb 1.5.9`, which is the newest release on PyPI, so no fixed upstream version exists:
  CVE-2026-45829 / PYSEC-2026-311 (pre-authentication code injection through `trust_remote_code` on the
  server's collections endpoint), CVE-2026-45833 / PYSEC-2026-3814 (the same injection for an
  authenticated caller with the update-collection permission), CVE-2026-45830 / PYSEC-2026-3813 (no
  tenant check on authenticated reads and writes), and CVE-2026-45831 / PYSEC-2026-3815 (the simple RBAC
  provider ignores which tenant, database, or collection a permission applies to). All four are in
  ChromaDB's **HTTP server mode** (`chroma run`, its authentication and RBAC providers). This project
  uses ChromaDB strictly as an **embedded in-process `PersistentClient`** over a local file: it never
  starts the server, never exposes the HTTP API, never uses `HttpClient` or an auth provider, and never
  loads a client-supplied embedding-function configuration, so none of the vulnerable code paths is
  reachable. The dependency is also pinned `<1.6` because the distributed prebuilt vector index (the
  GitHub release asset `scripts/setup.py` downloads) was built with 1.5.9. If upstream ships a fix, take
  it together with the next data rebuild, since a newer major index format may not read the shipped
  store.

## Reporting a vulnerability

If you find a security issue, please open a GitHub issue describing it, or contact the maintainer
through the repository. Please do not include working exploit payloads in a public issue.
