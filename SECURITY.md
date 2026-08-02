# Security

This is a local-first application. By default the web server binds to `127.0.0.1` (localhost) and is
meant to be run by a single user on their own machine. It holds one secret, a Gemini API key, read
from a gitignored `.env` and sent only to Google's Gemini API.

## What the code does to stay safe

- **Model allowlist:** the web `/ask` endpoint accepts only a fixed set of Gemini model ids. An
  arbitrary model string from the client is ignored, so a caller can never steer requests to an
  unintended model or endpoint.
- **No SSRF surface:** outbound requests go only to the configured wiki API base URL and (for art)
  fixed Wikimedia hosts. No request target is user-controlled.
- **Sanitized lexical search:** free-text questions are tokenized and each token is quoted before it
  reaches SQLite FTS5, so a question can never form a malformed or injected MATCH expression. All SQL
  uses bound parameters.
- **Structured metadata filter:** the game filter builds a structured ChromaDB `where` clause, not a
  query string, so it cannot be used for injection.
- **Safe error responses:** failures in the answer stream are returned as a generic message. Stack
  traces, internal paths, and secrets are never sent to the client.
- **Rate limiting:** `/ask` fans out to the paid Gemini API and a CPU cross-encoder, so it is rate
  limited per client (a small in-process sliding window, configurable, default 30 requests/minute).
  This protects API credits and CPU if the server is ever exposed beyond localhost. It can be disabled
  for a trusted single-user deployment.
- **Proxy trust is opt-in.** Rate limiting keys on the direct peer address by default and ignores the
  client-supplied `X-Forwarded-For` header, since trusting it on a directly-exposed port would let a
  caller spoof a fresh bucket per request and defeat the limiter. Behind a reverse proxy that sets XFF
  itself and is the only path in, set the environment variable `XENO_TRUST_PROXY=1` so the first XFF
  hop is used as the key. Documented in `.env.example`.

## Dependency audit

`pip-audit` is run as part of the release checklist. Current status:

- **pip advisories** apply to the package installer in the development environment, not to the shipped
  application's runtime dependencies. The local toolchain is kept current.
- **setuptools (PYSEC-2026-3447) and torch (PYSEC-2025-194), resolved.** Both were fixed in-run by
  bumping to their patched releases: `setuptools 83.0.0` and `torch 2.13.0`. torch 2.13.0 requires only
  `setuptools >= 77.0.3` (no upper bound), so the fixed setuptools installs cleanly alongside the ML
  stack. The torch advisory is a `torch.jit.script` memory-corruption issue the app never exercised (torch
  is used only transitively, for embedding and reranking, never through `torch.jit.script`); it is patched
  regardless, so a fresh `pip-audit` on the committed lock reports no torch or setuptools finding.
- **pyasn1 (PYSEC-2026-3455 / 3456 / 3457), resolved.** Three algorithmic-complexity denial-of-service
  issues in the ASN.1 BER/CER/DER decoder (quadratic-time OID, tag-id, and REAL parsing), pulled in
  transitively via `pyasn1_modules`. Fixed in-run by bumping `pyasn1` to `0.6.4`, the release carrying
  the upstream fix; a fresh `pip-audit` on the committed lock reports no pyasn1 finding.
- **chromadb (CVE-2026-45829 / PYSEC-2026-311, "ChromaToast"):** a pre-authentication code-injection
  RCE in ChromaDB's optional **FastAPI server mode**, reachable only when running `chroma run`,
  exposing its HTTP API, and accepting a client-supplied embedding-function config that pulls remote
  code (`trust_remote_code`). It affects chromadb `<= 1.5.9` (fixed in 1.6.0). This project uses
  ChromaDB strictly as an **embedded in-process `PersistentClient`** over a local file: it never starts
  the server, never exposes the HTTP API, and never loads a client-supplied embedding-function
  configuration, so the vulnerable code path is not reachable. The version is pinned to 1.5.9 because
  the distributed prebuilt vector index (the GitHub release asset `scripts/setup.py` downloads) was
  built with it; moving to 1.6.0 means rebuilding and re-publishing that asset, tracked for a future
  data rebuild rather than done reactively for an unreachable path.

## Reporting a vulnerability

If you find a security issue, please open a GitHub issue describing it, or contact the maintainer
through the repository. Please do not include working exploit payloads in a public issue.
