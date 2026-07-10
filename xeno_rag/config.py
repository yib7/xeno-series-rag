"""Load the project YAML config and optional .env secrets."""

import os
from pathlib import Path

import yaml


def load_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    Does not override variables already set in the environment. Missing file is a no-op.
    Secrets live here (gitignored); they are never written to the YAML config or committed.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Strip one pair of matching wrapping quotes (the common KEY="value" / KEY='value' .env
        # style) — only when the quote wraps the WHOLE value; interior or mismatched quotes are
        # kept verbatim.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# The repo/package root (parent of the xeno_rag package) — fallback anchor for the config lookup
# when the server/CLI is started from another directory (P2-10).
_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config file into a dict. Also loads .env secrets if present.

    A relative ``path`` is tried against the CWD first (existing workflows), then against the
    repo root, so ``uvicorn xeno_rag.web.app:app`` works from any directory. Raises
    FileNotFoundError with an actionable message if the config is absent from both.

    Note: relative ``paths.*`` VALUES inside the config remain CWD-relative by design — pipeline
    and server runs happen from the repo root, and re-anchoring them would break workflows that
    deliberately point at a different data directory via CWD.
    """
    load_env()
    p = Path(path)
    if not p.is_file() and not p.is_absolute():
        fallback = _REPO_ROOT / p
        if fallback.is_file():
            p = fallback
    if not p.is_file():
        raise FileNotFoundError(
            f"Config not found: {path} (tried the current directory {Path.cwd()} and the repo "
            f"root {_REPO_ROOT}). Run from the repo root or pass an explicit config path."
        )
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
