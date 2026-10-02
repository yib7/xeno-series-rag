"""Load the project YAML config and optional .env secrets."""

import os
from pathlib import Path

import yaml

from .errors import SetupError

# The repo/package root (parent of the xeno_rag package): fallback anchor for the config and .env
# lookups when the server/CLI is started from another directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(SetupError, ValueError):
    """config.yaml exists but cannot be used (not valid YAML, or not a mapping of settings)."""


class ConfigNotFoundError(ConfigError, FileNotFoundError):
    """config.yaml is in neither the current directory nor the repo root."""


def load_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    A relative ``path`` is tried against the CWD first, then against the repo root (the same lookup
    ``load_config`` uses for config.yaml), so starting the server from another directory still finds
    the keys. Does not override variables already set in the environment. Missing file is a no-op.
    Secrets live here (gitignored); they are never written to the YAML config or committed.
    """
    p = Path(path)
    if not p.is_file() and not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read {p} as UTF-8 text ({type(exc).__name__}); fix or delete it.") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Strip one pair of matching wrapping quotes (the common KEY="value" / KEY='value' .env
        # style), only when the quote wraps the WHOLE value; interior or mismatched quotes are
        # kept verbatim.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config file into a dict. Also loads .env secrets if present.

    A relative ``path`` is tried against the CWD first (existing workflows), then against the
    repo root, so ``uvicorn xeno_rag.web.app:app`` works from any directory. Raises
    ``ConfigNotFoundError`` (a FileNotFoundError) with a message that says what to fix if the config is absent
    from both, and ``ConfigError`` (a ValueError) if the file is not valid YAML or not a mapping.

    Note: relative ``paths.*`` VALUES inside the config remain CWD-relative by design: pipeline
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
        raise ConfigNotFoundError(
            f"Config not found: {path} (tried the current directory {Path.cwd()} and the repo "
            f"root {_REPO_ROOT}). Run from the repo root or pass an explicit config path."
        )
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config {p} is not valid YAML: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Config {p} is not valid UTF-8 text: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config {p} must be a YAML mapping of settings (key: value lines), "
            f"found {type(data).__name__ if data is not None else 'an empty file'}."
        )
    return data
