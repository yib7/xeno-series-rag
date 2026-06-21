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
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config file into a dict. Also loads .env secrets if present.

    Raises FileNotFoundError if the config file is absent.
    """
    load_env()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
