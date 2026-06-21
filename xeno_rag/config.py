"""Load the project YAML config."""

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config file into a dict. Raises FileNotFoundError if absent."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
