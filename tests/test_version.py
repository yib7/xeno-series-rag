"""Version is single-sourced from installed metadata; no drift vs pyproject (P2-1)."""
import tomllib
from importlib.metadata import version
from pathlib import Path

import xeno_rag

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_version_matches_installed_metadata() -> None:
    # Proves __version__ is derived from installed package metadata, not a literal.
    assert xeno_rag.__version__ == version("xeno-rag")


def test_version_matches_pyproject() -> None:
    # Proves no drift against pyproject [project].version -- the exact P2-1 finding.
    assert xeno_rag.__version__ == _pyproject_version()


def test_version_not_stale_scaffold() -> None:
    # Guards the stale scaffold value from regressing.
    assert xeno_rag.__version__ != "0.1.0"
