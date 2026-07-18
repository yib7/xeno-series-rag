"""Tests for scripts/setup.py's asset extraction (fail-closed zip-slip validation, P2-3).

`scripts/` is not an importable package, so the module is loaded by file path via
``importlib.util.spec_from_file_location`` (mirrors tests/test_reindex.py's pattern).
"""

import importlib.util
import os
import zipfile

import pytest

SETUP_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "setup.py")


def _load_setup_module():
    spec = importlib.util.spec_from_file_location("setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_rejects_zip_slip_member(tmp_path, monkeypatch):
    """A member whose path would resolve outside VS (e.g. "../evil.txt") must be rejected outright.

    CPython's zipfile.extractall already strips leading ".." components (verified empirically on
    3.12.10: the member lands sanitized *inside* the target, not escaping it), so a naive "did it
    escape VS" check on the post-extraction result would never trip. The real hardening is failing
    closed on the tampered member *before* extracting anything, so a `--skip-verify` run doesn't
    silently rewrite a crafted archive into something that merely looks safe.
    """
    setup = _load_setup_module()
    vs = tmp_path / "vs"
    vs.mkdir()
    monkeypatch.setattr(setup, "VS", str(vs))

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("good.txt", "benign")
        z.writestr("../evil.txt", "malicious")

    with pytest.raises(RuntimeError):
        setup._extract(str(zip_path))

    assert not (tmp_path / "evil.txt").exists()   # never escaped into VS's parent
    assert not (vs / "evil.txt").exists()          # nor was it silently sanitized into VS either


def test_extract_accepts_normal_members(tmp_path, monkeypatch):
    """A well-formed archive (top-level file + nested collection dir) still extracts cleanly."""
    setup = _load_setup_module()
    vs = tmp_path / "vs"
    vs.mkdir()
    monkeypatch.setattr(setup, "VS", str(vs))

    zip_path = tmp_path / "good.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("chroma.sqlite3", "sqlite-bytes")
        z.writestr("coll/data.bin", "collection-bytes")

    setup._extract(str(zip_path))  # must not raise

    assert (vs / "chroma.sqlite3").read_text(encoding="utf-8") == "sqlite-bytes"
    assert (vs / "coll" / "data.bin").read_text(encoding="utf-8") == "collection-bytes"
