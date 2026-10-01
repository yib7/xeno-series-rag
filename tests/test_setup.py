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


def test_extract_rejects_cross_drive_member(tmp_path, monkeypatch):
    """A member that resolves onto a different drive than VS (e.g. "D:/evil.txt" when VS is under
    C:) makes os.path.commonpath raise ValueError - on Windows, paths on different drives share no
    common root. _extract must treat that as an ordinary outside-VS rejection (RuntimeError, fail
    closed), not let a bare ValueError leak out.

    The real trigger is a drive letter, which only exists on Windows - a "D:/evil.txt" member is
    just a benign relative subdir on POSIX, so a real cross-drive archive can't exercise this guard
    on the Linux CI runner. Instead we reproduce the exact condition the guard defends against:
    commonpath raising ValueError for the crafted member. Well-formed inside-VS members still
    resolve normally, so this stays a faithful test of the except-branch on every platform.
    """
    setup = _load_setup_module()
    vs = tmp_path / "vs"
    vs.mkdir()
    monkeypatch.setattr(setup, "VS", str(vs))

    real_commonpath = os.path.commonpath

    def commonpath_no_shared_drive(paths):
        # Emulate Windows' "paths on different drives have no common root" for the crafted member,
        # while leaving the benign inside-VS member to resolve through the real implementation.
        if any(p.endswith("evil.txt") for p in paths):
            raise ValueError("Paths don't have the same drive")
        return real_commonpath(paths)

    monkeypatch.setattr(os.path, "commonpath", commonpath_no_shared_drive)

    zip_path = tmp_path / "cross_drive.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("good.txt", "benign")
        z.writestr("evil.txt", "malicious")

    with pytest.raises(RuntimeError):
        setup._extract(str(zip_path))

    assert not (vs / "evil.txt").exists()


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


def test_download_falls_back_to_https_when_gh_fails(tmp_path, monkeypatch):
    """An installed but logged-out ``gh`` exits non-zero even for a public repo; setup must then fall
    back to the plain HTTPS download instead of crashing."""
    import subprocess

    setup = _load_setup_module()
    monkeypatch.setattr(setup.shutil, "which", lambda name: "gh")

    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(4, args[0])

    fetched = []
    monkeypatch.setattr(setup.subprocess, "run", failing_run)
    monkeypatch.setattr(setup.urllib.request, "urlretrieve", lambda url, out: fetched.append((url, out)))

    out = setup._download(str(tmp_path))

    assert out == str(tmp_path / setup.ASSET)
    assert fetched == [(f"https://github.com/{setup.REPO}/releases/download/{setup.TAG}/{setup.ASSET}", out)]


def test_download_uses_gh_when_it_succeeds(tmp_path, monkeypatch):
    setup = _load_setup_module()
    monkeypatch.setattr(setup.shutil, "which", lambda name: "gh")
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(setup.urllib.request, "urlretrieve",
                        lambda *a: pytest.fail("HTTPS must not be used when gh succeeds"))

    out = setup._download(str(tmp_path))

    assert calls and calls[0][:3] == ["gh", "release", "download"]
    assert out == str(tmp_path / setup.ASSET)
