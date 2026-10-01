"""Tests for scripts/setup.py's asset extraction (fail-closed zip-slip validation, P2-3).

`scripts/` is not an importable package, so the module is loaded by file path via
``importlib.util.spec_from_file_location`` (mirrors tests/test_reindex.py's pattern).
"""

import importlib.util
import os
import sys
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


def _point_setup_at(setup, monkeypatch, tmp_path):
    vs = tmp_path / "vs"
    monkeypatch.setattr(setup, "VS", str(vs))
    monkeypatch.setattr(setup, "CHROMA", str(vs / "chroma.sqlite3"))
    monkeypatch.setattr(setup, "BM25", str(vs / "bm25.sqlite3"))
    return vs


def test_extract_bad_zip_leaves_existing_store_untouched(tmp_path, monkeypatch):
    """A corrupt archive used to wipe the working store before the zip was even opened."""
    setup = _load_setup_module()
    vs = _point_setup_at(setup, monkeypatch, tmp_path)
    vs.mkdir()
    (vs / "chroma.sqlite3").write_text("old-store", encoding="utf-8")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip archive")

    with pytest.raises(zipfile.BadZipFile):
        setup._extract(str(bad))

    assert (vs / "chroma.sqlite3").read_text(encoding="utf-8") == "old-store"
    assert not (tmp_path / "vs.partial").exists()


def test_extract_replaces_old_store_and_leaves_no_partial_dir(tmp_path, monkeypatch):
    setup = _load_setup_module()
    vs = _point_setup_at(setup, monkeypatch, tmp_path)
    vs.mkdir()
    (vs / "stale_collection").mkdir()
    zip_path = tmp_path / "good.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("chroma.sqlite3", "new-store")

    setup._extract(str(zip_path))

    assert (vs / "chroma.sqlite3").read_text(encoding="utf-8") == "new-store"
    assert not (vs / "stale_collection").exists()
    assert not (tmp_path / "vs.partial").exists()


def test_main_download_failure_is_a_clear_exit_not_a_traceback(tmp_path, monkeypatch, capsys):
    import urllib.error

    setup = _load_setup_module()
    _point_setup_at(setup, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["setup"])

    def offline(dest_dir):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(setup, "_download", offline)

    with pytest.raises(SystemExit) as exc:
        setup.main()

    message = str(exc.value)
    assert "download failed" in message and "no route to host" in message
    assert setup.ASSET in message and "releases/tag" in message


def test_main_resumes_with_only_the_bm25_step_when_a_prior_run_was_interrupted(tmp_path, monkeypatch):
    setup = _load_setup_module()
    vs = _point_setup_at(setup, monkeypatch, tmp_path)
    vs.mkdir()
    (vs / "chroma.sqlite3").write_text("vectors", encoding="utf-8")   # no bm25.sqlite3: interrupted run
    monkeypatch.setattr(sys, "argv", ["setup"])
    monkeypatch.setattr(setup, "_download", lambda d: pytest.fail("must not re-download the store"))
    rebuilt = []
    monkeypatch.setattr(setup, "_rebuild_bm25", lambda: rebuilt.append(True))

    setup.main()

    assert rebuilt == [True]


def test_main_is_a_noop_when_vectors_and_bm25_both_exist(tmp_path, monkeypatch, capsys):
    setup = _load_setup_module()
    vs = _point_setup_at(setup, monkeypatch, tmp_path)
    vs.mkdir()
    (vs / "chroma.sqlite3").write_text("vectors", encoding="utf-8")
    (vs / "bm25.sqlite3").write_text("bm25", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["setup"])
    monkeypatch.setattr(setup, "_download", lambda d: pytest.fail("must not download"))
    monkeypatch.setattr(setup, "_rebuild_bm25", lambda: pytest.fail("must not rebuild"))

    setup.main()

    assert "nothing to do" in capsys.readouterr().out


def test_main_bm25_failure_exits_with_a_retry_hint(tmp_path, monkeypatch):
    setup = _load_setup_module()
    vs = _point_setup_at(setup, monkeypatch, tmp_path)
    vs.mkdir()
    (vs / "chroma.sqlite3").write_text("vectors", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["setup"])

    def boom():
        raise RuntimeError("collection is empty")

    monkeypatch.setattr(setup, "_rebuild_bm25", boom)

    with pytest.raises(SystemExit) as exc:
        setup.main()

    assert "BM25 rebuild failed" in str(exc.value) and "re-run" in str(exc.value)
