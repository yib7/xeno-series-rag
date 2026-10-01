"""Lockfile and resume guards of scripts/run_fetch_durable.py (loaded by path: scripts/ is no package)."""

import importlib.util
import os

import pytest

PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_fetch_durable.py")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_fetch_durable", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOCK", str(tmp_path / "raw" / "fetch.lock"))
    return module


def test_first_instance_takes_the_lock_and_second_is_refused(mod, monkeypatch):
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)
    assert mod._locked() is False
    with open(mod.LOCK, encoding="utf-8") as f:
        assert f.read() == str(os.getpid())
    assert mod._locked() is True                     # a live holder: refuse, leave the lock alone


def test_stale_lock_from_a_dead_pid_is_taken_over(mod, monkeypatch):
    os.makedirs(os.path.dirname(mod.LOCK))
    with open(mod.LOCK, "w", encoding="utf-8") as f:
        f.write("999999")
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: False)
    assert mod._locked() is False
    with open(mod.LOCK, encoding="utf-8") as f:
        assert f.read() == str(os.getpid())


def test_garbage_lock_contents_count_as_stale(mod, monkeypatch):
    os.makedirs(os.path.dirname(mod.LOCK))
    with open(mod.LOCK, "w", encoding="utf-8") as f:
        f.write("not-a-pid")
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)
    assert mod._locked() is False


def test_missing_title_lists_exit_cleanly_and_release_the_lock(mod, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mod, "REPO", str(tmp_path))
    monkeypatch.setattr(mod, "MAIN_TITLES", str(tmp_path / "nope.jsonl"))
    monkeypatch.chdir(tmp_path)
    mod.main()
    assert "cannot read the title lists" in capsys.readouterr().out
    assert not os.path.exists(mod.LOCK)
