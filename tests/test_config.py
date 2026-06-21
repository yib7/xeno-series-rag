import os

from xeno_rag.config import load_config, load_env


def test_load_config_reads_expected_keys():
    cfg = load_config("config.yaml")
    assert cfg["base_url"].endswith("/api.php")
    assert cfg["batch_size"] == 50
    assert cfg["maxlag"] == 5
    assert isinstance(cfg["paths"], dict)
    assert "titles" in cfg["paths"]
    assert "vectorstore" in cfg["paths"]


def test_load_config_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_load_env_sets_environ(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=bar123\n# a comment\nEMPTY=\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    load_env(str(env))
    assert os.environ["FOO_KEY"] == "bar123"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "preset")
    load_env(str(env))
    assert os.environ["FOO_KEY"] == "preset"


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(str(tmp_path / "nope.env"))  # must not raise
