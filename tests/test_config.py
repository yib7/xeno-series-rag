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


def test_load_env_strips_matching_wrapping_quotes(tmp_path, monkeypatch):
    # The common `.env` style KEY="value" / KEY='value' must yield the bare payload, not a
    # quote-wrapped string that breaks the API key downstream (P2-7).
    env = tmp_path / ".env"
    env.write_text('DQ_KEY="abc123"\nSQ_KEY=\'xyz789\'\nEMPTYQ_KEY=""\n', encoding="utf-8")
    for k in ("DQ_KEY", "SQ_KEY", "EMPTYQ_KEY"):
        monkeypatch.delenv(k, raising=False)
    load_env(str(env))
    assert os.environ["DQ_KEY"] == "abc123"
    assert os.environ["SQ_KEY"] == "xyz789"
    assert os.environ["EMPTYQ_KEY"] == ""


def test_load_env_keeps_partial_or_interior_quotes(tmp_path, monkeypatch):
    # Only a quote that wraps the WHOLE value is stripped; mismatched / one-sided / interior
    # quotes are part of the value and must survive verbatim.
    env = tmp_path / ".env"
    env.write_text(
        "LEAD_KEY=\"abc\nTRAIL_KEY=abc\"\nMIX_KEY=\"abc'\nINNER_KEY=ab\"cd\nBARE_QUOTE_KEY=\"\n",
        encoding="utf-8",
    )
    for k in ("LEAD_KEY", "TRAIL_KEY", "MIX_KEY", "INNER_KEY", "BARE_QUOTE_KEY"):
        monkeypatch.delenv(k, raising=False)
    load_env(str(env))
    assert os.environ["LEAD_KEY"] == '"abc'
    assert os.environ["TRAIL_KEY"] == 'abc"'
    assert os.environ["MIX_KEY"] == "\"abc'"
    assert os.environ["INNER_KEY"] == 'ab"cd'
    assert os.environ["BARE_QUOTE_KEY"] == '"'   # a lone quote is not a wrapped value


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "preset")
    load_env(str(env))
    assert os.environ["FOO_KEY"] == "preset"


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(str(tmp_path / "nope.env"))  # must not raise
