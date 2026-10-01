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


def test_load_config_from_other_cwd_falls_back_to_repo_root(tmp_path, monkeypatch):
    # Starting the server/CLI outside the repo root must still find the repo config (P2-10).
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["base_url"].endswith("/api.php")


def test_load_config_missing_everywhere_raises_actionable_message(tmp_path, monkeypatch):
    import pytest

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="repo root"):
        load_config("no_such_config_xyz.yaml")


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


def test_load_config_malformed_yaml_is_an_actionable_config_error(tmp_path):
    import pytest

    from xeno_rag.config import ConfigError

    bad = tmp_path / "config.yaml"
    bad.write_text("paths: [unclosed\n  key: : :\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(str(bad))


def test_load_config_empty_or_non_mapping_file_is_a_config_error(tmp_path):
    import pytest

    from xeno_rag.config import ConfigError

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty file"):
        load_config(str(empty))
    as_list = tmp_path / "list.yaml"
    as_list.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(str(as_list))


def test_config_not_found_is_both_a_setup_error_and_a_file_not_found():
    import pytest

    from xeno_rag.errors import SetupError

    with pytest.raises(SetupError) as info:
        load_config("does/not/exist.yaml")
    assert isinstance(info.value, FileNotFoundError)


def test_load_env_falls_back_to_the_repo_root(tmp_path, monkeypatch):
    """Starting from another directory must still find the repo's .env, like config.yaml does."""
    from xeno_rag import config as config_mod

    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("ROOT_ONLY_KEY=from-root\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(config_mod, "_REPO_ROOT", tmp_path / "repo")
    monkeypatch.delenv("ROOT_ONLY_KEY", raising=False)
    load_env()
    assert os.environ["ROOT_ONLY_KEY"] == "from-root"


def test_load_env_unreadable_file_is_a_config_error(tmp_path):
    import pytest

    from xeno_rag.config import ConfigError

    env = tmp_path / ".env"
    env.write_bytes(b"KEY=\xff\xfe\xfa\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_env(str(env))
