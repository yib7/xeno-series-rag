"""Tests for the CLI. Injects a fake answer function (no model/LLM)."""

import pytest

from xeno_rag.cli import main


def test_cli_prints_answer_and_sources(capsys):
    def fake(question, **kw):
        return {"answer": "Infinity Blade has 250 power.", "sources": ["https://w/Infinity_Blade"]}

    main(["--question", "power of infinity blade"], answer_fn=fake)
    out = capsys.readouterr().out
    assert "Infinity Blade has 250 power." in out
    assert "https://w/Infinity_Blade" in out


def test_cli_passes_game_filter(capsys):
    seen = {}

    def fake(question, **kw):
        seen.update(kw)
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--game", "XC3"], answer_fn=fake)
    assert seen.get("game_filter") == "XC3"


def test_cli_passes_tier_override(capsys):
    seen = {}

    def fake(question, **kw):
        seen.update(kw)
        return {"answer": "ok", "sources": [], "tier": "scholar"}

    main(["--question", "q", "--tier", "scholar"], answer_fn=fake)
    assert seen.get("tier") == "scholar"
    assert "scholar" in capsys.readouterr().err


def test_cli_off_topic_answer_prints_no_mode_line(capsys):
    """An off-topic result carries tier: None (the router.off_topic_gate short-circuit): the CLI's
    existing `if result.get("tier")` guard must treat that as falsy and print no `[... mode]` line."""
    def fake(question, **kw):
        return {"answer": "I can only help with the Xeno series.", "sources": [], "tier": None}

    main(["--question", "what's the weather"], answer_fn=fake)
    out = capsys.readouterr()
    assert "I can only help with the Xeno series." in out.out
    assert "mode]" not in out.err


def test_cli_auto_routes_by_default(capsys):
    seen = {}

    def fake(question, **kw):
        seen.update(kw)
        return {"answer": "ok", "sources": []}

    main(["--question", "q"], answer_fn=fake)
    assert seen.get("tier") is None


@pytest.mark.parametrize("argv", [["--question", "q", "--tier", "ultra"],
                                  ["--question", "q", "--model", "gemini-3.8-flash"]])
def test_cli_rejects_bad_tier_and_removed_model_flag(argv):
    with pytest.raises(SystemExit) as e:
        main(argv, answer_fn=lambda q, **kw: {"answer": "", "sources": []})
    assert e.value.code == 2


def test_cli_answer_fn_failure_prints_clean_message_not_traceback(capsys):
    """A failing answer_fn (e.g. GeminiClient's RuntimeError for a missing API key -- the most
    common first-run bad path) must reach the console as a short, actionable line, never as a raw
    Python traceback with internal file paths (checklist 4.9: errors shown to users leak nothing
    sensitive)."""
    def fake(question, **kw):
        raise RuntimeError("Gemini credentials not found. Set GOOGLE_API_KEY to enable live answers.")

    with pytest.raises(SystemExit) as exc_info:
        main(["--question", "q"], answer_fn=fake)
    assert "Gemini credentials not found" in str(exc_info.value)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "cli.py" not in err


def test_cli_survives_chars_the_console_codepage_cannot_encode(monkeypatch):
    # Redirected stdout on Windows is cp1252; a wiki title like "Alpha (∞)" used to crash the
    # print loop after the (paid) answer had already been generated.
    import io
    import sys

    buf = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buf, encoding="cp1252"))

    def fake(question, **kw):
        return {"answer": "Klaus → Zanza", "sources": [{"title": "Alpha (∞)", "url": "https://w/A"}]}

    main(["--question", "q"], answer_fn=fake)
    sys.stdout.flush()
    out = buf.getvalue().decode("cp1252")
    assert "Alpha (?)" in out
    assert "https://w/A" in out


def test_cli_setup_error_prints_just_the_actionable_message(capsys):
    from xeno_rag.errors import SetupError

    def fake(question, **kw):
        raise SetupError("Vector store not found at data/vectorstore. Run `python -m scripts.setup`.")

    with pytest.raises(SystemExit) as exc_info:
        main(["--question", "q"], answer_fn=fake)
    assert str(exc_info.value) == ("error: Vector store not found at data/vectorstore. "
                                   "Run `python -m scripts.setup`.")
    assert "Traceback" not in capsys.readouterr().err


def test_cli_unusable_config_is_a_clean_exit_not_a_traceback(monkeypatch):
    from xeno_rag import cli
    from xeno_rag.config import ConfigError

    def bad_config():
        raise ConfigError("Config config.yaml is not valid YAML: line 3")

    monkeypatch.setattr(cli, "load_config", bad_config)
    with pytest.raises(SystemExit) as exc_info:
        main(["--question", "q"], answer_fn=lambda q, **kw: pytest.fail("must not run"))
    assert "not valid YAML" in str(exc_info.value)


def test_cli_unexpected_failure_names_the_exception_type(capsys):
    def fake(question, **kw):
        raise ValueError("boom")

    with pytest.raises(SystemExit) as exc_info:
        main(["--question", "q"], answer_fn=fake)
    assert str(exc_info.value) == "error: ValueError: boom"


@pytest.mark.parametrize("bad_k", ["0", "-3", "many"])
def test_cli_rejects_a_non_positive_or_non_numeric_k(bad_k):
    with pytest.raises(SystemExit) as exc_info:
        main(["--question", "q", "--k", bad_k], answer_fn=lambda q, **kw: {"answer": "", "sources": []})
    assert exc_info.value.code == 2
