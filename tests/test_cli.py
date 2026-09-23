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


def test_cli_model_override(capsys):
    captured = {}

    def fake(question, **kw):
        captured["cfg"] = kw.get("cfg")
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--model", "gemini-3.5-flash"], answer_fn=fake)
    assert captured["cfg"]["gemini_model"] == "gemini-3.5-flash"


def test_cli_warns_on_unlisted_model(capsys):
    """A model with no answer_tiers entry (typo or a not-yet-configured id) silently skips the
    retrieval-depth pairing and falls back to base depth; this must at least print an advisory
    warning to stderr instead of failing silently or blocking the request."""
    def fake(question, **kw):
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--model", "bogus-model"], answer_fn=fake)
    err = capsys.readouterr().err
    assert "bogus-model" in err and "answer_styles" in err


def test_cli_model_with_answer_styles_check(capsys):
    """When answer_styles is no longer in config (replaced by answer_tiers), any --model still
    triggers the advisory warning until cli.py is updated to use answer_tiers."""
    def fake(question, **kw):
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--model", "gemini-3.5-flash"], answer_fn=fake)
    err = capsys.readouterr().err
    assert "answer_styles" in err


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
