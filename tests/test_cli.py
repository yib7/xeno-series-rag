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
