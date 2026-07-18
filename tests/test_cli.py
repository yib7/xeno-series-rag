"""Tests for the CLI. Injects a fake answer function (no model/LLM)."""

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
    """A model with no answer_styles entry (typo or a not-yet-configured id) silently skips the
    retrieval-depth pairing and falls back to base depth; this must at least print an advisory
    warning to stderr instead of failing silently or blocking the request."""
    def fake(question, **kw):
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--model", "bogus-model"], answer_fn=fake)
    err = capsys.readouterr().err
    assert "bogus-model" in err and "answer_styles" in err


def test_cli_no_warning_for_listed_model(capsys):
    """A model that IS a key in config.yaml's answer_styles must not trigger the advisory warning."""
    def fake(question, **kw):
        return {"answer": "ok", "sources": []}

    main(["--question", "q", "--model", "gemini-3.5-flash"], answer_fn=fake)
    err = capsys.readouterr().err
    assert err == ""
