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
