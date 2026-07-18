"""Tests for eval/analyze.py: import must be side-effect-free (P2-4).

`eval/` is not an importable package, so the module is loaded by file path via
``importlib.util.spec_from_file_location`` (mirrors tests/test_reindex.py's pattern).
"""

import importlib.util
import os

ANALYZE_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "analyze.py")


def _load_analyze_module():
    spec = importlib.util.spec_from_file_location("analyze", ANALYZE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_import_is_side_effect_free(tmp_path, monkeypatch):
    """Loading the module (from a working dir with no eval/results.json) must not raise. With the
    pre-fix module-level `Path("eval/results.json").read_text(...)`, import itself blows up with
    FileNotFoundError -- the module can't be imported or unit-tested at all."""
    monkeypatch.chdir(tmp_path)
    module = _load_analyze_module()  # must not raise
    assert module.title_hint("Foo (XC2)") == "XC2"  # pure helper stays importable/usable


def test_analyze_main_prints_friendly_message_when_results_missing(tmp_path, monkeypatch, capsys):
    """With no eval/results.json present, main() prints a clear pointer instead of a bare traceback."""
    monkeypatch.chdir(tmp_path)
    module = _load_analyze_module()

    module.main()

    out = capsys.readouterr().out
    assert "eval/results.json not found" in out
