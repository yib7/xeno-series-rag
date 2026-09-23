"""Tests for eval/run_gold_eval.py: --tier applies the tier's retrieval-depth override to cfg via
`router.apply_tier`, and an explicit --top-k still wins over it.

`eval/` is not an importable package, so the module is loaded by file path via
`importlib.util.spec_from_file_location` (mirrors tests/test_analyze.py's pattern).
"""

import importlib.util
import json
import os
import sys

RUN_GOLD_EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "run_gold_eval.py")


def _load_run_gold_eval_module():
    spec = importlib.util.spec_from_file_location("run_gold_eval", RUN_GOLD_EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gold(tmp_path):
    gold_path = tmp_path / "gold_questions.json"
    gold_path.write_text(json.dumps({"questions": [
        {"id": "q1", "game": "XC2", "category": "stat", "question": "test?",
         "answer": "a", "source_url": "https://x/a", "source_title": "A"},
    ]}), encoding="utf-8")
    return gold_path


def _stub_config():
    return {
        "top_k": 16,
        "embed_model": "stub",
        "answer_tiers": {
            "fast": {"model": "gemini-3.5-flash-lite", "top_k": 20, "max_chunks_per_page": 5},
        },
    }


def test_tier_flag_applies_apply_tier_before_top_k(tmp_path, monkeypatch):
    """--tier fast merges the fast tier's top_k/model into cfg via apply_tier, and the retrieval
    call sees the merged cfg (no explicit --top-k given, so the tier's value survives)."""
    module = _load_run_gold_eval_module()
    monkeypatch.setattr(module, "GOLD", _write_gold(tmp_path))
    monkeypatch.setattr(module, "load_config", _stub_config)
    monkeypatch.setattr(module, "Embedder", lambda cfg: object())

    seen_cfg = {}

    def fake_retrieve(question, cfg, game_filter=None, embedder=None):
        seen_cfg.update(cfg)
        return []

    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        sys, "argv",
        ["run_gold_eval.py", "--tier", "fast", "--out", str(tmp_path / "out.jsonl")],
    )

    module.main()

    assert seen_cfg["top_k"] == 20
    assert seen_cfg["gemini_model"] == "gemini-3.5-flash-lite"
    assert seen_cfg["answer_tier"] == "fast"


def test_explicit_top_k_overrides_tier(tmp_path, monkeypatch):
    """A user-supplied --top-k still wins over the tier's top_k, since apply_tier runs before the
    --top-k override in main()."""
    module = _load_run_gold_eval_module()
    monkeypatch.setattr(module, "GOLD", _write_gold(tmp_path))
    monkeypatch.setattr(module, "load_config", _stub_config)
    monkeypatch.setattr(module, "Embedder", lambda cfg: object())

    seen_cfg = {}

    def fake_retrieve(question, cfg, game_filter=None, embedder=None):
        seen_cfg.update(cfg)
        return []

    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        sys, "argv",
        ["run_gold_eval.py", "--tier", "fast", "--top-k", "5", "--out", str(tmp_path / "out.jsonl")],
    )

    module.main()

    assert seen_cfg["top_k"] == 5
