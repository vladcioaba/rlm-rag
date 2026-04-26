"""Tests for the TOML config loader."""

from __future__ import annotations

from rlm_rag.config import load_config


def _write(tmp_path, body: str):
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


def test_loads_all_sections(tmp_path):
    p = _write(tmp_path, '''
[tiers]
fast = "haiku"
balanced = "sonnet"
smart = "opus"

[thinking]
fast = 0
balanced = 0
smart = 8000

[review]
enabled = true
reviewer_model = "opus"
rounds = 2

[retrieval]
mode = "bm25"
rerank = true
rerank_model = "mxbai-rerank-base"

[budget]
concurrency = 5
''')
    cfg = load_config(p)
    assert cfg.models.fast_model == "haiku"
    assert cfg.models.smart_thinking_budget == 8000
    assert cfg.review.enabled is True
    assert cfg.review.rounds == 2
    assert cfg.retrieval.mode == "bm25"
    assert cfg.retrieval.rerank is True
    assert cfg.retrieval.rerank_model == "mxbai-rerank-base"
    assert cfg.budget.concurrency == 5


def test_partial_config_uses_defaults_for_missing_sections(tmp_path):
    p = _write(tmp_path, '''
[review]
enabled = true
''')
    cfg = load_config(p)
    assert cfg.review.enabled is True
    # Other sections default to env-or-package-defaults; just check they exist.
    assert cfg.models.fast_model
    assert cfg.retrieval.mode == "hybrid"


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.review.enabled is False
    assert cfg.retrieval.mode == "hybrid"


def test_garbage_section_ignored_not_crashed(tmp_path):
    p = _write(tmp_path, '''
[tiers]
fast = "haiku"

[unknown_section]
whatever = 42
''')
    cfg = load_config(p)
    assert cfg.models.fast_model == "haiku"  # known section still works
