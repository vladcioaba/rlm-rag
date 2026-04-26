"""Tests for the model-tiering layer (no API calls)."""

from __future__ import annotations

import pytest

from rlm_rag.models import (
    DEFAULT_BALANCED, DEFAULT_FAST, DEFAULT_SMART,
    ModelConfig, ModelTier, default_config,
)


def test_default_tiers_resolve_to_known_models():
    cfg = default_config()
    assert cfg.model_for(ModelTier.FAST) == DEFAULT_FAST
    assert cfg.model_for(ModelTier.BALANCED) == DEFAULT_BALANCED
    assert cfg.model_for(ModelTier.SMART) == DEFAULT_SMART


def test_explicit_overrides_take_precedence():
    cfg = ModelConfig(
        fast_model="my-fast",
        balanced_model="my-balanced",
        smart_model="my-smart",
    )
    assert cfg.model_for(ModelTier.FAST) == "my-fast"
    assert cfg.model_for(ModelTier.BALANCED) == "my-balanced"
    assert cfg.model_for(ModelTier.SMART) == "my-smart"


def test_thinking_budget_zero_means_off():
    cfg = ModelConfig()  # all budgets default to 0
    assert cfg.thinking_for(ModelTier.FAST) is None
    assert cfg.thinking_for(ModelTier.BALANCED) is None
    assert cfg.thinking_for(ModelTier.SMART) is None


def test_thinking_budget_nonzero_returned():
    cfg = ModelConfig(smart_thinking_budget=8000)
    assert cfg.thinking_for(ModelTier.SMART) == 8000
    assert cfg.thinking_for(ModelTier.FAST) is None


def test_string_tier_accepted():
    cfg = ModelConfig(fast_model="x")
    assert cfg.model_for("fast") == "x"


def test_unknown_tier_raises():
    cfg = ModelConfig()
    with pytest.raises(ValueError):
        cfg.model_for("nonsense")


def test_env_var_overrides_picked_up_by_default_config(monkeypatch):
    monkeypatch.setenv("RLM_RAG_FAST_MODEL", "haiku-from-env")
    monkeypatch.setenv("RLM_RAG_SMART_THINKING_BUDGET", "12000")
    cfg = default_config()
    assert cfg.model_for(ModelTier.FAST) == "haiku-from-env"
    assert cfg.thinking_for(ModelTier.SMART) == 12000
