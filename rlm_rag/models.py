"""Model tiering: route per-call work to fast / balanced / smart models.

The pattern, borrowed from the RLM paper's spirit:

  fast      cheap throughput — relevance triage, per-chunk summaries
  balanced  default workhorse — iterative root, single-shot synthesis
  smart     for genuinely hard reasoning — adversarial review, deep impact

Defaults map to current Anthropic offerings; override per-tier via env vars
or the ModelConfig dataclass (the config-file layer feeds this in).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    SMART = "smart"


@dataclass
class ModelConfig:
    """Per-tier model identifiers + per-tier thinking budgets.

    All fields are strings/ints (never None) at construction; callers that
    need to express "no override" should compare against the defaults.
    """
    fast_model: str = ""
    balanced_model: str = ""
    smart_model: str = ""
    fast_thinking_budget: int = 0      # 0 = no thinking
    balanced_thinking_budget: int = 0
    smart_thinking_budget: int = 0

    def model_for(self, tier: ModelTier | str) -> str:
        if isinstance(tier, str):
            tier = ModelTier(tier)
        if tier == ModelTier.FAST:
            return self.fast_model or DEFAULT_FAST
        if tier == ModelTier.BALANCED:
            return self.balanced_model or DEFAULT_BALANCED
        if tier == ModelTier.SMART:
            return self.smart_model or DEFAULT_SMART
        raise ValueError(f"unknown tier {tier!r}")

    def thinking_for(self, tier: ModelTier | str) -> int | None:
        if isinstance(tier, str):
            tier = ModelTier(tier)
        budget = {
            ModelTier.FAST: self.fast_thinking_budget,
            ModelTier.BALANCED: self.balanced_thinking_budget,
            ModelTier.SMART: self.smart_thinking_budget,
        }[tier]
        return budget if budget > 0 else None


# Hardcoded fallbacks if no provider can be resolved (used in tests where
# we don't want to instantiate a provider). Real defaults come from the
# active provider's `default_models()` — claude-* for Anthropic, gpt-* for
# OpenAI-compatible.
DEFAULT_FAST = os.environ.get("RLM_RAG_FAST_MODEL", "claude-haiku-4-5")
DEFAULT_BALANCED = os.environ.get("RLM_RAG_BALANCED_MODEL", "claude-sonnet-4-6")
DEFAULT_SMART = os.environ.get("RLM_RAG_SMART_MODEL", "claude-opus-4-7")


def _provider_defaults() -> dict[str, str]:
    """Ask the active provider for its tier defaults. Returns the hardcoded
    Claude defaults if no provider is configured (no API key set, etc.).
    """
    try:
        from .providers import get_provider  # noqa: PLC0415
        return get_provider().default_models()
    except Exception:
        return {
            "fast": DEFAULT_FAST,
            "balanced": DEFAULT_BALANCED,
            "smart": DEFAULT_SMART,
        }


def default_config() -> ModelConfig:
    """Construct a ModelConfig from environment variables, falling back to
    the active provider's default models. The config-file layer can
    override this further.
    """
    def _int_env(name: str, default: int = 0) -> int:
        v = os.environ.get(name)
        try:
            return int(v) if v else default
        except ValueError:
            return default

    pd = _provider_defaults()
    return ModelConfig(
        fast_model=os.environ.get("RLM_RAG_FAST_MODEL", pd["fast"]),
        balanced_model=os.environ.get("RLM_RAG_BALANCED_MODEL", pd["balanced"]),
        smart_model=os.environ.get("RLM_RAG_SMART_MODEL", pd["smart"]),
        fast_thinking_budget=_int_env("RLM_RAG_FAST_THINKING_BUDGET"),
        balanced_thinking_budget=_int_env("RLM_RAG_BALANCED_THINKING_BUDGET"),
        smart_thinking_budget=_int_env("RLM_RAG_SMART_THINKING_BUDGET"),
    )
