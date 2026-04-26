"""Persistent configuration loaded from ~/.rlm-rag/config.toml.

Schema (all fields optional):

    [tiers]
    fast      = "claude-haiku-4-5"
    balanced  = "claude-sonnet-4-6"
    smart     = "claude-opus-4-7"

    [thinking]
    fast      = 0
    balanced  = 0
    smart     = 8000

    [review]
    enabled        = false
    reviewer_model = "claude-opus-4-7"
    rounds         = 1

    [retrieval]
    mode    = "hybrid"     # "cosine" | "bm25" | "hybrid"
    rerank  = false
    rerank_model = "ms-marco-mini"

    [budget]
    concurrency = 10

CLI flags override config-file values; config-file values override env-var
defaults; env-var defaults override the package defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # py311+
except ImportError:                                    # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import ModelConfig, default_config


CONFIG_PATH_ENV = "RLM_RAG_CONFIG"


@dataclass
class ReviewConfig:
    enabled: bool = False
    reviewer_model: str = ""
    rounds: int = 1


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    rerank: bool = False
    rerank_model: str = ""


@dataclass
class BudgetConfig:
    concurrency: int = 10


@dataclass
class FullConfig:
    models: ModelConfig = field(default_factory=default_config)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)


def config_path() -> Path:
    """Where the config file is read from. Override with $RLM_RAG_CONFIG."""
    p = os.environ.get(CONFIG_PATH_ENV)
    if p:
        return Path(p).expanduser()
    return Path.home() / ".rlm-rag" / "config.toml"


def load_config(path: Path | None = None) -> FullConfig:
    """Load config. Returns env-var defaults if no file exists."""
    cfg = FullConfig()
    p = path or config_path()
    if not p.exists():
        return cfg

    with open(p, "rb") as f:
        data = tomllib.load(f)

    tiers = data.get("tiers", {})
    if isinstance(tiers, dict):
        if "fast" in tiers:
            cfg.models.fast_model = str(tiers["fast"])
        if "balanced" in tiers:
            cfg.models.balanced_model = str(tiers["balanced"])
        if "smart" in tiers:
            cfg.models.smart_model = str(tiers["smart"])

    thinking = data.get("thinking", {})
    if isinstance(thinking, dict):
        if "fast" in thinking:
            cfg.models.fast_thinking_budget = int(thinking["fast"])
        if "balanced" in thinking:
            cfg.models.balanced_thinking_budget = int(thinking["balanced"])
        if "smart" in thinking:
            cfg.models.smart_thinking_budget = int(thinking["smart"])

    review = data.get("review", {})
    if isinstance(review, dict):
        cfg.review.enabled = bool(review.get("enabled", cfg.review.enabled))
        cfg.review.reviewer_model = str(review.get("reviewer_model", cfg.review.reviewer_model))
        cfg.review.rounds = int(review.get("rounds", cfg.review.rounds))

    retrieval = data.get("retrieval", {})
    if isinstance(retrieval, dict):
        cfg.retrieval.mode = str(retrieval.get("mode", cfg.retrieval.mode))
        cfg.retrieval.rerank = bool(retrieval.get("rerank", cfg.retrieval.rerank))
        cfg.retrieval.rerank_model = str(retrieval.get("rerank_model", cfg.retrieval.rerank_model))

    budget = data.get("budget", {})
    if isinstance(budget, dict):
        cfg.budget.concurrency = int(budget.get("concurrency", cfg.budget.concurrency))

    return cfg
