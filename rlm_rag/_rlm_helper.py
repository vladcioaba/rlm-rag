"""Sub-LLM call helper.

Delegates the actual chat call to a Provider (rlm_rag.providers).
Keeps the budget-tracking and FINAL/FINAL_VAR machinery used by both
the rlm-rag query layer and the standalone REPL driver in rlm-skill.

Call signatures intentionally stable so the rest of the package and
external callers don't need to change when providers are added.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from .providers import ProviderError, get_provider


DEFAULT_MODEL = os.environ.get("RLM_SUB_MODEL", "")  # "" → ask provider for default
DEFAULT_MAX_TOKENS = int(os.environ.get("RLM_SUB_MAX_TOKENS", "4096"))
DEFAULT_CONCURRENCY = int(os.environ.get("RLM_CONCURRENCY", "20"))
STDOUT_MAX = int(os.environ.get("RLM_STDOUT_MAX", "2048"))

BUDGET_PATH_ENV = "RLM_BUDGET_PATH"
DEFAULT_BUDGET_LIMIT = int(os.environ.get("RLM_BUDGET_LIMIT", "100"))


def _optional_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


DEFAULT_TOKEN_LIMIT = _optional_int("RLM_TOKEN_LIMIT")
DEFAULT_TOKEN_WARNING = _optional_int("RLM_TOKEN_WARNING")


class RLMError(RuntimeError):
    pass


class BudgetExceeded(RLMError):
    pass


# Re-export so existing callers can keep `except rlm_helper.ProviderError`.
__all__ = [
    "RLMError", "BudgetExceeded", "ProviderError",
    "llm_query", "llm_query_batch",
    "FINAL", "FINAL_VAR",
    "make_truncating_print",
    "BUDGET_PATH_ENV", "DEFAULT_BUDGET_LIMIT",
    "DEFAULT_TOKEN_LIMIT", "DEFAULT_TOKEN_WARNING",
]


# ---------- Budget tracking ---------------------------------------------

_budget_lock = threading.Lock()


def _budget_path() -> Path | None:
    p = os.environ.get(BUDGET_PATH_ENV)
    return Path(p) if p else None


def _read_budget() -> dict:
    p = _budget_path()
    if p and p.exists():
        try:
            state = json.loads(p.read_text())
            state.setdefault("calls", 0)
            state.setdefault("limit", DEFAULT_BUDGET_LIMIT)
            state.setdefault("tokens_in", 0)
            state.setdefault("tokens_out", 0)
            state.setdefault("cache_reads", 0)
            state.setdefault("cache_writes", 0)
            state.setdefault("token_limit", DEFAULT_TOKEN_LIMIT)
            state.setdefault("token_warning", DEFAULT_TOKEN_WARNING)
            state.setdefault("warned_tokens", False)
            return state
        except Exception:
            pass
    return {
        "calls": 0,
        "limit": DEFAULT_BUDGET_LIMIT,
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_reads": 0,
        "cache_writes": 0,
        "token_limit": DEFAULT_TOKEN_LIMIT,
        "token_warning": DEFAULT_TOKEN_WARNING,
        "warned_tokens": False,
    }


def _write_budget(state: dict) -> None:
    p = _budget_path()
    if p:
        p.write_text(json.dumps(state))


def _reserve(n: int) -> dict:
    with _budget_lock:
        state = _read_budget()
        new = state["calls"] + n
        if new > state["limit"]:
            raise BudgetExceeded(
                f"Sub-call budget exhausted: would be {new} / {state['limit']}. "
                f"Raise via `rlm_repl.py budget --session <id> --set N` "
                f"to extend. Or reconsider your chunking strategy — this may be a runaway loop."
            )
        tl = state.get("token_limit")
        if tl is not None:
            total = state.get("tokens_in", 0) + state.get("tokens_out", 0)
            if total >= tl:
                raise BudgetExceeded(
                    f"Token budget exhausted: {total} tokens used >= limit {tl}. "
                    f"Raise via `rlm_repl.py budget --session <id> --set-token-limit N` "
                    f"or accept and call FINAL with partial results."
                )
        state["calls"] = new
        _write_budget(state)
        return state


def _record_tokens(tokens_in: int, tokens_out: int,
                   cache_reads: int = 0, cache_writes: int = 0) -> None:
    with _budget_lock:
        state = _read_budget()
        state["tokens_in"] = state.get("tokens_in", 0) + tokens_in
        state["tokens_out"] = state.get("tokens_out", 0) + tokens_out
        state["cache_reads"] = state.get("cache_reads", 0) + cache_reads
        state["cache_writes"] = state.get("cache_writes", 0) + cache_writes

        tw = state.get("token_warning")
        if tw is not None and not state.get("warned_tokens"):
            total = state["tokens_in"] + state["tokens_out"]
            if total >= tw:
                state["warned_tokens"] = True
                msg = (
                    f"[rlm] token warning: {total} tokens used, crossed threshold {tw}. "
                    f"Limit is {state.get('token_limit') or 'unset'}."
                )
                try:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        _write_budget(state)


# ---------- Public API --------------------------------------------------

def _resolve_model(provider, model: str | None) -> str:
    """Resolve the model id: explicit > RLM_SUB_MODEL env > provider default (balanced)."""
    if model:
        return model
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    return provider.default_models()["balanced"]


def llm_query(
    prompt: str,
    system: str | None = None,
    prefix: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    thinking_budget: int | None = None,
) -> str:
    """One sub-LLM call. Returns text. See module docstring for contract."""
    _reserve(1)
    p = get_provider()
    resp = p.chat(
        prompt=prompt,
        system=system,
        prefix=prefix,
        model=_resolve_model(p, model),
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        thinking_budget=thinking_budget,
        cache=cache,
    )
    _record_tokens(resp.tokens_in, resp.tokens_out, resp.cache_reads, resp.cache_writes)
    return resp.text


def llm_query_batch(
    prompts: Iterable[str],
    system: str | None = None,
    prefix: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    concurrency: int | None = None,
    cache: bool = True,
    thinking_budget: int | None = None,
) -> list[str]:
    """Parallel sub-LLM calls. All share the same system+prefix so the cache
    (when supported by the provider) gets reused across the batch.
    """
    prompts = list(prompts)
    if not prompts:
        return []
    _reserve(len(prompts))

    cap = concurrency or DEFAULT_CONCURRENCY
    p = get_provider()
    resolved_model = _resolve_model(p, model)
    effective_max = max_tokens or DEFAULT_MAX_TOKENS

    def _one(text: str) -> str:
        resp = p.chat(
            prompt=text,
            system=system,
            prefix=prefix,
            model=resolved_model,
            max_tokens=effective_max,
            thinking_budget=thinking_budget,
            cache=cache,
        )
        _record_tokens(resp.tokens_in, resp.tokens_out, resp.cache_reads, resp.cache_writes)
        return resp.text

    with ThreadPoolExecutor(max_workers=cap) as pool:
        return list(pool.map(_one, prompts))


# ---------- FINAL / FINAL_VAR sentinel machinery ------------------------

_FINAL_KEY = "__rlm_final__"


def FINAL(answer) -> None:
    g = _caller_globals()
    g[_FINAL_KEY] = {"kind": "direct", "value": str(answer)}


def FINAL_VAR(var_name: str) -> None:
    g = _caller_globals()
    if var_name not in g:
        raise RLMError(f"FINAL_VAR: variable '{var_name}' does not exist in REPL")
    g[_FINAL_KEY] = {"kind": "var", "name": var_name}


def _caller_globals() -> dict:
    frame = sys._getframe(2)
    return frame.f_globals


# ---------- Truncating print --------------------------------------------

def make_truncating_print(buffer, cap: int):
    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        text = sep.join(str(a) for a in args) + end
        buffer.write(text)
    return _print
