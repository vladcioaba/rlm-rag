"""Tests for the OpenAI-compatible provider and the env-based factory."""

from __future__ import annotations

import json

import pytest

from rlm_rag.providers import (
    AnthropicProvider, OpenAIProvider, ProviderError,
    get_provider, reset_provider,
)


@pytest.fixture(autouse=True)
def _reset_provider():
    reset_provider()
    yield
    reset_provider()


def _capture(monkeypatch) -> list:
    captured: list[tuple[str, dict, dict]] = []
    def fake_post(url, headers, payload, timeout=300.0):
        captured.append((url, headers, payload))
        return {
            "choices": [{"message": {"content": "hello", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "prompt_tokens_details": {"cached_tokens": 4}},
        }
    monkeypatch.setattr("rlm_rag.providers._post", fake_post)
    return captured


# ---------- OpenAI provider ---------------------------------------------

def test_openai_payload_minimal(monkeypatch):
    captured = _capture(monkeypatch)
    p = OpenAIProvider(api_key="sk-test")
    r = p.chat(prompt="hi", system=None, prefix=None,
               model="gpt-4o-mini", max_tokens=64,
               thinking_budget=None, cache=True)
    url, headers, payload = captured[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-test"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert r.text == "hello"


def test_openai_system_prefix_combined_into_user_message(monkeypatch):
    captured = _capture(monkeypatch)
    OpenAIProvider(api_key="x").chat(
        prompt="q", system="sys", prefix="rubric",
        model="gpt-4o", max_tokens=64,
        thinking_budget=None, cache=True,
    )
    msgs = captured[0][2]["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    # prefix prepended to the user prompt with a separator
    assert msgs[1]["role"] == "user"
    assert "rubric" in msgs[1]["content"] and msgs[1]["content"].endswith("q")


def test_openai_thinking_budget_translates_to_reasoning_effort(monkeypatch):
    captured = _capture(monkeypatch)
    p = OpenAIProvider(api_key="x")

    p.chat(prompt="q", system=None, prefix=None, model="o1", max_tokens=64,
           thinking_budget=1500, cache=True)
    p.chat(prompt="q", system=None, prefix=None, model="o1", max_tokens=64,
           thinking_budget=5000, cache=True)
    p.chat(prompt="q", system=None, prefix=None, model="o1", max_tokens=64,
           thinking_budget=20000, cache=True)
    p.chat(prompt="q", system=None, prefix=None, model="o1", max_tokens=64,
           thinking_budget=None, cache=True)

    assert captured[0][2]["reasoning_effort"] == "low"
    assert captured[1][2]["reasoning_effort"] == "medium"
    assert captured[2][2]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in captured[3][2]


def test_openai_base_url_override(monkeypatch):
    captured = _capture(monkeypatch)
    p = OpenAIProvider(api_key="x", base_url="https://openrouter.ai/api/v1")
    p.chat(prompt="q", system=None, prefix=None, model="claude/sonnet",
           max_tokens=64, thinking_budget=None, cache=True)
    assert captured[0][0] == "https://openrouter.ai/api/v1/chat/completions"


def test_openai_extra_headers_forwarded(monkeypatch):
    captured = _capture(monkeypatch)
    p = OpenAIProvider(api_key="x", extra_headers={"X-Trace-Id": "abc"})
    p.chat(prompt="q", system=None, prefix=None, model="m",
           max_tokens=64, thinking_budget=None, cache=True)
    assert captured[0][1]["X-Trace-Id"] == "abc"


def test_openai_usage_extracted_with_cache(monkeypatch):
    _capture(monkeypatch)
    r = OpenAIProvider(api_key="x").chat(
        prompt="q", system=None, prefix=None, model="m",
        max_tokens=64, thinking_budget=None, cache=True,
    )
    assert r.tokens_in == 10
    assert r.tokens_out == 5
    assert r.cache_reads == 4
    assert r.cache_writes == 0


# ---------- Factory: env-based selection --------------------------------

def test_factory_picks_anthropic_when_only_anthropic_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = get_provider()
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"


def test_factory_picks_openai_when_only_openai_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    p = get_provider()
    assert isinstance(p, OpenAIProvider)


def test_factory_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    p = get_provider()
    assert isinstance(p, OpenAIProvider)


def test_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(ProviderError):
        get_provider()


def test_factory_no_keys_falls_back_to_anthropic_class(monkeypatch):
    """No keys + no PROVIDER → AnthropicProvider class is chosen, but its
    constructor raises because there's no key. We assert the right error,
    not silent fallback to OpenAI.
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        get_provider()


# ---------- Default models ---------------------------------------------

def test_anthropic_defaults_are_claude():
    d = AnthropicProvider(api_key="x").default_models()
    assert d["fast"].startswith("claude")
    assert d["balanced"].startswith("claude")
    assert d["smart"].startswith("claude")


def test_openai_defaults_are_gpt():
    d = OpenAIProvider(api_key="x").default_models()
    assert d["fast"].startswith("gpt")
    assert d["balanced"].startswith("gpt")
    assert d["smart"].startswith("gpt")


def test_openai_defaults_overridable_via_env(monkeypatch):
    monkeypatch.setenv("RLM_RAG_OPENAI_FAST", "gpt-5-nano")
    monkeypatch.setenv("RLM_RAG_OPENAI_SMART", "o1")
    d = OpenAIProvider(api_key="x").default_models()
    assert d["fast"] == "gpt-5-nano"
    assert d["smart"] == "o1"
