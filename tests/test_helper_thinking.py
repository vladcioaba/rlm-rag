"""Tests for thinking_budget plumbing in the AnthropicProvider's payload."""

from __future__ import annotations

import json
from unittest import mock

from rlm_rag.providers import AnthropicProvider


def _capture_payload(monkeypatch) -> list:
    """Patch providers._post and return a list that will collect each payload."""
    captured: list[dict] = []

    def fake_post(url, headers, payload, timeout=300.0):
        captured.append(payload)
        # Return a minimal valid Anthropic response.
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0},
        }

    monkeypatch.setattr("rlm_rag.providers._post", fake_post)
    return captured


def _provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test-key")


def test_thinking_budget_off_by_default(monkeypatch):
    captured = _capture_payload(monkeypatch)
    p = _provider()
    p.chat(prompt="hi", system=None, prefix=None,
           model="claude-x", max_tokens=1024,
           thinking_budget=None, cache=True)
    assert "thinking" not in captured[0]


def test_thinking_budget_when_set_appears_in_payload(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="hi", system=None, prefix=None,
                     model="claude-x", max_tokens=1024,
                     thinking_budget=8000, cache=True)
    assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_thinking_budget_bumps_max_tokens_to_fit(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="hi", system=None, prefix=None,
                     model="claude-x", max_tokens=512,
                     thinking_budget=4000, cache=True)
    assert captured[0]["max_tokens"] >= 4000 + 1024


def test_thinking_budget_does_not_shrink_user_max_tokens(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="hi", system=None, prefix=None,
                     model="claude-x", max_tokens=20000,
                     thinking_budget=4000, cache=True)
    assert captured[0]["max_tokens"] == 20000


def test_anthropic_caches_system_when_cache_true(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="hi", system="sys", prefix=None,
                     model="claude-x", max_tokens=1024,
                     thinking_budget=None, cache=True)
    payload = captured[0]
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_does_not_cache_when_cache_false(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="hi", system="sys", prefix=None,
                     model="claude-x", max_tokens=1024,
                     thinking_budget=None, cache=False)
    assert captured[0]["system"] == "sys"


def test_anthropic_prefix_first_block_only_cached(monkeypatch):
    captured = _capture_payload(monkeypatch)
    _provider().chat(prompt="q", system=None, prefix="pre",
                     model="claude-x", max_tokens=1024,
                     thinking_budget=None, cache=True)
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
