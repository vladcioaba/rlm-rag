"""LLM provider abstraction.

Two backends ship by default:

  - AnthropicProvider — talks to api.anthropic.com directly. Native cache
    control and `thinking` blocks.
  - OpenAIProvider    — talks to any OpenAI-compatible chat completions
    endpoint. Works for OpenAI native, Azure OpenAI, OpenRouter, Ollama,
    LM Studio, GitHub Models, etc., by overriding `base_url`.

Auto-detected from the environment by `get_provider()`:
  LLM_PROVIDER  explicit selection: "anthropic" | "openai"
  ANTHROPIC_API_KEY if set and PROVIDER unset → anthropic
  OPENAI_API_KEY    if set and PROVIDER unset → openai
  LLM_API_KEY   generic fallback used by either provider
  LLM_BASE_URL  override for the OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    text: str
    tokens_in: int        # regular input + cache reads + cache writes
    tokens_out: int
    cache_reads: int
    cache_writes: int


class Provider(Protocol):
    name: str

    def chat(
        self,
        prompt: str,
        system: str | None,
        prefix: str | None,
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
        cache: bool,
    ) -> ChatResponse: ...

    def default_models(self) -> dict[str, str]:
        """Returns the default model id for each tier (fast/balanced/smart)."""
        ...


# ---------- shared HTTP helper ------------------------------------------

def _post(url: str, headers: dict, payload: dict, timeout: float = 300.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise ProviderError(f"HTTP {e.code} from {url}: {detail[:500]}") from None


# ---------- Anthropic ---------------------------------------------------

class AnthropicProvider:
    name = "anthropic"
    api_url = "https://api.anthropic.com/v1/messages"
    api_version = "2023-06-01"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") \
            or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "AnthropicProvider needs an API key. Set ANTHROPIC_API_KEY "
                "(or LLM_API_KEY)."
            )

    def default_models(self) -> dict[str, str]:
        return {
            "fast":     "claude-haiku-4-5",
            "balanced": "claude-sonnet-4-6",
            "smart":    "claude-opus-4-7",
        }

    def chat(self, prompt, system, prefix, model, max_tokens, thinking_budget, cache):
        effective_max = max_tokens
        if thinking_budget:
            effective_max = max(effective_max, thinking_budget + 1024)

        payload: dict = {"model": model, "max_tokens": effective_max}

        if thinking_budget:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_budget),
            }

        if system:
            if cache:
                payload["system"] = [{
                    "type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                payload["system"] = system

        if prefix:
            blocks = [
                {"type": "text", "text": prefix},
                {"type": "text", "text": prompt},
            ]
            if cache:
                blocks[0]["cache_control"] = {"type": "ephemeral"}
            payload["messages"] = [{"role": "user", "content": blocks}]
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]

        resp = _post(self.api_url, {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }, payload)

        text = "".join(
            p.get("text", "") for p in resp.get("content", [])
            if p.get("type") == "text"
        )
        usage = resp.get("usage", {})
        regular = usage.get("input_tokens", 0)
        reads = usage.get("cache_read_input_tokens", 0)
        writes = usage.get("cache_creation_input_tokens", 0)
        return ChatResponse(
            text=text,
            tokens_in=regular + reads + writes,
            tokens_out=usage.get("output_tokens", 0),
            cache_reads=reads,
            cache_writes=writes,
        )


# ---------- OpenAI-compatible -------------------------------------------

class OpenAIProvider:
    """Targets any OpenAI Chat Completions-compatible endpoint.

    Set base_url to switch providers:
      OpenAI native: https://api.openai.com/v1
      Azure OpenAI:  https://<resource>.openai.azure.com/openai/v1
      OpenRouter:    https://openrouter.ai/api/v1
      Ollama:        http://localhost:11434/v1
      LM Studio:     http://localhost:1234/v1
      GitHub Models: https://models.github.ai/inference

    `cache` is ignored — OpenAI does prompt caching automatically when the
    same prefix is sent repeatedly. `thinking_budget` is translated to
    `reasoning_effort` (low/medium/high) for the o-series; ignored for
    other models.
    """

    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 extra_headers: dict | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") \
            or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "OpenAIProvider needs an API key. Set OPENAI_API_KEY "
                "(or LLM_API_KEY)."
            )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.extra_headers = extra_headers or {}

    def default_models(self) -> dict[str, str]:
        return {
            "fast":     os.environ.get("RLM_RAG_OPENAI_FAST", "gpt-4o-mini"),
            "balanced": os.environ.get("RLM_RAG_OPENAI_BALANCED", "gpt-4o"),
            "smart":    os.environ.get("RLM_RAG_OPENAI_SMART", "gpt-4o"),
        }

    def chat(self, prompt, system, prefix, model, max_tokens, thinking_budget, cache):
        # Build messages.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if prefix:
            messages.append({"role": "user", "content": f"{prefix}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Translate Anthropic-style thinking_budget to OpenAI reasoning_effort.
        # Only meaningful for o-series / GPT-5; otherwise quietly ignored
        # by compliant servers.
        if thinking_budget:
            if thinking_budget <= 2000:
                payload["reasoning_effort"] = "low"
            elif thinking_budget <= 8000:
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = "high"

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        resp = _post(f"{self.base_url}/chat/completions", headers, payload)

        choices = resp.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        # OpenAI exposes cached_tokens under prompt_tokens_details on supported models.
        details = usage.get("prompt_tokens_details", {}) or {}
        cached = details.get("cached_tokens", 0)
        return ChatResponse(
            text=text,
            tokens_in=prompt_tokens,
            tokens_out=usage.get("completion_tokens", 0),
            cache_reads=cached,
            cache_writes=0,  # OpenAI doesn't bill cache writes separately.
        )


# ---------- Factory -----------------------------------------------------

_PROVIDER: Provider | None = None


def get_provider() -> Provider:
    """Returns the cached active provider, instantiating from env on first call."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = _create_from_env()
    return _PROVIDER


def reset_provider() -> None:
    """Forget the cached provider — used by tests that want to swap env."""
    global _PROVIDER
    _PROVIDER = None


def _create_from_env() -> Provider:
    name = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not name:
        # Auto-detect by which API key is set.
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "anthropic"  # default; will raise on first call without key

    if name == "anthropic":
        return AnthropicProvider()
    if name in ("openai", "openai-compat", "openai-compatible"):
        return OpenAIProvider()
    raise ProviderError(
        f"Unknown LLM_PROVIDER {name!r}. "
        f"Use 'anthropic' or 'openai' (also set OPENAI_BASE_URL for "
        f"Azure / OpenRouter / Ollama / LM Studio / GitHub Models)."
    )
