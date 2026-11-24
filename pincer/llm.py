"""LLM backends shared by the controller (pincer/policies.py) and the FunSearch loop (pincer/funsearch/).

- `AnthropicBackend`: Claude through the official `anthropic` SDK. Default model `claude-opus-5-5`; effort is the
  cost/quality knob (`low` is enough for the controller's one-line JSON decisions, `medium`/`high` for writing
  programs). Server-side refusal fallback is on (`fallbacks="default"`), and a refusal is returned as an error.
  Credentials: whatever the SDK resolves (ANTHROPIC_API_KEY, or a profile from `ant auth login`).
- `LocalBackend`: a local OpenAI-style chat endpoint, e.g. the vLLM server of scripts/minrank/05_vllm_serve.sh
  (Qwen2.5-Coder-32B on an A6000), which is what the earlier OpenEvolve runs used. Plain HTTP, no extra package.
- `ScriptedBackend`: returns canned replies (tests, dry runs).

Every call returns (text, info) with latency and token counts so runs can report LLM cost honestly.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LLMError(RuntimeError):
    pass


class AnthropicBackend:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.usage = Usage()
        self.name = f"anthropic:{model}:{effort}"

    def complete(self, system: str, prompt: str) -> str:
        import anthropic

        t0 = time.perf_counter()
        self.usage.calls += 1
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            self.usage.errors += 1
            raise LLMError(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            self.usage.errors += 1
            raise LLMError(f"API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            self.usage.errors += 1
            raise LLMError(f"connection error: {e}") from e
        finally:
            self.usage.seconds += time.perf_counter() - t0
        if resp.usage is not None:
            self.usage.input_tokens += resp.usage.input_tokens or 0
            self.usage.output_tokens += resp.usage.output_tokens or 0
        if resp.stop_reason == "refusal":
            self.usage.errors += 1
            raise LLMError("model declined the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class LocalBackend:
    """OpenAI-style /v1/chat/completions endpoint (vLLM, Ollama)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
                 temperature: float = 0.8, max_tokens: int = 4096, timeout: float = 600) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout
        self.usage = Usage()
        self.name = f"local:{model}"

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        self.usage.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except Exception as e:  # network / server errors become LLMError for the caller's fallback path
            self.usage.errors += 1
            raise LLMError(f"local LLM call failed: {e}") from e
        finally:
            self.usage.seconds += time.perf_counter() - t0
        u = d.get("usage") or {}
        self.usage.input_tokens += int(u.get("prompt_tokens", 0))
        self.usage.output_tokens += int(u.get("completion_tokens", 0))
        return d["choices"][0]["message"]["content"]


@dataclass
class ScriptedBackend:
    replies: list[str]
    usage: Usage = field(default_factory=Usage)
    name: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> str:
        self.usage.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            raise LLMError("no scripted replies left")
        return self.replies.pop(0)


def make_backend(spec: str):
    """'anthropic[:model[:effort]]', 'local[:base_url[|model]]', or 'none'."""
    if spec in ("", "none"):
        return None
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        parts = rest.split(":") if rest else []
        return AnthropicBackend(*(parts[:2]))
    if kind == "local":
        url, _, model = rest.partition("|")
        kw = {}
        if url:
            kw["base_url"] = url
        if model:
            kw["model"] = model
        return LocalBackend(**kw)
    raise ValueError(f"unknown backend {spec!r}")
