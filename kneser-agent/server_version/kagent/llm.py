"""LLM access for the three agents. Reuses pincer/llm.py's LocalBackend (imported, not copied); the subclass only adds
vLLM's JSON response format. Every call is logged to the trace: agent, prompt hash, latency, tokens in/out, reply.
A reply that does not parse or breaks the schema is retried once with the error; a second failure returns None and
logs llm_fallback (the scheduler's default is used instead)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pincer.llm import LLMError, LocalBackend  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8000/v1"
MODELS = {"qwen32b": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507"}


class JSONBackend(LocalBackend):
    """LocalBackend + response_format json_object (vLLM guided decoding)."""

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        self.usage.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except Exception as e:
            self.usage.errors += 1
            raise LLMError(f"local LLM call failed: {e}") from e
        finally:
            self.usage.seconds += time.perf_counter() - t0
        u = d.get("usage") or {}
        self.usage.input_tokens += int(u.get("prompt_tokens", 0))
        self.usage.output_tokens += int(u.get("completion_tokens", 0))
        return d["choices"][0]["message"]["content"]


def parse_json(text: str):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    return json.loads(t)


class LLM:
    """One per run: model name, URL, call budget, and the shared usage counters for the run."""

    def __init__(self, model: str = MODELS["qwen32b"], url: str = DEFAULT_URL, max_calls: int = 12,
                 max_tokens: int = 500, temperature: float = 0.2, timeout: float = 120.0) -> None:
        self.model, self.url, self.max_calls = model, url, max_calls
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.calls = 0
        self.seconds = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.fallbacks = 0

    def left(self) -> int:
        return self.max_calls - self.calls

    def ask(self, agent: str, system: str, prompt: str, validate, trace) -> dict | None:
        """validate(obj) -> None if ok, else an error string. Returns the parsed object or None (fallback)."""
        err = None
        for attempt in range(2):
            if self.calls >= self.max_calls:
                trace.log("llm_fallback", agent=agent, reason="LLM call budget spent")
                self.fallbacks += 1
                return None
            p = prompt if err is None else (prompt + f"\n\nYour previous reply was rejected: {err}\n"
                                            "Reply again with strict JSON in the required schema.")
            be = JSONBackend(base_url=self.url, model=self.model, temperature=self.temperature,
                             max_tokens=self.max_tokens, timeout=self.timeout)
            h = hashlib.sha1((system + "\n" + p).encode()).hexdigest()[:10]
            self.calls += 1
            t0 = time.perf_counter()
            try:
                text = be.complete(system, p)
            except LLMError as e:
                dt = time.perf_counter() - t0
                self.seconds += dt
                trace.log("llm_call", agent=agent, attempt=attempt + 1, prompt_hash=h, latency=round(dt, 2),
                          tokens_in=0, tokens_out=0, error=str(e)[:300])
                err = f"API error {e}"
                continue
            dt = time.perf_counter() - t0
            self.seconds += dt
            self.tokens_in += be.usage.input_tokens
            self.tokens_out += be.usage.output_tokens
            try:
                obj = parse_json(text)
                err = validate(obj)
            except (ValueError, json.JSONDecodeError) as e:
                obj, err = None, f"not valid JSON ({e})"
            trace.log("llm_call", agent=agent, attempt=attempt + 1, prompt_hash=h, latency=round(dt, 2),
                      tokens_in=be.usage.input_tokens, tokens_out=be.usage.output_tokens,
                      prompt=prompt if attempt == 0 else p[-600:], reply=obj if obj is not None else text[:2000],
                      schema_error=err)
            if err is None:
                return obj
        trace.log("llm_fallback", agent=agent, reason=err)
        self.fallbacks += 1
        return None

    def usage(self) -> dict:
        return {"model": self.model, "calls": self.calls, "seconds": round(self.seconds, 2),
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "fallbacks": self.fallbacks}
