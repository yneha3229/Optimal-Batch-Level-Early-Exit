"""LLM backends for the agents.

  chat     : every request is written to chat/req_<id>.json and the pipeline waits for chat/rep_<id>.json.
             A person (or another assistant) answers by writing the reply file. Used to test the architecture
             with the assistant in this conversation acting as the model.
  local    : an OpenAI-style server (vLLM with Qwen2.5-Coder-32B-Instruct-AWQ on the GPU server), via pincer/llm.py.
  none     : no LLM; the orchestrator uses its deterministic defaults (the `no_llm` ablation).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ChatBackend:
    name = "chat"

    def __init__(self, folder=ROOT / "chat", timeout=3600.0):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.calls, self.seconds = 0, 0.0
        self.n = max([int(p.stem.split("_")[1]) for p in self.dir.glob("req_*.json")] + [0])

    def complete(self, agent: str, system: str, prompt: str) -> str:
        self.n += 1
        req, rep = self.dir / f"req_{self.n:03d}.json", self.dir / f"rep_{self.n:03d}.json"
        req.write_text(json.dumps({"id": self.n, "agent": agent, "system": system, "prompt": prompt}, indent=1))
        t0 = time.perf_counter()
        while not rep.exists():
            if time.perf_counter() - t0 > self.timeout:
                raise TimeoutError(f"no reply to {req.name}")
            time.sleep(0.5)
        time.sleep(0.2)
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return rep.read_text()


class LocalBackend:
    """vLLM / OpenAI-style server via pincer/llm.py. json_mode (default on; KAGENT_JSON_MODE=0 turns it off) adds
    vLLM's response_format {"type": "json_object"} (guided decoding), as the server version's JSONBackend did: the
    reply is then always syntactically valid JSON; the schema is still checked by kagent/agents.py."""

    def __init__(self, url="http://127.0.0.1:8000/v1", model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", json_mode=None):
        sys.path.insert(0, str(ROOT.parent))
        from pincer.llm import LocalBackend as LB
        self.json_mode = (os.environ.get("KAGENT_JSON_MODE", "1") != "0") if json_mode is None else json_mode
        self.b = LB(base_url=url, model=model, temperature=0.2, max_tokens=600)
        if self.json_mode:
            self.b.complete = self._complete_json
        self.name = f"local:{model}"
        self.last_raw = ""

    def _complete_json(self, system: str, prompt: str) -> str:
        import urllib.request
        from pincer.llm import LLMError
        b = self.b
        body = json.dumps({"model": b.model, "temperature": b.temperature, "max_tokens": b.max_tokens,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(b.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        b.usage.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=b.timeout) as r:
                d = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            b.usage.errors += 1
            raise LLMError(f"local LLM call failed: {e}") from e
        finally:
            b.usage.seconds += time.perf_counter() - t0
        u = d.get("usage") or {}
        b.usage.input_tokens += int(u.get("prompt_tokens", 0))
        b.usage.output_tokens += int(u.get("completion_tokens", 0))
        return d["choices"][0]["message"]["content"]

    @property
    def calls(self):
        return self.b.usage.calls

    @property
    def seconds(self):
        return self.b.usage.seconds

    def complete(self, agent: str, system: str, prompt: str) -> str:
        self.last_raw = self.b.complete(system, prompt)
        return self.last_raw


def make(spec: str):
    if spec == "chat":
        return ChatBackend()
    if spec.startswith("local"):
        _, _, rest = spec.partition(":")
        return LocalBackend(*(rest.split("|") if rest else []))
    return None


def parse_json(text: str) -> dict:
    """The reply's JSON object: inside a ``` fence if there is one, else from the first '{' to the last '}'
    (the server version's parser)."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    return json.loads(t)
