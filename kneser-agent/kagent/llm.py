"""LLM backends for the agents.

  chat     : every request is written to chat/req_<id>.json and the pipeline waits for chat/rep_<id>.json.
             A person (or another assistant) answers by writing the reply file. Used to test the architecture
             with the assistant in this conversation acting as the model.
  local    : an OpenAI-style server (vLLM with Qwen2.5-Coder-32B-Instruct-AWQ on the GPU server), via pincer/llm.py.
  none     : no LLM; the orchestrator uses its deterministic defaults (the `no_llm` ablation).
"""
from __future__ import annotations

import json
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
    def __init__(self, url="http://127.0.0.1:8000/v1", model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"):
        sys.path.insert(0, str(ROOT.parent))
        from pincer.llm import LocalBackend as LB
        self.b = LB(base_url=url, model=model, temperature=0.2, max_tokens=600)
        self.name = f"local:{model}"

    @property
    def calls(self):
        return self.b.usage.calls

    @property
    def seconds(self):
        return self.b.usage.seconds

    def complete(self, agent: str, system: str, prompt: str) -> str:
        return self.b.complete(system, prompt)


def make(spec: str):
    if spec == "chat":
        return ChatBackend()
    if spec.startswith("local"):
        _, _, rest = spec.partition(":")
        return LocalBackend(*(rest.split("|") if rest else []))
    return None


def parse_json(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1])
