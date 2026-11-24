"""Supervised data and LoRA fine-tuning for the LLM controller (offline, CPU).

make_examples(logs): replay a behaviour controller over training logs; at every decision record (prompt, option
the hindsight oracle would take). The behaviour controller follows the oracle 70 % of the time and a random option
otherwise, so the data also covers states the model reaches after its own mistakes (DAgger-style).

finetune(examples, base, out): LoRA (rank 16, attention projections) on a small instruct model; the loss is on the
answer digit only. The merged model is saved to `out` and loaded by controllers.LLMController.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .controllers import OracleController, options_for, prompt_for
from .replay import Sim


class _Teacher:
    """Behaviour policy for data collection (one action per decision, like the LLM controller)."""

    def __init__(self, sink: list, rng: random.Random, follow: float = 0.7):
        self.sink, self.rng, self.follow = sink, rng, follow
        self.oracle = OracleController()
        self.reflex_done = False

    def decide(self, view):
        if not self.reflex_done:
            self.reflex_done = True
            acts, free = [], view.free
            for k in sorted(view.open_comps(), key=lambda k: -view.comp(k)["n"])[:1]:
                for m in ("acyclic", "cover"):
                    if free > 0 and m in view.candidates(k):
                        acts.append(("start", k, m))
                        free -= 1
            return acts
        opts = options_for(view)
        if len(opts) == 1:
            return []
        label = self.oracle.label(view, opts)
        self.sink.append({"prompt": prompt_for(view, opts), "label": label + 1, "n_opts": len(opts)})
        i = label if self.rng.random() < self.follow else self.rng.randrange(len(opts))
        kind, k, m = opts[i]
        return [] if kind == "wait" else [(kind, k, m)]


def make_examples(logs: list[dict], seeds: int = 3, latency: float = 0.2) -> list[dict]:
    out = []
    for s in range(seeds):
        rng = random.Random(s)
        for L in logs:
            Sim(L, _Teacher(out, rng), slots=3, budget=60, latency=latency).run()
    return out


def finetune(examples: list[dict], base: str, out_dir: str, epochs: int = 2, lr: float = 3e-4, batch: int = 16,
             threads: int = 4, cache_dir: str | None = None, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(base, cache_dir=cache_dir, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    digit = {i: tok.encode(f" {i}", add_special_tokens=False)[-1] for i in range(1, 10)}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    data = list(examples)
    hist = []
    model.train()
    for ep in range(epochs):
        rng.shuffle(data)
        tot, n_ok, n = 0.0, 0, 0
        for b in range(0, len(data), batch):
            chunk = data[b:b + batch]
            enc = tok([e["prompt"] for e in chunk], return_tensors="pt", padding=True, padding_side="left")
            logits = model(**enc).logits[:, -1]                      # next-token logits after "best:"
            ids = torch.tensor([digit[i] for i in range(1, 10)])
            sub = logits[:, ids]                                     # restrict to the answer digits
            mask = torch.tensor([[j < e["n_opts"] for j in range(9)] for e in chunk])
            sub = sub.masked_fill(~mask, -1e9)
            target = torch.tensor([e["label"] - 1 for e in chunk])
            loss = torch.nn.functional.cross_entropy(sub, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(chunk)
            n_ok += int((sub.argmax(-1) == target).sum())
            n += len(chunk)
            if (b // batch) % 50 == 0:
                log(f"epoch {ep} step {b // batch} loss {tot / n:.3f} acc {n_ok / n:.3f}")
        hist.append({"epoch": ep, "loss": tot / n, "train_acc": n_ok / n})
    model = model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"history": hist}


def accuracy(model_dir: str, examples: list[dict], threads: int = 4) -> float:
    from .controllers import LLMController

    c = LLMController(model_dir, threads=threads)
    ok = sum(c.choose(e["prompt"], e["n_opts"]) == e["label"] - 1 for e in examples)
    return ok / max(1, len(examples))


def save_jsonl(rows, path):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
