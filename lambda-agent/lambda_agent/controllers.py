"""Controllers: who decides which move runs in which worker slot, and when to stop one.

Every controller has decide(view) -> list of ("start", comp, move) / ("stop", comp, move) actions. The same
controller objects drive the replay simulator (replay.Sim) and the live runtime (live.Live).

  StaticController   start everything, cheapest-looking first, never stop anything (a plain parallel portfolio)
  RuleController     the deterministic lambda-agent playbook, run in parallel: cheap bounds first, then LP and
                     fractional cover, then SAT questions with the LP-guided one first (rate = lower bound,
                     smallest block length), then rates upward; never stops a move
  LLMController      the fine-tuned language model chooses among the same candidates, one action per decision,
                     including stopping a running move; a reflex first wave (acyclic set + cover on the largest
                     open component) starts before the first model call, so the solvers never wait on the model
  OracleController   replay only: knows from the logs which moves pin each component soonest (the teacher)
"""
from __future__ import annotations

from fractions import Fraction

ONE_SHOT_ORDER = ["acyclic", "cover", "lp", "frac"]


def code_rate(name: str) -> Fraction:
    _, t, r = name.split(":")
    return Fraction(int(r), int(t))


def code_t(name: str) -> int:
    return int(name.split(":")[1])


def rule_key(view, k: int, name: str):
    """The playbook's order on one component."""
    c = view.comp(k)
    if name in ONE_SHOT_ORDER:
        return (0, ONE_SHOT_ORDER.index(name))
    rate = code_rate(name)
    return (1, 0 if rate == c["lo"] else 1, rate, code_t(name))


class StaticController:
    name = "static portfolio"

    def decide(self, view):
        acts, free = [], view.free
        for k in sorted(view.open_comps(), key=lambda k: -view.comp(k)["n"]):
            for name in sorted(view.candidates(k), key=lambda m: (m not in ONE_SHOT_ORDER, m)):
                if free <= 0:
                    return acts
                acts.append(("start", k, name))
                free -= 1
        return acts


class RuleController:
    name = "rule-based agent (parallel)"

    def decide(self, view):
        acts, free = [], view.free
        cands = [(rule_key(view, k, m), -view.comp(k)["n"], k, m) for k in view.open_comps() for m in view.candidates(k)]
        for _, _, k, m in sorted(cands):
            if free <= 0:
                break
            acts.append(("start", k, m))
            free -= 1
        return acts


# ------------------------------------------------------------------------------------- oracle (teacher)
def oracle_plan(comp_log: dict) -> list[str]:
    """Moves that pin this component soonest in hindsight: the earliest event reaching the best lower bound any
    move reaches, and the earliest event reaching the best upper bound (they meet when the component is
    pinnable). Returned in the order to start them (the slower one first)."""
    best_lo, best_hi = None, None
    for name, mv in comp_log["moves"].items():
        for ev in mv["events"]:
            if ev[1] == "lower":
                v = Fraction(ev[2])
                if best_lo is None or v > best_lo[0] or (v == best_lo[0] and ev[0] < best_lo[1]):
                    best_lo = (v, ev[0], name)
            elif ev[1] == "upper":
                v = Fraction(ev[2])
                if best_hi is None or v < best_hi[0] or (v == best_hi[0] and ev[0] < best_hi[1]):
                    best_hi = (v, ev[0], name)
    plan = sorted([x for x in (best_lo, best_hi) if x], key=lambda x: -x[1])
    out = []
    for _, _, name in plan:
        if name not in out:
            out.append(name)
    return out


class OracleController:
    name = "oracle (hindsight)"

    def decide(self, view):
        acts, free = [], view.free
        need = {}
        for k in view.open_comps():
            need[k] = [m for m in oracle_plan(view.comp(k)["log"]) if m in view.candidates(k)]
        for k in sorted(need, key=lambda k: -view.comp(k)["n"]):
            for m in need[k]:
                if free <= 0:
                    return acts
                acts.append(("start", k, m))
                free -= 1
        return acts

    def label(self, view, options):
        """Teacher signal for one decision: the index of the option the oracle would take."""
        running_needed = set()
        for k in view.open_comps():
            plan = oracle_plan(view.comp(k)["log"])
            for i, (kind, kk, m) in enumerate(options):
                if kind == "start" and kk == k and m in plan:
                    return i
            running_needed |= {(k, m) for m in plan}
        if view.free <= 0:
            for i, (kind, kk, m) in enumerate(options):
                if kind == "stop" and (kk, m) not in running_needed:
                    return i
        return next(i for i, o in enumerate(options) if o[0] == "wait")


# ------------------------------------------------------------------------------------------ LLM controller
MAX_OPTIONS = 9


def options_for(view):
    """The numbered choices offered to the model (at most 9, so the answer is one digit token)."""
    opts = []
    cands = sorted(((rule_key(view, k, m), k, m) for k in view.open_comps() for m in view.candidates(k)))
    if view.free > 0:
        for _, k, m in cands[: MAX_OPTIONS - 3]:
            opts.append(("start", k, m))
    for k, m, _ in sorted(view.running(), key=lambda x: -x[2])[:2]:
        opts.append(("stop", k, m))
    opts.append(("wait", -1, ""))
    return opts[:MAX_OPTIONS]


def _fmt_move(m: str) -> str:
    if m.startswith("code:"):
        _, t, r = m.split(":")
        return f"code t{t} r{r}"
    return m


def prompt_for(view, opts) -> str:
    """Compact state (about 100 tokens): the model never reads raw arcs."""
    lines = [f"time {view.time:.1f}s free {view.free}"]
    for k in sorted(view.open_comps(), key=lambda k: -view.comp(k)["n"])[:3]:
        c = view.comp(k)
        f = c.get("features", {})
        lines.append(f"c{k} n{c['n']} d{f.get('density', 0):.2f} w{f.get('two_way', 0):.2f} lo {c['lo']} hi {c['hi']}")
    run = [f"c{k} {_fmt_move(m)} {s:.1f}s" for k, m, s in view.running()]
    lines.append("run " + (", ".join(run) if run else "none"))
    lines.append("opts " + " | ".join(
        f"{i + 1} {kind} c{k} {_fmt_move(m)}" if kind != "wait" else f"{i + 1} wait" for i, (kind, k, m) in enumerate(opts)))
    lines.append("best:")
    return "\n".join(lines)


class LLMController:
    name = "fine-tuned LLM agent"

    def __init__(self, model_dir: str, threads: int = 1):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
        self.digit_ids = [self.tok.encode(f" {i}", add_special_tokens=False)[-1] for i in range(1, MAX_OPTIONS + 1)]
        self.decisions = 0
        self.seconds = 0.0
        self.reflex_done = False

    def choose(self, prompt: str, n_opts: int) -> int:
        import time

        t0 = time.perf_counter()
        ids = self.tok(prompt, return_tensors="pt").input_ids
        with self.torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        scores = [float(logits[d]) for d in self.digit_ids[:n_opts]]
        self.decisions += 1
        self.seconds += time.perf_counter() - t0
        return max(range(n_opts), key=lambda i: scores[i])

    def decide(self, view):
        if not self.reflex_done:  # reflex: cheap bounds on the largest open component, no model call needed
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
        i = self.choose(prompt_for(view, opts), len(opts))
        kind, k, m = opts[i]
        return [] if kind == "wait" else [(kind, k, m)]
