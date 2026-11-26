# kneser-agent: a multi-agent pipeline that certifies the broadcast rate β

Given only a side-information graph (vertices shuffled, no name), the pipeline brackets β between a certified
lower bound (an independent set) and the length of an explicit linear code. When the two meet, it outputs β
together with an optimal code, both re-checked by an independent verifier.

```
graph ──► Solver (LLM) ◄──► Critic (LLM)        hypothesis + plan, challenged for up to 3 rounds
            │ accepted plan
            ▼
        Scheduler (code) ── worker processes: summary, independent set, clique cover, codes I + cA over F_p
            │ events: new bound -> retarget; near miss -> more constants for that field; bounds meet -> stop;
            │         plan exhausted -> checkpoint back to the Solver
            ▼
        Bound board: only certified tool results can enter
            ▼
        Verifier (LLM + independent code): fit, exact factorization M = M[:,S] M[S,:], own elimination,
                                           decoding simulation; the verdict comes from the checks, not the LLM
```

- `kagent/graphs.py`: Kneser graphs, edge deletion, random regular graphs, relabelling, storage.
- `kagent/tools.py`: graph summary, greedy bounds, code construction, rule library.
- `kagent/rank.py`: exact rank over F_p by rank factorization with early stop (ported from the server version).
- `kagent/verify.py`: the verifier's own code (it imports nothing from `tools.py`).
- `kagent/agents.py`: prompts, JSON schemas, validation, fallback to defaults.
- `kagent/llm.py`: backends `chat` (a person or assistant answers files in `chat/`), `local` (vLLM / Qwen), `none`.
- `kagent/pipeline.py`: orchestrator, scheduler, board, trace, report.
- `bench/baselines.py`: standard methods and Kushagra's theory-guided script, on the same graphs.
- `bench/run_agent.py`, `bench/report.py`: benchmark and `RESULTS.md`.

Code rules in the library:
- `spectral`: generic. It reads (p, c) from the integer spectrum; c = −1/θ (mod p).
- `family_rule`: the theorem, used as one rule; disable it with `--no-family-rule`.
- `try_code`: any (p, c).

```bash
python -m kagent.pipeline "K(16,4)" --llm none            # deterministic plan
python -m kagent.pipeline "K(16,4)" --llm chat            # answer chat/req_*.json with chat/rep_*.json
python -m kagent.pipeline "K(16,4)" --llm "local:http://127.0.0.1:8000/v1|Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
python bench/baselines.py run --limit 60 && python bench/run_agent.py && cd bench && python report.py
```

Traces of every run (every event, LLM message and board change, with timestamps) are in `traces/`. Proven codes
are in `results/codes/`.

Tests: `python -m pytest -q tests` (rank engine vs flint and the verifier's own elimination, evidence validation,
the verifier imports nothing from the tools).

`server_version/` is a second, larger implementation of the same pipeline written on the GPU server (rich live demo,
frac/tabu covers, F_2 SAT, generic rules only). It is kept whole; see `MERGE_NOTES.md` for what each version has and
what was ported into `kagent/`.
