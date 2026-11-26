# Merge notes: branch version (kneser-agent/) and server version (kneser-agent/server_version/)

Two independent implementations of the same idea (Solver/Critic/Verifier LLM agents around a deterministic
scheduler, a bound board that only accepts certified tool results, and an independent verifier) existed on 26 Nov:

| | **branch version** (`kneser-agent/`, pushed to `feature/kneser-agent`, commits 79a5449, 9b5ed59) | **server version** (written on the GPU server on 26 Nov 09:16-12:37, untracked in the `main` worktree `~/ie624-main/kneser-agent/`) |
|---|---|---|
| size | ~1,270 lines (kagent + bench) | ~4,200 lines in 23 modules, 63 tests |
| entry point | `python -m kagent.pipeline "K(16,4)" --llm none\|chat\|local` | `python -m kagent.run bench/graphs/<id> --mode full\|no_critic\|no_llm\|small_model` |
| graph names | readable (`K_16_4_s0`), truth in `bench/truth.json` | opaque ids (`g6fd833`), truth mapped by the driver only |
| scheduler | `ProcessPoolExecutor`, 3 workers, events: new bound, near miss, closed, plan exhausted -> one checkpoint | own pool with workers pinned to cores, killed by PID; events NEW_LOWER/UPPER, NEAR_MISS, STALL, CONTRADICTION (LLM predictions vs certified bounds), NEW_PHASE; LLM calls in a background thread so tools never wait |
| lower bound | greedy independent set | iterated local search (numba) + CP-SAT (n <= 300) + Hoffman/Lovasz info |
| upper bounds | greedy clique cover; codes I + cA over F_p (spectral ranking, `family_rule`, `try_code`) | DSatur + TabuCol clique cover, fractional cover (column generation), spectral codes over every prime <= n, F_2 minrank SAT (n <= 150), blow-up vector codes (n t <= 150) |
| **family rule** | yes: the theorem as one library rule, switchable (`--no-family-rule`) | **no**: generic rules only (the Solver can only name primes / (p, theta) pairs) |
| rank engine | `factor_check` (BLAS) when rank = \|S\|, else full flint rank, only N <= 4000 | `rank.py`: numba echelon on a sample + flint inverse + BLAS residual; exact rank and factorization M = C M[S,:] for any N, early stop at a cap |
| prompts | inline in `agents.py`, short; tool names spectral/try_code/family_rule/... | `prompts/*.txt`, longer, with the rank_pred formula, settings, predictions; tool names code_rules/frac_cover/... |
| validation | schema only | schema + every evidence item must quote a number of the summary |
| LLM backends | `chat` (file-based: an assistant answers), `local` (vLLM), `none` | `local` with vLLM JSON mode (`response_format`), call budget (12), token counts per call |
| verifier | `verify.py`: own elimination, exact blocked factorization check, decoding simulation; LLM may request decode_more/own_rank | `verify/checks.py` + `stage.py`: Freivalds (30 rounds) + exact own elimination (n <= EXACT_MAX), fractional cover in exact fractions, UNSAT CNF re-solved with Glucose; no flint, no kagent.rank |
| demo | live one-line-per-event print | `demo.py`: rich live terminal view (dialogue, board, workers, baseline timers) and trace replay |
| baselines | `bench/baselines.py`: CP-SAT alpha/cover, fractional cover, F_2 minrank SAT (from `minrank-agent/kneser_bench.py`), Kushagra's script | `kagent/baselines.py`: indep_cpsat, clique_cover_best, frac_cover, sat_minrank_f2, lambda_search, pincer, minrank_agent |
| results so far | container runs (4 cores), 3 chat-backend reference traces | server runs: 5 graphs no_llm, 3 graphs full with Qwen 32B (all 3 proven; the first vLLM server died at 12:04 on GPU 5, log kept), baselines on 4 graphs |

## Conflicts

- **Same folder name, very different code** (`kneser-agent/kagent/`): neither overwrites the other. The server version
  is moved, unchanged except for four repository-root path depths, to `kneser-agent/server_version/` (its 63 tests
  pass there). Its 80 MB of generated graphs (`bench/graphs/`) are not committed (gitignored; regenerable, and backed
  up in `~/agent_backup_1795715420/`).
- **Agent prompts and tool vocabularies differ** (`family_rule`/`try_code`/`spectral` vs `code_rules`/`frac_cover`/
  `sat_minrank`). The branch's vocabulary is kept in `kagent/`, because the benchmark, the reference chat traces and
  the `--no-family-rule` ablation are built on it. The server prompts stay with the server version.
- **Rank engine**: the branch's engine is replaced for the non-closing case by the server's (below). The verifier's
  own elimination is unchanged and still shares no code with it.
- **Demo view**: the server's rich view is tied to its orchestrator's trace format; it stays in `server_version/`.
- **Graph naming**: readable names (branch) vs opaque ids (server). Both hide the family from the agents (the agents
  only see the numeric summary); the branch's naming is kept for the benchmark.

## What was ported into kneser-agent/kagent/ (each covered by `tests/test_merged.py`)

1. `kagent/rank.py` (from server `kagent/rank.py`, unchanged): used by `tools.try_code` when the code does not close
   the bracket. Exact for any N (the branch returned no rank above N = 4000); the scheduler passes cap = U - 1, so a
   code that cannot improve the upper bound is abandoned early. Tested equal to flint and to the verifier's own
   elimination on 4 graphs x 5 primes; the factorization M = C M[S,:] is checked in the test. The numba kernel is
   loaded once before the run clock starts (`warmup`, idea from server `warmup.py`); the time is recorded as
   `warmup_s`. Effect with `--llm none`: K(14,4) 6.65 s -> 2.00 s, K(16,4) 2.55 s -> 1.78 s, same brackets.
2. Solver reply validation (from server `agents/__init__.py`): a family hypothesis must come with evidence items that
   each quote a number of the graph summary; otherwise the reply is rejected and retried once, then the default plan
   is used.
3. vLLM JSON mode (from server `llm.py` JSONBackend): `response_format: json_object` for the `local` backend, on by
   default (`KAGENT_JSON_MODE=0` turns it off); the robust JSON parser; the raw reply is kept in `llm_invalid` events.

4. A rich terminal view, `python -m kagent.view <trace> [--replay 1]` (a light port of server `demo.py` to this
   pipeline's trace format): agent dialogue, bound board, scheduler events, verdict.
5. The server version's rule "a wrong plan costs only time", as a safety net in the scheduler: if the LLM's plans
   leave the bracket open after the checkpoint, the deterministic generic search still runs.

After the first Qwen 32B runs (see the commit message of a46af20 and `traces/*_promptsv1_*`), more fixes in
`kagent/` (schemas unchanged): deterministic arithmetic check of the Solver's evidence ("C(a,b) = X" claims, and
C(n,k) = receivers for a Kneser hypothesis), tool-argument validation, bad arguments logged and skipped instead of
crashing the run, check history shown to both agents, prompt rules (copy (n, k) from binomial_match, query the rule
first, keep a generic step, a rule naming no field is an exception). `PROMPT_VERSION` = v3 is recorded per run.

Known limit of the merged pipeline (not changed): LLM calls block the main loop, so tool results that finish during
an LLM call enter the board only after it (the workers keep computing). The server version runs LLM calls in a
background thread.

Not ported (kept in `server_version/`): frac/tabu/CP-SAT covers, ILS independent sets, F_2 SAT, blow-up codes,
Lanczos spectrum for large n, the CONTRADICTION/STALL events, the rich demo. They are useful but would change the
tool vocabulary the benchmark and the reference traces use; on the Kneser benchmark the codes close the bracket.

## Rules kept

- No number comes from an LLM: the board accepts only tool results with certificates (`offer_lower` re-checks the
  independent set, the clique cover is re-checked, code lengths come from `try_code`).
- The verdict is computed from the checks (`finish`), the Verifier LLM only writes the analysis and may add checks.
- `kagent/verify.py` imports nothing from `kagent/tools.py` or `kagent/rank.py` (asserted by a test).
- The LLM is called only at the start (Solver <-> Critic, at most 3 rounds), at the checkpoint, and at the end.
