## Analysis (26 Nov 2026, GPU server)

Conditions: other users' jobs ran on the machine throughout. To save time, the agent runs overlapped with the
baselines (on separate cores), so the baseline timings were not measured on an otherwise idle machine. All Qwen
numbers below are **Qwen2.5-Coder-32B-AWQ with prompts v3, before the plan-results fix d22d46c** (results of checks
run by the Solver's own plan were not shown to the agents). The rerun on the fixed code (v4) and the Qwen3-4B
ablation could not run: from 18:59 UTC, `cuInit` failed on every GPU of the server (error 3, also on an idle
3080 Ti), which needs an admin reset; already-running processes of other users were not affected.

- **Proven exact, 20 benchmark graphs:** standard methods 2, Kushagra's script 14, agent without LLM 14, agent +
  Qwen 32B 14 (14 with generic rules only). All 14 are the regular Kneser graphs; the 3 exceptions and 3 controls
  end at gaps for every method; no method claimed a false value, and every agent bound passed the independent
  verifier.
- **Large graphs (4,845-8,855 receivers):** Kushagra's script 3/3, agent + Qwen 32B with the rule 3/3, agent without
  LLM 0/3, agent + Qwen with generic rules only 0/3, standard methods 0/3 (open; no lower bound on K(23,4)). The
  agent computes no spectrum above N = 4000, so its generic search has nothing to try there; the family rule,
  reachable only through an LLM hypothesis, closes these graphs.
- **Speed:** without an LLM the agent is slower than Kushagra's script on the shared exact graphs (median time ratio
  x0.61; it is slower on 8 of 14, by at most 0.6 s), and x135 faster than the standard methods on the 2 graphs
  they prove. On the large graphs the agent's compute time is 5-8x below Kushagra's script (2.3 / 4.5 / 8.0 s vs
  18.7 / 23.0 / 41.6 s), but with the LLM seconds its wall time is about 2x longer on two of them and equal on K(23,4).
- **LLM vs no LLM:** on the 20 benchmark graphs Qwen proved nothing more and its compute time was about the same
  (median x1.05). It adds a median 43 s of LLM time per run (max 56 s). Its value showed only on the large graphs
  (3 proofs that the no-LLM run did not get) and on one control (reg(1000,6): lower bound 332 vs 327, from the extra
  independent-set restarts it asked for).
- **Family guess vs truth.json:** the first Solver reply named the right family and (n, k) in 2/40 runs (with v3 it
  usually asks binomial_match before giving (n, k)); the accepted plan was right in 27/40. On 5 exact Kneser graphs
  (K(40,2), K(53,2), K(12,3), K(21,3), K(17,4), rule allowed) the final label was a wrong "perturbed Kneser",
  after false Critic objections; no verdict changed, because the verdict comes from the checks.
- **What the Critic caught:** 110 challenges in 40 runs. Useful: the false arithmetic of prompts v1 ("C(14,7) = 1001",
  "C(14,5) = 1820"; now also rejected by code), asking for binomial_match before trusting a guess, and making the
  Solver query the rule on K(14,4) (it names no field) before a generic search, as in the chat reference run. Not
  useful: repeated or wrong objections ("degree should be C(12,3) = 220" on K(16,4)).
- **Invalid replies:** 14 of 336 requests (9 were family_rule without arguments), 2 fallbacks to the default plan;
  no JSON syntax errors (vLLM JSON mode; also none in a run without it).
- **Generic rules only vs with the family rule:** identical on the 20 benchmark graphs (the generic spectral search
  closes every regular Kneser graph up to 3,003 receivers); 0 vs 3 on the large graphs.
- **Losses:** to Kushagra's script, on time, on most exact graphs (listed below); never on accuracy: on the 3
  exceptions both reach the same brackets ([84, 93], [286, 287], [495, 507]), on the 3 controls the agent's brackets
  are tighter (e.g. K(16,4)-del0.01: [455, 534] vs [0, 1818]). With prompts v1, Qwen lost to the no-LLM agent on
  K(14,4) ([286, 299] vs [286, 287]); the scheduler's safety net fixed this.
- **32B vs 4B:** not run (GPU failure above).
