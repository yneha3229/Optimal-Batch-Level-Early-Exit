## Analysis (26 Nov 2026, GPU server)

Conditions: other users' jobs ran on the machine throughout. To save time, the agent runs overlapped with the
baselines (on separate cores), so the baseline timings were not measured on an otherwise idle machine. All benchmark Qwen
numbers below are **Qwen2.5-Coder-32B-AWQ with prompts v3, before the plan-results fix 9081f70** (results of checks
run by the Solver's own plan were not shown to the agents). The model ablation below ran Qwen on CPU (llama.cpp) on the
fixed code, because from 18:59 UTC `cuInit` failed on every GPU of the server (error 3): the A6000 at PCI 3D:00.0
reports "GPU requires reset", which blocks CUDA machine-wide until an admin resets it.

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
- **Model ablation (bench/ablation/; 6 graphs x 2 relabelling seeds = 12 cells, one blind run per cell):**
  K(16,4), K(16,4) minus 1% of edges, K(14,4) (rule exception), K(13,4), K(15,5), and a random 6-regular graph on
  1000 vertices (non-Kneser control). Every model reaches the same verdict on every graph instance and the same
  bracket on the 10 Kneser-type instances; on the random control the lower bound differs (Claude 336, Qwen 331-333)
  because the Claude Solvers asked for 200-1,000 independent-set restarts and Qwen for 10. So the model changes the
  dialogue and its cost, and only through its plan the bound, never the correctness. The five Claude configurations (Haiku 5.5 medium,
  Sonnet 5.5 medium, Opus 5.5 low / medium / high; 12 runs each) made **0 false Critic objections in 60 runs**,
  named the right family in the first Solver reply in 11-12 of 12 runs, and needed 3.5-4.1 LLM calls per run. Qwen
  2.5-Coder 32B (CPU, fixed code, 12 runs): **25 false objections (2.08 per run)**, the right family in the first
  reply in 0 of 12 runs (on the random control it first called the graph "perturbed Kneser"), 8.7 calls per run;
  the 3-run GPU pilot on the old code (reference traces *_ref_*) gave 1.67 per run. Its Critic objects to numbers
  that are right ("the common-neighbour counts [70, 126, 210, 330] do not match K(16,4)"; "family_rule (14,5) was
  not queried" when it had been, while missing that (14,5) cannot fit 715 or 3003 receivers); it made one real
  catch ("3003 receivers do not match C(14,4) = 1001"). Every objection is labelled with a reason in
  bench/ablation/label_detail.json (rule in labels.py; Qwen CPU and new Claude rows labelled by Claude Code).
- **Qwen3-4B (CPU, GGUF Q4_K_M, fixed code, 12 runs):** at the pipeline's 600-token reply limit (the limit Qwen 32B
  ran under) its JSON was cut off in 26 of 48 requests (logs/kneser_agent_qwen4b_cap600_summary.txt), so it was rerun
  with a 1,500-token limit (the Claude agents had none). Then: **83 false Critic objections (6.92 per run)**, 23
  invalid replies and 9 fallbacks to the default plan (10 rejections were arithmetic caught by code: a K(n,k) whose
  C(n,k) is not the receiver count, or a false "C(a,b) = X"); right family in the first reply in 7 of 12 runs.
  Its false objections are confident mathematics: eigenvalue sums without multiplicities ("148 != 0"), "1820 is not a
  binomial coefficient", "K(16,4) has degree C(12,3)", "a 6-regular graph has at most 7 distinct eigenvalues",
  strongly-regular identities applied to Kneser graphs. It still never changed a verdict: every bound is checked.
- **Model ranking by false objections per run:** Claude (all five configurations) 0, Qwen 32B 2.08 (CPU) / 1.67 (GPU
  pilot), Qwen3-4B 6.92. LLM calls per run: Claude 3.5-4.1, Qwen 8.2-8.7.
- **Fast path (opt-in, no LLM):** recognising K(n,k) from the receiver count and degree and applying the rule proves the
  three large graphs K(20,4), K(17,5), K(23,4) in 4.2 / 14.5 / 21.4 s on 4 quiet cores (Kushagra's script: 18.7 / 23.0 /
  41.6 s; the agent without the fast path leaves gaps), with the same results as before on the 20 benchmark graphs.
- **Two issues the blind agents found:** the summary's sampled common-neighbour counts can miss a rare triangle
  that the exact triangle count sees (reg(1000,6): adjacent pairs "share no neighbour", yet 1 triangle); and the
  run result records the hypothesis of the accepted plan, not the checkpoint's revision. Neither touches a bound.
