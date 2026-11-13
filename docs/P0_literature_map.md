# P0: Novelty check and literature map

Date: 2026-10-28. This is a web search, not a full systematic review. **Read** means I read the paper's text. **Abstract** means I saw only the abstract or a search snippet, so the claim carries [VERIFY].

## Verdict

**The core claim is still novel, but it is narrower than the playbook says.** Two parts of the pitch are already published:

1. *"Batching erases early-exit speedups"* is a known observation (DREX, GEEP, LEAP, FlexEE, Miao et al.). We can't claim the observation itself. Our contribution has to be measuring it as `c(n) = a + b·n` and deriving a policy from that measurement.
2. *"Collective batch termination"* already exists as heuristics. DREX implements and evaluates three whole-batch rules: **Consensus** (exit only if all agree, from Miao et al.), **Majority**, and **Greedy** (exit if any one sample agrees). DREX also derives a cost-based batch threshold, its Adaptive Rebatching Threshold `ART = (c / t_d)·b`.

What we found no prior work on:
- an **optimal-stopping derivation** that trades the accuracy gain of continuing against the batch cost `Δ·(a + b·n)`, with a **regime switch in ρ = a/(b·n)**;
- a **label-free, partial-feedback bandit** that tunes a *collective* termination rule; Hanawal-lab bandits are all per-sample;
- **LLM program search (FunSearch/OpenEvolve) applied to early-exit policies**. The closest work evolves KV-cache eviction and serving policies.

**Suggested reframing for the deck:** "Consensus and Greedy are the two endpoints of batch exit. We derive *where between them* is optimal as a function of ρ, then show that evolution and a bandit can find that point."

## Three closest papers, and the sentence that separates us from each

| # | Paper | Sentence we must write |
|---|---|---|
| 1 | **DREX**: Liu, Alexander, Kakarla, Arzani, Liu. *Dynamic Rebatching for Efficient Early-Exit Inference*. arXiv 2512.15705, Dec 2025 (UPenn + MSR). **Read.** | "DREX's break-even threshold decides whether *rebatching* is worth its overhead, so it optimizes throughput only and never trades away accuracy. We derive when the *whole batch* should stop by weighing the expected accuracy gain of stragglers against the per-layer cost `a + b·n`. Their Consensus and Greedy rules are the two extremes of our family." |
| 2 | **CeeBERT / UCBEE**: Bajpai & Hanawal, ACL Findings 2024; Pacheco, …, Hanawal, IEEE TNSM 2024. **Abstract.** | "These bandits learn per-sample exit thresholds, and their cost term scales with the sample's own depth. We use a batch-coupled cost, where the action is collective termination, and the reward is censored: if the batch stops, the gain from continuing is never observed." |
| 3 | **GEEP**: Li, Wang, Qiu, Chen, Gao, Zhou. *Accelerating BERT inference with GPU-efficient exit prediction*. Frontiers of CS 18(3), 2024. **Abstract.** | "GEEP avoids batch shrinkage by predicting exit layers and *regrouping* samples before inference, which is a scheduling fix. We keep the batch as given and change the *exit decision rule*, so the two approaches compose." |

## Full map

Axes: (a) models early-exit cost under batched GPU inference; (b) decides when to stop a whole batch; (c) bandits for exit thresholds; (d) LLM program search for inference policies.

| Paper | Venue, year | Axis | One-line method | Overlap with us | Status |
|---|---|---|---|---|---|
| DREX (Liu et al.) | arXiv 2512.15705, 2025 | a, b | Regroups the batch at each exit ramp; copy-free buffer; SLA-aware flush; break-even `ART = c/t_d · b`. Evaluates Consensus, Majority and Greedy whole-batch rules. Llama-EE-13B/70B, Qwen-EE-14B on A100/H200; +2–12% throughput. | Highest. It has collective rules and a cost threshold. But its threshold covers rebatching overhead, not accuracy; it gives no ρ theory; and it targets LLM decoding, not classification. | Read |
| Miao, Yan, Yao, Yang. *Efficient Inference Framework for Early-exit LLMs* | arXiv 2407.20272, 2024 | b | Runs the batch until every sequence clears the threshold (Consensus); fills KV caches. Up to 1.25× over vLLM. | Source of the Consensus baseline. No cost model or theory. | Abstract |
| GEEP (Li et al.) | Frontiers CS, 2024 | a | Predicts exit layer per sample, sorts and groups samples by it; ~2× on BERT. | Same problem (batch shrinkage), opposite lever (scheduling vs decision rule). | Abstract |
| SkipDecode (Del Corro et al.) | arXiv 2307.02628, 2023 | a, b | One exit point per sequence *position*, shared by the whole batch, monotone decreasing. 2–5× on OPT. | A fixed, data-independent batch-level schedule. Ours is data-dependent and optimal. | Abstract |
| Apparate (Dai et al.) | SOSP 2024 | a, c | Automatically places and tunes EE ramps in serving; latency-only mode keeps exited samples in the batch. | Online threshold adaptation under batching. Must check whether it uses labels and how it treats batch cost. [VERIFY] | Abstract |
| SEEB-GPU | ACM, 2025 (full text embargoed until 2026-12-03 per NSF PAR) | a | Deadline-aware heuristic that adjusts batch size and confidence thresholds on edge GPUs; "up to 15×". | Possibly close: it jointly sets batch size and threshold. Authors and method are unconfirmed. **[VERIFY: must read before claiming novelty]** | Snippet |
| Fluid Batching (Kouris et al.?) | arXiv 2209.13443 | a | Exit-aware *preemptive* batching on edge NPUs at sub-network granularity. | Scheduling, not the exit rule. | Abstract |
| LEAP | arXiv 2605.01058, 2026 | a | States that batched early exit runs until the slowest sample finishes, so speedup appears mostly at batch size 1. | Independent confirmation of H1 that we can cite. | Snippet |
| FlexEE | arXiv 2609.17008, 2026 | a | Same observation; sidesteps it by targeting single-user inference. | Cite for H1. | Snippet |
| UCBEE (Pacheco, …, Hanawal) | IEEE TNSM, 2024 | c | UCB over exit thresholds, label-free, log regret. | Per-sample bandit baseline. | Abstract |
| CeeBERT (Bajpai, Hanawal) | ACL Findings, 2024 | c | UCB over thresholds for cross-domain; reward = confidence − processing cost. | Main baseline. | Abstract |
| SplitEE (Bajpai, Trivedi, Yadav, Hanawal) | 2023 | c | Bandit to pick the exit/split layer with side information; sublinear regret. | Per-sample. | Snippet |
| UEE-UCB (Hanawal lab) | arXiv 2209.09480 | c | Unsupervised multi-exit bandit. | Per-sample. | Snippet |
| *Beyond Greedy Exits* | arXiv 2509.23666, 2025 | c | Learns the threshold policy and states cumulative regret; batch size 1. | Shows the threshold-learning gap, but no batching. | Snippet |
| FunSearch (Romera-Paredes et al.) | Nature, 2024 | d | LLM plus evaluator in an evolutionary loop over one function. | Method source. | Known |
| AlphaEvolve (DeepMind) | white paper, 2025 | d | Whole-file evolution; data-center scheduling and kernel gains. | Method source. | Abstract |
| *Barbarians at the Gate* / ADRS (Cheng, Liu, Pan, …, Stoica) | arXiv 2510.06189; follow-up 2512.14806 | d | OpenEvolve on systems policies (transaction scheduling, cloud spot scheduling). | Closest "evolve a systems policy" precedent. Numbers differ between versions, so cite the version. | Abstract |
| *Discovering KV Cache Eviction Policies via LLM-Guided Program Evolution* | arXiv 2608.14555, 2026 | d | OpenEvolve with a MAP-Elites archive and a cascade evaluator for inference eviction policies. | Closest inference-policy evolution. Not early exit. | Snippet |
| AgentServeSim | arXiv 2606.09613, 2026 | d | LLM rewrites serving-policy code, scored by a simulator; no future-trace access. | Same "simulator as honest judge" design. Cite it. | Snippet |
| *Evolution or Illusion?* (Oved, Pony, Naparstek, Barzelay, IBM) | arXiv 2609.19799, Sep 2026 | d | Strategy rankings flip with the seeds × iterations budget; report the frontier. | **Changes P4:** report evolution vs random search across several budgets, not a single one. | Abstract |

## Effect on the plan

- **Add DREX's Consensus / Majority / Greedy as baselines** in P4 and P6. They are the published whole-batch rules, so our rule has to beat or explain them. They take a few lines each in the simulator.
- **Rewrite "The gap"** in the playbook. Drop "Nobody… derives the optimal exit policy under the true batched cost" in favor of: "Whole-batch exit rules exist only as fixed heuristics (Consensus, Majority, Greedy), and the only batch-level cost threshold (DREX) prices rebatching overhead, not accuracy."
- **Before claiming novelty, read SEEB-GPU and Apparate in full.** These are the two papers most likely to weaken it.
- **P4 budget reporting:** follow *Evolution or Illusion?* and plot score against the number of programs for evolution and random search at 3+ budgets.

## Not checked

DeepSloth (playbook seed idea), DAdEE and BEEM details, and EE-LLM's batch policy. We also haven't run a Google Scholar forward-citation search on DREX or CeeBERT for anything from 2026, which is the cheapest next novelty check.
