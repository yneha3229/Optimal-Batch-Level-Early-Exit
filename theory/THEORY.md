# When should a batch stop? Theory write-up

Implemented in `simulator/policies.py::theory_rule`, which the simulator uses. The real-GPU runner executes the same function.
The label-free version is `bandit/bandit.py::LabelFreeStopRule`. Citations marked [VERIFY] must be checked before
they go on a slide.

## 1. Model

- A batch has `B` inputs. Exits sit after layers `l_1 < … < l_E = L`.
- At exit `e`, the active set is `A_e`, with `n_e = |A_e|`.
- One layer on `n` active inputs costs
  `c(n) = a·1[n > 0] + b·n`.
  Every exit point evaluated adds an overhead `h`.
- The batch's latency is
  `T = Σ_e (l_e − l_{e−1}) · c(n_e) + h·(#exits evaluated)`,
  and the full model's latency is `F = L·(a + b·B) + h`.
- The objective trades accuracy against relative latency with a price `λ`:

```latex
J = \frac{1}{B}\sum_{j=1}^{B} \mathbf{1}[\text{input } j \text{ correct}] \;-\; \lambda\,\frac{T}{F}
```

- At exit `e`, the policy chooses a set `S ⊆ A_e` of inputs that leave with their current prediction. It may also
  stop the batch, in which case every remaining input leaves.
- The information about input `j` is its confidence at exit `e`. The calibration table gives
  `P_e(e' | c) = P(correct at exit e' | conf at e = c)`, estimated on the evolve split.

## 2. What each action saves

- **One input leaves alone at exit `e`** while others continue. Only its per-input share is saved: `b` on every
  remaining layer, so `b·(L − l_e)`. The fixed cost `a` is still paid, because the batch keeps running.
- **The remnant `R` stops at exit `e`.** Every remaining layer's full cost is saved,
  `Σ_{e'>e} (l_{e'} − l_{e'−1})·(a + b·n_{e'})`, plus the remaining exit overheads.

This asymmetry is the whole story. The fixed cost `a` can only be saved collectively.

## 3. The rule

Let `g_j(e→e') = max(0, P_e(e' | c_j) − P_e(e | c_j))` be the expected accuracy gain of carrying input `j` from
exit `e` to exit `e'`. Negative estimates are clipped to 0: they are mostly calibration noise, and acting on them
would make inputs exit even at `λ = 0`.

**Individual exit.** Input `j` leaves at exit `e` iff its gain from running to the end is worth less than the
per-input cost it would consume:

```latex
\frac{g_j(e \to E)}{B} \;<\; \lambda\,\frac{b\,(L - l_e)}{F}
```

**Collective stop.** The rest, `R`, stop at exit `e` iff no later exit is worth reaching:

```latex
\forall e' > e:\quad \frac{1}{B}\sum_{j \in R} g_j(e \to e') \;<\; \lambda\,\frac{(l_{e'} - l_e)\,(a + b\,|R|) + (e' - e)\,h}{F}
```

## 4. Regime switch in ρ = a / (b·n)

Divide the individual threshold by the collective per-input threshold. The ratio is

`b·B / (a + b·|R|) ≈ 1 / (1 + ρ)`,

with `ρ = a / (b·|R|)`, up to the ratio of layer spans. Two limits follow.

- **ρ → 0 (compute-bound; an encoder at large batch).** The two thresholds coincide. A collective stop is then
  just every input leaving individually at once, so it adds nothing. The rule reduces to per-input exits with a
  threshold that does not depend on the batch.
  - Measured: on MNLI in the encoder regime, `theory_indiv_only` matches `theory` at every batch size in
    `results/main_mnli_selected.csv`.
- **ρ → ∞ (fixed-cost-bound; decoder-like decoding).** The individual threshold goes to 0, so individual exits
  save nothing. The optimal policy becomes a pure collective stopping rule.
  - Measured: on MNLI in the decoder regime at B=64, per-sample exits give 0.94×, below the full model, while the
    collective rule gives 1.36× at matched accuracy.
- **The threshold drops as the batch drains.** Divide the collective condition by `|R|`. The mean gain per
  remaining input that justifies continuing is about `λ·Δ·(a/|R| + b)·B/F`, which *rises* as `|R|` shrinks. So
  the confidence needed to stop the stragglers *falls* as the batch drains.
  - This is H3's prediction. The bandit's learned threshold is plotted in `figures/ex5_bandit_*.png`.

## 5. When the rule is optimal (assumptions)

The rule is a lookahead over "continue to `e'`, then stop". Following the classical *monotone case* of optimal
stopping, it is optimal among policies that use only current confidences if all of the following hold. The
monotone-case result is Chow, Robbins & Siegmund, *Great Expectations*, 1971 [VERIFY].

- **A1, independence:** inputs are conditionally independent given their confidences.
- **A2, known calibration:** the calibration table is exact (or a consistent plug-in estimate).
- **A3, diminishing returns:** the marginal gain of one more segment is non-increasing in depth, for every
  confidence bin.
- **A4, linear cost:** cost is linear in `n` and non-increasing as the batch drains.

Under A3 and A4, once stopping is optimal it stays optimal, which is exactly the monotone case. In that case
one-step lookahead is optimal.

What breaks in practice:
- **A2 is estimated.** Hence the shrinkage and isotonic smoothing in `Calibration`, and the noise found in this
  project (Section 8).
- **A4 is only approximate.** In eager mode `c(n)` is closer to `max(a, b·n)`; see `docs/P1_profiling.md`.
- **Only the current confidence is used.** The rule is not the optimum of the full POMDP that also uses the
  confidence trajectory.

## 6. Slowdown attacks: a certified, detection-free bound

**Robust rule.** Replace the sum in the collective test with a trimmed sum `TS_m`, which leaves out the `m`
largest gains (`m = ⌊B/16⌋` or `⌊B/8⌋`). The individual-exit test is unchanged.

**Lemma (sandwich).** For nonnegative gains of the clean remnant `C` and `k ≤ m` attack inputs `K`:

```latex
TS_m(C) \;\le\; TS_m(C \cup K) \;\le\; TS_{m-k}(C)
```

- *Left inequality.* Adding an element either adds itself to the untrimmed part, or pushes a former top-`m`
  element into it. Either way the trimmed sum cannot fall.
- *Right inequality.* Removing the attackers plus the `m − k` largest clean gains leaves exactly `TS_{m−k}(C)`.
  The `m` largest of `C ∪ K` sum to at least as much as that removed set, so what remains is no larger.

**Theorem (pathwise robustness).** Every input's gain depends only on its own confidences. The remnant's cost term
`a + b·|R|` only grows when attackers are present. So for every batch and every `k ≤ m`:

```latex
\mathrm{depth}(C \cup K;\, m) \;\le\; \mathrm{depth}(C;\, m-k)
```

The attacked batch ends no later than the same clean inputs would under the rule trimmed by `m − k`. Victims'
latency is therefore certified by a clean-batch quantity, up to the attackers' own `b·k` per layer. No attacker
is ever identified. The case where every clean input has already left is covered too: the remnant is then only
`K`, its trimmed sum is 0, and the batch stops.

**Verified.** `experiments/robust_bound.py` checks the inequality batch by batch: 0 violations in 2,100 batches
on each of CIFAR-100 (real DeepSloth inputs), QQP and MNLI. The property test is
`tests/test_core.py::test_robust_rule_pathwise_depth_bound`.

**Price.** On clean batches the `m`-trimmed rule ignores the `m` hardest inputs' gains as well. At matched accuracy
this costs nothing; the robust rule is in fact slightly faster with decoder-like cost, B = 64:

| Task | m = 0 | m = 8 |
|---|---|---|
| CIFAR-100 | 1.05× | 1.07× |
| QQP | 1.89× | 1.92× |
| MNLI | 1.43× | 1.49× |

Ignoring the hardest few gains is also good straggler policy.

**What it does not give: flat latency.**
- The bound is relative to the `(m − k)`-trimmed clean rule. At `k = m` that is the untrimmed rule at the same λ.
- With DeepSloth on CIFAR-100, victims' latency under `m = 8` rises from 0.92× to 1.01× of the full model at `k = 4`.
  The plain rule reaches 1.05×.
- **Consensus**, the rule of Miao et al. that DREX evaluates, loses its entire speedup to a single attacker
  (0.95× → 1.07×).
- **Per-sample exits** have nothing left to lose: at B = 64 natural stragglers already push almost every batch to
  full depth.
- **Majority vote** is robust up to `B/2` attackers by construction. On QQP it degrades faster than the robust rule
  (0.60× vs 0.53× at `k = 8`).

## 7. Label-free bandit: regret sketch

- **Probes.** A probe batch runs to full depth. Every head is computed, so a probe reveals the whole trace and,
  counterfactually, the outcome of *every* table cell its inputs touch.
- **Probe rate.** Probes are drawn with probability `p_t = c·t^(−1/3)`, so there are about `1.5·c·T^(2/3)` probes
  by time `T`. Each costs at most `λ` in score, which totals `O(T^(2/3))`.
- **Estimation error.** After `m` probes a table cell's error is `O(sqrt(log m / m))`, with `m ≈ t^(2/3)`. The
  per-batch decision loss is linear in the table error and sums to `Σ_t t^(−1/3) = O(T^(2/3))`.
- **Total regret** against the rule with the true (fidelity) tables is `O(T^(2/3))`. This is the partial-monitoring
  rate for a revealing action (cf. label-efficient prediction, Cesa-Bianchi, Lugosi & Stoltz 2005 [VERIFY]).
- **Measured.** Cumulative regret over 3000 batches of 64 MNLI inputs grows with a log-log slope of **0.65** in the
  decoder regime and **0.49** in the encoder regime (`results/bandit_mnli_*_B64.csv`).
- **The target is fidelity, not accuracy.** The bandit optimizes agreement with the full model's answer, a
  label-free proxy. Disagreement with the full model upper-bounds the accuracy loss, so a 1% disagreement target
  guarantees a drop of at most 1 point. The guarantee is conservative: in practice the drop is well under 1 point,
  and the speedup is smaller than the label-tuned rule's.

## 8. Known gaps (say them before someone asks)

1. **Selection noise.** With about 4,900 evolve inputs, picking λ at the edge of the 1-point constraint lands
   1–2 points below on held-out data. `results/matched_table_*.md` compares methods at matched held-out accuracy.
2. **Ties with random search.** A random search over a 3-parameter hand-written collective rule is as good as the
   theory rule on MNLI and CIFAR. The theory rule's edge is clearest on QQP.
3. **Rescue.** Combining exits for cut-off inputs did not raise accuracy on the hardest 10% in any setting.
   Early exits are weaker than the exit where the input was stopped, so pooling dilutes the prediction.
4. **The decoder regime is a cost what-if.** Classification traces were replayed under a measured Qwen2.5-1.5B
   decode cost. It is not a decoder early-exit experiment.
