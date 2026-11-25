# When is the lookahead rule optimal? (report item 5.8, 24 Nov 2026)

Rules (2) and (3) are as in the report (`experiments/local_hw.py::theory_c`). Exact checks of every claim on small
synthetic instances: `theory/verify_58.py` → `results/theory58_checks.json` (CPU, 10 s). Real-trace test of the
corrections suggested by the theory: `experiments/theory58_front.py` → `results/theory58_front_*.csv`.

**Summary.**
- **Rule (3), collective stop:** proved optimal in the monotone case, for any cost c(n) (Theorem 1).
- **Affine cost:** at the two ends of the range of ρ the problem reduces to one we can solve (Theorem 2):
  - at ρ = 0 it splits into one stopping problem per input;
  - at ρ = ∞ it is pure collective stopping.
  At ρ = 0, rule (2) as written is *not* the optimal per-input rule. It compares only against the last exit, so it
  sends inputs out early, never late. We give an exact bound on what this costs. The optimal rule is the
  all-exits lookahead that rule (3) already uses.
- **Floor-shaped cost:**
  - Below the knee, individual exits are worthless and the implemented rule is optimal in the monotone case
    (Proposition 3a). This is the regime switch inside a batch.
  - Above the knee, rule (2) over-counts the saving when several inputs leave together; we give a counterexample
    and an exact bound (Proposition 3b).
- **Between the two ends of ρ:** we have no theorem. Exact DP on small batches shows small average gaps but no
  optimality.
- **On real traces:** after λ is re-tuned, neither correction speeds up the rule at matched held-out accuracy. The
  implemented rule stays as it is.

## 1. Model

- Evaluated exits 1, …, m sit at layers d_1 < … < d_m = L.
- Input j's confidence information at exit e is X_{j,e}.
- The rule uses p_e(x) = P(correct at e | X_e = x) and the calibration table
  q_e(x, e′) = P(correct at e′ | X_e = x) = E[p_{e′}(X_{e′}) | X_e = x], for e′ > e.
- The gain is g_j(e→e′) = q_e(X_{j,e}, e′) − p_e(X_{j,e}).

Assumptions:
- **M1 (independent Markov inputs).** Each X_{j,·} is a Markov chain, and the chains are independent across inputs.
- **M2 (exact calibration).** p and q are the true conditional probabilities. In practice they are a shrunk,
  isotonic plug-in estimate.
- **M3 (deeper is not worse in expectation).** q_e(x, e+1) ≥ p_e(x), i.e. p_e(X_e) is a submartingale. The
  implementation clips negative gains to 0, which imposes this on the table.

Under M2 the expected objective of any policy is E[J] with

> J = (1/B) Σ_j p_{τ_j}(X_{j,τ_j}) − (λ/F) T,

where τ_j is the exit at which input j leaves and T is the latency (1). This holds because, by the tower property,
replacing "correct" with p does not change expectations. Policies may use all confidences seen so far.

**Optimal stopping facts used.** Let Y_e be the reward of stopping at e, adapted to the information F_e, with a
finite horizon m.
- The **one-step lookahead** (OLA) stops on A_e = {Y_e ≥ E[Y_{e+1} | F_e]}.
- The problem is **monotone** if A_e ⊆ A_{e+1} on every path: once OLA says stop, it keeps saying stop.

**Fact 1 (monotone case [11]).** In a finite-horizon monotone problem, OLA is optimal.

*Proof (backward induction on V_e, the optimal value from e).* We claim V_e = Y_e on A_e.
- At e = m − 1 this is the definition of A_e.
- If it holds at e + 1, then on A_e ⊆ A_{e+1} we have V_{e+1} = Y_{e+1}.
- Hence E[V_{e+1} | F_e] = E[Y_{e+1} | F_e] ≤ Y_e on A_e (A_e ∈ F_e), and so V_e = max(Y_e, E[V_{e+1} | F_e]) = Y_e.
- Off A_e, Y_e < E[Y_{e+1} | F_e] ≤ E[V_{e+1} | F_e], so continuing is optimal. ∎

**Lemma 1 (all-exits lookahead = OLA in the monotone case).** Let Â_e = {Y_e ≥ E[Y_{e′} | F_e] for every e′ > e}.
In a monotone problem, Â_e = A_e, so the all-exits lookahead is also optimal.

*Proof.*
- Â_e ⊆ A_e: take e′ = e + 1.
- A_e ⊆ Â_e: on A_e, Fact 1 gives Y_e = V_e. Since V_e ≥ E[Y_{e′} | F_e] for every fixed e′ (a fixed time is a
  stopping time), Y_e ≥ E[Y_{e′} | F_e] for every e′ > e. ∎

So in the monotone case it does not matter whether a rule looks one exit ahead or at every later exit. Outside the
monotone case they differ, and both can be suboptimal.

## 2. Rule (3): collective stopping, for any cost

**Theorem 1.** Fix the active set R from exit e₀ on, so there are no more individual exits. If the collective
stopping problem is monotone, rule (3) is an optimal stopping rule for R. This holds for every cost function c(n)
and every head cost h.

*Proof.*
- With R fixed, the latency of the segment from e to e + 1 is deterministic: Δ_e = (d_{e+1} − d_e) c(|R|) + h.
- Hence Y_e = (1/B) Σ_{j∈R} p_e(X_{j,e}) − (λ/F) Σ_{i<e} Δ_i.
- "Y_e ≥ E[Y_{e′} | F_e]" reads (1/B) Σ_{j∈R} g_j(e→e′) ≤ (λ/F) [(d_{e′} − d_e) c(|R|) + (e′ − e) h].
- That is rule (3)'s test for exit e′. Rule (3) stops when it holds for every e′ > e, i.e. on Â_e.
- Lemma 1 and Fact 1 finish the proof. ∎

**A sufficient condition (A3).** The batch's one-step gain G_e = (1/B) Σ_{j∈R} g_j(e→e+1) does not increase along
any path, and the per-step price κ_e = (λ/F) Δ_e does not decrease. Equally spaced exits give a constant κ.

*Why it suffices:* G_e < κ_e implies G_{e+1} ≤ G_e < κ_e ≤ κ_{e+1}.

A3 is stronger than needed; the exact check tests monotonicity directly.

*Note.* The shape of c plays no role in Theorem 1: with R fixed, c(|R|) is a constant per layer. The difficulty
with a measured cost comes only from its interaction with individual exits (Sections 3–4).

**Exact check (T1).** 400 random instances: 4 exits, 3 confidence bins, 3 inputs, affine and floor costs, h > 0.
- Rule (3) equals the DP optimum in all 210 instances that are monotone on the reachable states.
- In the 190 non-monotone ones it is suboptimal in 17, by at most 2.3% of the value.

## 3. Affine cost c(n) = a + b n: the two limits

**Theorem 2.**

**(i) ρ = 0 (a = h = 0).**
- The latency is T = b Σ_j d_{τ_j}, so J = Σ_j [p_{τ_j}(X_{j,τ_j}) / B − (λ b/F) d_{τ_j}].
- J is a sum of single-input stopping problems. By M1 the optimum is reached by per-input rules, and the
  collective stop adds nothing.
- In the monotone case, the optimal per-input rule is the all-exits lookahead (Lemma 1):

> input j leaves at e iff for every e′ > e: g_j(e→e′) / B < (λ b / F)(d_{e′} − d_e). &nbsp; (2′)

- Rule (2) is the special case that tests only e′ = m. Two consequences follow:

  **(a) Rule (2) is never late.** In the monotone case, the optimum's exit set is contained in rule (2)'s. On A_e,
  Lemma 1 gives the test for e′ = m. So rule (2) continues only where the optimum also continues, and the two agree
  until rule (2)'s exit τ₂.

  **(b) Loss bound.** Under M3 the expected loss of rule (2) against the optimum is

  > E[V_{τ₂} − Y_{τ₂}] < (λ b / F) · E[L − d_{τ₂+1}],

  i.e. at most the per-input cost of the layers from the exit after the one where rule (2) lets the input go, to
  the end.

  *Proof.*
  - Rule (2) leaves at e, so (1/B) E[p_m − p_e | F_e] < (λ b/F)(L − d_e).
  - For any stopping time τ with e < τ ≤ m: by optional stopping for the submartingale p (M3),
    E[p_τ | F_e] ≤ E[p_m | F_e], and d_τ ≥ d_{e+1}.
  - So E[Y_τ − Y_e | F_e] < (λ b/F)(L − d_e) − (λ b/F)(d_{e+1} − d_e) = (λ b/F)(L − d_{e+1}).
  - Taking the supremum over τ and the expectation over τ₂ gives the bound. ∎

**(ii) ρ = ∞ (b = 0).** c(n) = a is constant.
- An individual exit, while another input stays, saves nothing.
- Under M3 it can only lose: keep the input until the collective stop instead, ignoring it in all decisions. The
  latency is unchanged, and by optional stopping its expected accuracy does not fall.
- So there is an optimal policy with no individual exits. It is collective stopping of the whole batch, and
  Theorem 1 applies. Rule (2) makes no individual exits here (marginal cost 0, clipped gains ≥ 0).

**(iii) 0 < ρ < ∞.**
- The coupling term a·(depth of the batch) does not split across inputs, and rules (2)–(3) are not the joint optimum
  in general.
- Exact DP of the full joint problem (any subset may leave at any exit) for B = 3: rules (2)+(3) are optimal in
  122/150 random instances. The mean gap is 0.0037 in objective units (about 0.4 points of batch accuracy at λ = 0)
  and the maximum is 0.135.
- We have no theorem here.

**Exact checks (T2).** 1,000 single-input instances with 5 exits.
- In the 301 monotone ones:
  - (2′) is optimal in every case;
  - rule (2) is never later than (2′), and it is strictly suboptimal in 165 (mean loss 0.048, max 0.45, in
    accuracy units of that input).
- In the 31 instances that are both monotone and satisfy M3, the bound (b) holds in every case. The mean
  loss/bound ratio is 0.10 and the maximum is 0.80.

## 4. Floor-shaped cost: the knee

Take the floor-plus-linear cost of Section 4.3: c(n) = c₀ for 1 ≤ n ≤ n_k, and c₀ + b(n − n_k) above the knee n_k.

**Proposition 3a (regime switch at the knee).** Once |R| ≤ n_k:
- under M3 there is an optimal policy that makes no further individual exits;
- if the remaining collective problem is monotone, rule (3) is optimal from that point;
- rule (2) makes no individual exits there.

So the implemented rule (2)+(3) is optimal below the knee in the monotone case. Above the knee it uses per-input
exits, and below it only whole-batch stopping. This switch happens inside one batch as it drains.

*Proof.* The cost does not change as the active set shrinks within [1, n_k], so the argument of Theorem 2(ii)
applies verbatim. The active set never grows again, so the batch stays below the knee. Rule (2)'s marginal saving
c(n) − c(n−1) is 0 for 2 ≤ n ≤ n_k. ∎

**Proposition 3b (over-counting above the knee).** Rule (2) is applied to all inputs at once, each priced at
today's marginal cost m(n) = c(n) − c(n−1).
- When k inputs leave together, it credits a saving of k · m(n) per layer. The real saving is c(n) − c(n − k).
- For a convex c, which the floor cost is, k · m(n) ≥ c(n) − c(n−k).
- For the floor cost the excess is b · max(0, k − (n − n_k)) per layer: every input that takes the batch below the
  knee is credited b but saves nothing.
- Each leaving input had gain/B < (λ/F) b (L − d_e), and the saving is at most (λ/F)(L − d_e)(c(n) − c(n−k)).
- So the objective lost by one such exit event, compared with keeping those inputs, is less than

> (λ/F) (L − d_e) · b · max(0, k − (n − n_k)).

*Counterexample (tight up to ε).*
- Take n = n_k + 1 inputs: one hard input with a large gain, and n_k easy inputs, each with gain/B just below
  (λ/F) b (L − d_e).
- Rule (2) sends all n_k easy inputs out, and the batch keeps running for the hard one.
- The saving is c(n_k + 1) − c(1) = b per layer, while n_k gains are lost.
- The net change is −(n_k − 1) · (λ/F) b (L − d_e). It is −6.99 units for n_k = 8 (P3 in the script).

*Fix (sequential rule (2)).* Inputs leave one at a time, in increasing gain, while the next input's own marginal
saving c(n − k) − c(n − k − 1) exceeds its gain.
- For a convex c the saving c(n) − c(n − k) is concave in k, and the lost gains (sorted) are convex in k.
- So the greedy order maximises saving minus lost gain over k for this one-step comparison.
- On the counterexample it changes the objective by +0.001 instead of −6.99.
- In the joint DP with a floor cost (B = 3, 150 instances) it raises the number of optimal instances from 109 to 115
  and lowers the mean gap from 0.0094 to 0.0058.

**What remains open for the floor cost.**
- The joint problem above the knee, where whether inputs should leave depends on whether the batch will drain
  below the knee before it stops.
- Rule (2) also prices the saving up to layer L, although the batch may stop at an earlier depth D. That
  overestimate is at most (λ/F) m(n) (L − D) per leaving input.

## 5. Do the corrections help on real traces?

`experiments/theory58_front.py` adds the families `theory_la` (rule (2′) instead of (2)), `theory_seq` (sequential
rule (2)) and `theory_la_seq`, with and without the exit plan. Rule (3) is unchanged.

The protocol is the one of `local_hw.py front`: certified selection at each accuracy target, scored once on the
held-out half, cost-table replay. It covers QQP and MNLI, with the GTX 1650 tables (T = 128, 32, 16) and the
A6000 tables (T = 128, 32), at B = 8, 16, 64, 128, i.e. 40 settings per family.

- **At the one-point certified target,** (2′) is faster than (2) on average:
  - without the plan: +2.7%, 29 wins and 3 losses out of 40;
  - with the plan: +2.2%.
  But it also spends more of the budget: mean held-out drop 0.99 vs 0.91 points.
- **At matched held-out drop,** taking the fastest policy with held-out drop ≤ 1 point, (2′) is −1.4% on average
  (13 wins, 21 losses).
  - At ≤ 0.5 points it is +2.9%, on only 20 settings.
  - The sequential version has no consistent effect: +0.9% at ≤ 1 point, with losses of up to −19% at the
    certified target.
- **Reading.** λ is tuned per setting, and re-tuning λ absorbs most of the difference: rule (2) leaving early acts
  like a slightly larger λ for the individual test. The suboptimality that the theory finds is real in synthetic
  instances, but on these traces it is within selection noise. The implemented rule stays as it is, and the
  theorems describe when it is optimal and how it can fail.

## 6. What this lets the report claim

- **Proved (monotone case, exact calibration, independent inputs):**
  - rule (3) is the optimal collective stop for any cost;
  - with an affine cost, the problem reduces to per-input stopping at ρ = 0 and to collective stopping at ρ = ∞;
  - with a floor cost, individual exits are worthless below the knee, and the rule is optimal there.
- **Proved, negative:** rule (2) is not the optimal per-input rule even at ρ = 0. It is never late, and its loss is
  bounded by the cost of the layers after the next exit. With a floor cost, its simultaneous pricing over-counts
  the saving, by a bounded amount that can be attained.
- **Not proved:** optimality for 0 < ρ < ∞, or above the knee; whether A3 holds for the real calibration tables.
  "Optimal" stays out of the title.
