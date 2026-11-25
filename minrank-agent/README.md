# minrank-agent: an adaptive multi-field agent for index-coding minrank

For an undirected side-information graph `G`, the broadcast rate satisfies `α(G) ≤ β(G) ≤ minrk_F(G)` for every
field `F`. The gap `α … clique-cover` that the earlier agents left open on dense graphs is **intrinsic** there:
a random graph has `α ≈ 2 log₂ n` and large minrank, and no method closes that. But on **structured** graphs the
gap can close exactly — and *which field closes it is graph-dependent*. This agent exploits exactly that.

## The idea (from the Kneser minrank proof)

The optimal Kneser code in the proof is `H = I + εD` — an instance of `I + c·A` for a scalar `c`. For **any** graph,
`I + c·A` is a fitting matrix (nonzero diagonal, zero off the edges), so

```
minrk_{F_p}(G) ≤ n − max_{λ≠0 in F_p} nullity_p(A − λI)      ("spectral construction")
```

Structured graphs have adjacency eigenvalues of high multiplicity, so this bound is strong — and the multiplicities
change with `p`, so the right field varies. Random graphs have simple spectra, so the bound is ≈ `n−1`: useless.
That is *why* the gap closes on structure and not on random graphs.

When this bound meets `α` (a certified independent set), `β = minrk_{F_p} = α` is **proven**, and `I + cA` is an
explicit optimal scalar linear index code over `F_p`, re-checked here (`bounds.verify_spectral`).

## The agent (`minrank_agent/agent.py`)

Per graph it keeps a certified bracket `α ≤ β ≤ upper` and:
1. `α` (max independent set, or the EKR formula for Kneser) — lower bound + certificate.
2. clique cover (CP-SAT) — the classic field-independent upper bound.
3. the **multi-field spectral construction**: compute the integer spectrum once, order the fields by predicted
   payoff, and try them (exact mod-`p` rank) until the bound meets `α` or the fields run out.
4. optional `F_2` minrank SAT when a small gap remains over `F_2`.

**Learned/adaptive parts** (`learn.py`):
- a **gate** that predicts, from cheap spectral features (one eigendecomposition, no mod-`p` ranks), whether the
  multi-field search can close the gap — so the agent skips it on hopeless graphs. 5-fold accuracy **0.88**.
- a field-ordering policy. Honest result: the spectrum heuristic ties plain ascending order (~2 fields tried), and a
  trained ranker does **not** beat it here — too few closed instances across six field-classes, and `F_2` is modal.

## Results (`study.py`, 37 structured graphs)

| strategy | gaps closed (β proven) |
|---|---|
| clique cover only (classic) | 3 / 37 |
| best single fixed field (F₃) | 6 / 37 |
| **multi-field agent** | **11 / 37** |

Total remaining gap (Σ `upper − α`): clique-only **182** → multi-field **62** (a 66% reduction). Among the Kneser
graphs the agent closes **8/11**, using **five different fields** (F₂,F₃,F₅,F₇,F₁₁) — no fixed field closes them all,
matching the proof (e.g. `K(7,2)` needs F₃, `K(10,3)` needs F₂, `K(11,2)` needs F₇). Where the gap is genuinely open
(`K(8,3)`: `α=21 < minrk_{F₂}=22`, Johnson, Paley, random), the agent reports an honest bracket and the gate flags it.

Every closed case ships a verified explicit code. Run: `python minrank-agent/study.py run && … report`;
`python minrank-agent/learn.py data && … train`; `python -m pytest minrank-agent/test_minrank.py`.

## Why this is the right place for an agent

The LP / QP / primality / discrete-log studies in this repo showed that where a mature integrated solver exists, an
orchestration agent adds almost nothing. Index-coding minrank has **no** such solver, the bounds genuinely disagree,
and the best field is instance-dependent — so adaptively combining constructions over multiple fields is a real,
verifiable win. This directly repairs the one weakness of the earlier `gap-agent`: it stops grinding on the
intrinsically-open dense-random gap and instead closes the structured gaps that matter.
