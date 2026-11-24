# Lambda agent: the best linear code for an index-coding network

An offline agentic pipeline that finds **λ₂(G)**, the smallest broadcast rate any linear code over GF(2) can reach
(scalar or vector-linear), and certifies it. Self-contained: it does not import other code in this repository.

```
β  ≤  λ₂  ≤  min( clique cover,  fractional clique cover,  r / t  for every code found )
↑ lower bounds: acyclic set, Shannon LP        ↑ upper bounds: explicit linear codes over GF(2)
```

When the upper bound meets the lower bound, **β = λ₂ is proven**, and the network has an optimal linear code.

## Why λ₂ is a tighter target than the clique cover

- A clique cover is a linear code of a special kind (one XOR per group), so λ₂ ≤ clique cover.
- *Scalar* linear codes (minrank) can do better on directed graphs: e.g. receivers in a ring that each know the next
  two messages need 8 transmissions by clique cover but only 6 with a scalar code (n = 8).
- *Vector-linear* codes split each message into t pieces and can do better still: five friends in a ring (C₅)
  need 3 transmissions with any scalar code, but 5/2 per message with t = 2, which is optimal.

## The reduction that makes it computable

A vector-linear code with block length t on G is exactly a scalar-linear code on the **blow-up G⁽ᵗ⁾**: receiver i
becomes t sub-receivers, and (i, a) has piece (j, b) whenever i has message j. So
λ₂(G) ≤ minrank₂(G⁽ᵗ⁾) / t for every t, and a SAT solver can search for such codes (`A = U Vᵀ` with rank ≤ r).
Every code found is an explicit matrix, checked by a plain verifier (identity diagonal blocks, zeros where a
receiver lacks a message, exact GF(2) rank).

## The pipeline (`lambda_agent/agent.py`)

For each strongly connected component (λ₂ and both bounds add up over components):

1. **Cheap integer bounds:** largest acyclic set and smallest clique cover (CP-SAT). Equal → done.
2. **Fractional bounds:** Shannon LP (exact rational certificate) and fractional clique cover. Equal → done.
3. **LP-guided code search.** If the LP says β ≥ p/q, the first question is "is there a code with block length q
   and p transmissions?". A yes proves λ₂ = β = p/q immediately. Then the other rates r/t between the bounds,
   smallest first. Each SAT call fixes a blown-up acyclic set (symmetry breaking: those rows of U are forced to the
   identity) and gets a short time slice that doubles only for questions that timed out.
4. Stop when the bounds meet or the budget is spent. Every step is logged with its reason.

## Benchmark (`bench.py`, results in `RESULTS.md`)

119 graphs with 6-10 receivers (the Shannon LP limits the size): 80 random digraphs, 24 random undirected graphs,
15 structured ones (odd cycles and their complements, directed circulants, Petersen). Same certified lower bound for
every method; 60 s per graph per method; one core.

| method | β pinned (= λ₂ proven) | mean gap | median time | 90th pct | max |
|---|---|---|---|---|---|
| Clique cover (classic) | 100 / 119 | 0.214 | 22 ms | 248 ms | 3.3 s |
| Fractional clique cover (standard vector-linear bound) | 108 / 119 | 0.176 | 23 ms | 249 ms | 3.3 s |
| Scalar minrank, plain SAT (standard minrank method) | 111 / 119 | 0.037 | 50 ms | 3.3 s | 63.3 s |
| Vector-linear, plain search | 119 / 119 | 0 | 38.7 s | 45.9 s | 63.3 s |
| Vector-linear, plain search told the lower bound | 119 / 119 | 0 | 31 ms | 595 ms | 4.3 s |
| **Lambda agent** | **119 / 119** | **0** | **4 ms** | **39 ms** | **2.4 s** |

What this shows:

- **Accuracy: linear codes close every gap here.** Each classical bound fails on a different family (clique cover on
  odd cycles and circulants, the fractional cover on directed circulants, scalar minrank on odd cycles); vector-linear
  codes pin all 119, e.g. λ₂ = 9/4 for the complement of C₉ (block length 4).
- **Speed: the agent is the fastest method that pins everything.** Against the plain vector search it is
  10,000x faster at the median, mostly because it knows when to stop. Against the fairest baseline (plain search
  that is told the lower bound) it is still 8x faster at the median and 15x at the 90th percentile, from the
  LP-guided first question, the symmetry fix and computing the LP only when the cheap bounds leave a gap.

Limits, stated plainly: the graphs are small (the Shannon LP needs n ≤ 10 per component), so this shows the method
on cases where β can be certified, not at scale; single runs; times on a shared 4-core container.

## Run it

```bash
python -m pytest lambda-agent -q                    # 27 tests, incl. SAT vs brute force on blow-ups
python lambda-agent/bench.py suite                  # the 119 graphs
python lambda-agent/bench.py run --limit 60         # all methods (resumable; about an hour on 3 cores)
python lambda-agent/bench.py report                 # RESULTS.md and the table the demo page shows
```

Requires `ortools`, `python-sat`, `scipy`, `networkx`, `highspy` and (for exact LP certificates) `python-flint`.
