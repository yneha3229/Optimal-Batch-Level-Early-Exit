# dlog-agent: headroom study for discrete logarithm

Does an adaptive agent that picks and composes the classical discrete-log algorithms beat the generic solver?
Same method as the LP, QP and primality studies: run every method on every instance, verify every answer
(`g^x = h`), and compare the best single method against the per-instance oracle (the best method chosen in hindsight).

## Setup

- **Groups** (`dlog_agent/group.py`): `Fp` (integers mod a prime) and `EC` (elliptic curve over F_p), one interface.
- **Methods** (`dlog_agent/methods.py`): baby-step giant-step, Pollard rho, Pollard kangaroo (for a known interval),
  Pohlig–Hellman (factor the order, solve each prime-power subgroup, combine by CRT), and PARI's own `znlog`/`elllog`
  (which already does Pohlig–Hellman + index calculus internally).
- **Benchmark** (`make_instances.py`, 48 instances): five structure families — generic prime order, smooth order,
  partially smooth, interval-restricted exponent, and (F_p* only) sizes where index calculus beats sqrt(n) — at small
  benchmark sizes. The answer is stored only for checking.

Educational computational number theory (Cohen; Galbraith), at sizes far below any deployed parameters.

## Result (`results/headroom_report.txt`): no meaningful headroom

| family | best single method | per-instance oracle speedup |
|---|---|---|
| F_p* generic / smooth / partial / index-calculus | PARI | ×1.0 |
| EC generic / smooth / partial | PARI | ×1.0 |
| interval-restricted | kangaroo or PARI | ×1.0 |
| **all 48** | **PARI (44/48, sgm 0.76 s)** | **×1.1 (oracle 0.73 s)** |

PARI's generic solver is the best or tied-best single method in **every** family, because it already composes
Pohlig–Hellman with index calculus internally — exactly the "recognise the structure, use the matching method"
logic an agent would add. Choosing the method per instance in hindsight saves 10% overall and nothing within any
family.

The one genuine gap is **interval-restricted** discrete log: PARI ignores the interval, so a kangaroo can win
(e.g. `ec_interval_44_0`: kangaroo 0.7 s vs PARI 2.0 s vs BSGS 11.1 s). But that is a single specialised algorithm,
not an adaptive combination, and realising it at larger interval sizes needs a fast (C) implementation rather than
an orchestration layer — the pure-Python kangaroo here is too slow on the 2^44–2^48 F_p* intervals.

## Verdict

Same as LP, QP and primality: mature libraries (PARI, FLINT, HiGHS) already do the method composition inside, so an
external agent adds ×1.0–1.6 at most. Agentic combination pays where **no** mature integrated solver exists — which
is the index-coding / `gap-agent` setting, not these.
