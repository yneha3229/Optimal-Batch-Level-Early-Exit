# Preliminary results: independent-set symmetry breaking (23 Nov 2026)

Scratch runs from before the method was moved into `minrank/exact/` (see `WORK.md`, task C1). Kept for reference;
the repo runs (`exact_indep.csv`, `families.csv`) supersede them.

- `indep_fix_scratch.py`: the scratch encoder (base encoding of `minrank/exact/sat.py` without u_0 = e_1, plus
  U_I = V_I = [Id | 0] for an independent set I from `lower.mais`; directed: U_I = [Id | 0], V_I[:, s:] = 0).
- `sweep.log`: named families, ascending from alpha to the clique cover, 900 s per call, CaDiCaL 1.5.3 (pysat).
  One dict per graph; `log` lists (r, status, seconds).
- `open_run.log`: the benchmark graphs not proven by `exact.csv`, one call at (best known rank - 1), 1800 s.
  `sat` with `found_rank` improves the upper bound; `unsat` proves the best known rank optimal.
- `k83_r21.log`: the first K(8,3) rank-21 UNSAT run (822 s).
