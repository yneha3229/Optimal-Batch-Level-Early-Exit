"""The existing LP methods, each as  solve(lp, tol, time_limit) -> (x, y, status, seconds),  one thread each.

  dual_simplex     HiGHS dual simplex
  primal_simplex   HiGHS primal simplex
  ipm              HiGHS interior point (IPX), no crossover, tolerances = tol
  ipm_crossover    HiGHS interior point + crossover to a vertex
  pdlp_highs       HiGHS PDLP (first-order primal-dual hybrid gradient)
  pdlp_ortools     Google OR-Tools PDLP (restarted, adaptive steps, feasibility polishing)
"""
from __future__ import annotations

import time

import numpy as np

from .problem import LP

INF = 1e30


def _highs(lp: LP, options: dict, time_limit: float):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(max(0.01, time_limit)))
    for k, v in options.items():
        h.setOptionValue(k, v)
    model = highspy.HighsLp()
    m, n = lp.shape
    model.num_col_, model.num_row_ = n, m
    model.col_cost_ = lp.c
    model.offset_ = lp.c0
    model.col_lower_ = np.where(np.isfinite(lp.cl), lp.cl, -INF)
    model.col_upper_ = np.where(np.isfinite(lp.cu), lp.cu, INF)
    model.row_lower_ = np.where(np.isfinite(lp.rl), lp.rl, -INF)
    model.row_upper_ = np.where(np.isfinite(lp.ru), lp.ru, INF)
    A = lp.A.tocsc()
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = A.indptr
    model.a_matrix_.index_ = A.indices
    model.a_matrix_.value_ = A.data
    model.a_matrix_.num_col_, model.a_matrix_.num_row_ = n, m
    h.passModel(model)
    t = time.perf_counter()
    h.run()
    secs = time.perf_counter() - t
    sol = h.getSolution()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(sol.col_value) if sol.value_valid else None
    y = np.asarray(sol.row_dual) if sol.dual_valid else None
    return x, y, status, secs


def dual_simplex(lp, tol, tl):
    return _highs(lp, {"solver": "simplex", "simplex_strategy": 1}, tl)


def primal_simplex(lp, tol, tl):
    return _highs(lp, {"solver": "simplex", "simplex_strategy": 4}, tl)


def ipm(lp, tol, tl):
    return _highs(lp, {"solver": "ipm", "run_crossover": "off", "ipm_optimality_tolerance": tol,
                       "primal_feasibility_tolerance": tol, "dual_feasibility_tolerance": tol}, tl)


def ipm_crossover(lp, tol, tl):
    return _highs(lp, {"solver": "ipm", "run_crossover": "on"}, tl)


def pdlp_highs(lp, tol, tl):
    return _highs(lp, {"solver": "pdlp", "primal_feasibility_tolerance": tol, "dual_feasibility_tolerance": tol,
                       "pdlp_d_gap_tol": tol}, tl)


def pdlp_ortools(lp, tol, tl):
    from ortools.pdlp import solvers_pb2
    from ortools.pdlp.python import pdlp

    qp = pdlp.QuadraticProgram()
    m, n = lp.shape
    qp.objective_vector = lp.c
    qp.objective_offset = lp.c0
    qp.constraint_matrix = lp.A.tocsc()
    qp.constraint_lower_bounds = lp.rl
    qp.constraint_upper_bounds = lp.ru
    qp.variable_lower_bounds = lp.cl
    qp.variable_upper_bounds = lp.cu
    p = solvers_pb2.PrimalDualHybridGradientParams()
    p.num_threads = 1
    p.termination_criteria.simple_optimality_criteria.eps_optimal_relative = tol
    p.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = tol * 1e-3
    p.termination_criteria.time_sec_limit = float(max(0.01, tl))
    p.use_feasibility_polishing = True
    p.handle_some_primal_gradients_on_finite_bounds_as_residuals = False
    p.presolve_options.use_glop = False
    t = time.perf_counter()
    r = pdlp.primal_dual_hybrid_gradient(qp, p)
    secs = time.perf_counter() - t
    from ortools.pdlp import solve_log_pb2

    reason = solve_log_pb2.TerminationReason.Name(r.solve_log.termination_reason)
    return np.asarray(r.primal_solution), np.asarray(r.dual_solution), reason, secs


METHODS = {"dual_simplex": dual_simplex, "primal_simplex": primal_simplex, "ipm": ipm,
           "ipm_crossover": ipm_crossover, "pdlp_highs": pdlp_highs, "pdlp_ortools": pdlp_ortools}
