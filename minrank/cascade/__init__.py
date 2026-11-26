"""Option D: cost-aware early exit for evaluating candidates in LLM evolutionary search.

traces      trace contract, loading, batching, synthetic generator
calibration P(win | partial gap, checkpoint) table
policies    full, OpenEvolve cascade, successive halving, Hyperband, Hoeffding race, cost rule, certified cost rule
simulator   causal trace replay with cost accounting and lost-winner counting
online      CascadeGate for the live evaluator
run_replay  CLI
"""
