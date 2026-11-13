# When Should a Batch Stop? Cost-Aware Batch-Level Early Exit

IE624 course project (IEOR, IIT Bombay), instructor Prof. Manjesh K. Hanawal. Team of three.

**Problem.** Under batched GPU inference, early-exit savings shrink because a batch runs until its hardest input
finishes, and each layer costs `c(n) = a + b·n` for `n` active inputs. Which exit rule is fastest depends on the batch
size, the execution mode (eager or compiled) and the task.

**Current direction.** One exit rule that takes the measured cost `(a, b, head, n)` as input, tuned once, and stays
near the best method across deployments. See `docs/DIRECTION.md`.

## Where things stand (2026-11-13)

| Item | State | Read |
|---|---|---|
| Direction and test plan | Current | `docs/DIRECTION.md` |
| Results up to Nov 6 (eager PyTorch, simulator and real GPU) | Done, with corrections noted at the top | `docs/RESULTS.md` |
| CUDA-graph timings, headroom bound, hidden-state probe, LayerSkip exits (Nov 12) | Done | `docs/PLAN_PAPER.md` §7 |
| Theory: rule, regime switch, robustness bound | Done | `theory/THEORY.md` |
| Transfer test (step 1 of the direction) | Running | `experiments/transfer_test.py` |
| Exit plan (which exits to evaluate), load-trace experiment | Not started | `docs/DIRECTION.md` §5 |

Findings that hold so far:
- Wait-for-all early exit is slower than no early exit on a real GPU (0.78–0.97×).
- Per-sample exits fall below 1× at small batches; exit-head overhead is the main cause.
- No rule is fastest everywhere: unified exit wins at B ≤ 16 on text, per-sample exits on CIFAR-100, the theory rule
  or a tuned random search at B ≥ 64.
- The label-free bandit learns the stopping rule with sublinear regret (log-log slopes 0.49–0.65, single runs of
  3 seeds).
- The trimmed rule's pathwise bound had 0 violations in 10,500 batches.

Negative results: frozen-backbone and LayerSkip decoder exits are too weak, straggler rescue does not help,
trajectory-aware calibration does not help, FunSearch ties and overfits, hidden-state probes add nothing over
confidence.

## Layout

| Dir | Contents |
|---|---|
| `docs/` | direction, plans, results, literature map, profiling notes |
| `theory/` | optimal-stopping formulation, regime switch, robustness theorem |
| `simulator/` | trace-replay simulator, policies (baselines, theory rule, robust rule), exit model |
| `traces/` | exit training, trace logging, hidden-state extraction; `*_val.parquet` traces |
| `runner/` | real-GPU runners: eager (`gpu_runner.py`) and CUDA graphs (`graph_runner.py`) |
| `decoder/` | decoder exit heads, batched decode runner, LayerSkip exit check |
| `bandit/` | label-free partial-feedback bandit and regret harness |
| `evolve/` | OpenEvolve evaluator, seed program, random-search and BO baselines |
| `attacks/` | DeepSloth slowdown attack on the ViT |
| `experiments/` | main sweep, robustness checks, transfer test |
| `analysis/` | figures, headroom bound, correctness probe |
| `profiling/` | per-layer timing and the fit of `a`, `b` |
| `results/`, `figures/` | CSV/JSON outputs and plots |
| `scripts/` | launch scripts for the full runs |
| `tests/` | unit tests |

Not in the repository: model checkpoints, datasets, hidden-state dumps and smoke-test outputs (see `.gitignore`).
Checkpoints are rebuilt by `traces/train_exits.py`.

## Setup and conventions

- Python 3.12 environment at `~/venv` (torch 2.14, transformers 5.17).
- GPUs: RTX A6000. Launch with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>` and check the GPU is idle
  first; timing runs need an idle GPU.
- Policies are selected on the evolve split and reported on the held-out split. Policies never see labels at
  decision time.
- Headline numbers are wall-clock on the real GPU; simulated numbers are labelled as simulated.

## Reproduce

```
# exits and traces
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python traces/train_exits.py configs/train_mnli.yaml   # also qqp, cifar100
# simulation sweep, all methods
~/venv/bin/python experiments/run_main.py --task mnli
# real GPU, eager and CUDA graphs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python runner/gpu_runner.py --task mnli --calibrate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python runner/gpu_runner.py --task mnli
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python runner/graph_runner.py --task mnli
# bandit, headroom bound, probe, transfer test
~/venv/bin/python bandit/run_bandit.py --task mnli --regime dec_eager --B 64
~/venv/bin/python analysis/headroom.py
~/venv/bin/python analysis/probe_correctness.py mnli qqp cifar100
~/venv/bin/python experiments/transfer_test.py
# tests
~/venv/bin/python -m pytest tests simulator/test_simulator.py -q
```
