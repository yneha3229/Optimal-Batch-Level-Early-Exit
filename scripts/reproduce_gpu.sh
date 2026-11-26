#!/usr/bin/env bash
# GPU steps of the early-exit report, in order, with the GPU, a time estimate and the expected output of each
# (docs/REPRODUCE_EARLY_EXIT.md, section 4). The CPU steps that follow each GPU step are in scripts/reproduce_cpu.sh.
#
# By default this script only PRINTS the plan. To run steps:  RUN=1 GPU_ID=<id> STEPS="g5 g6" scripts/reproduce_gpu.sh
# Outputs of a run go to results_repro/gpu/ (a shadow copy, as in reproduce_cpu.sh); results/ is never written.
# Times are wall-clock from the status logs of the original runs (logs/*_status.log); "est." = not logged.
# Timing steps need a quiet CPU (no CPU sweeps on the same machine); otherwise the speedups are not comparable.
# GPU index: CUDA_DEVICE_ORDER=PCI_BUS_ID (the order of nvidia-smi). Check `nvidia-smi` for an idle A6000 first.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/venv/bin/python}
RUN=${RUN:-0}; STEPS=${STEPS:-all}; GPU_ID=${GPU_ID:-}
SH=$REPO/results_repro/gpu
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

plan() {   # id | GPU | time | output | command
cat <<'EOF'
g1|A6000 (48 GB)|not logged (est. 1-2 h per task)|checkpoints/{mnli,qqp}.pt, traces/{mnli,qqp}_val.parquet, results/train_{mnli,qqp}_exit_accuracy.csv (Table 1: 87.8%, 91.1%)|traces/train_exits.py configs/train_mnli.yaml; traces/train_exits.py configs/train_qqp.yaml   (= scripts/train_text.sh)
g2|A6000|not logged (est. 2-3 h)|checkpoints/cifar100.pt, traces/cifar100_val.parquet, results/train_cifar100_exit_accuracy.csv (Table 1: 93.3%); traces/cifar100_attack.parquet (DeepSloth)|traces/train_exits.py configs/train_cifar100.yaml; attacks/deepsloth_vit.py --n 500   (= scripts/train_vision.sh)
g3|A6000|est. 3-4 h in total|results/runner_cost_{mnli,qqp,cifar100}.json, runner_cost_graph_*.json, main_*_real_*_gpu*.csv, graph_*_gpu.csv (Figure 1, Table 2, sections 4.2, 4.7)|runner/gpu_runner.py --task <t> --calibrate; runner/gpu_runner.py --task <t> --regime real_<t> --batch-sizes 8 16 64 128; runner/graph_runner.py --task <t> (scripts/run_final.sh, run_parallel.sh, run_gpu_redo.sh; logs/step2_graph_<t>.log)
g4|GTX 1650 (4 GB, laptop)|profile est. 5 min per length; timing 63 min (logs/local_status.log)|results/local_cost_cuda_T{16,32,128}.json (Table 3), local_time_{qqp,mnli}_cuda_T{16,32,128}.csv (Tables 4-6, Figure 3)|experiments/local_hw.py profile --seq-len T; then scripts/run_local.sh (select and resplit are CPU)
g5|A6000|28 min (logs/server_a6000_status.log, 22 Nov)|results/local_cost_a6000_T{128,32}.json, local_time_{qqp,mnli}_a6000_T{128,32}.csv (Tables 11-12)|scripts/run_server_a6000.sh
g6|A6000 (not GPU 5: it failed twice on 24-25 Nov)|est. 40-60 min; NEVER RUN|results/local_cost_a6000_graph_T{128,32}.json, local_time_{qqp,mnli}_a6000_graph_T{128,32}.csv (5.1, CUDA graphs)|experiments/check_graph_device.py --task qqp --name a6000 --seq-len 128 first; then scripts/run_server_a6000_graph.sh
g7|A6000|eager est. 15-20 min (B = 8 took 3.5 min on 25 Nov); graph est. 20-30 min|results/local_time_cifar100_a6000_T197.csv (5.1, ViT; only B = 8 exists), local_time_cifar100_a6000_graph_T197.csv|scripts/run_server_a6000_vit.sh
g8|A6000|Qwen: train est. 1-2 h per task, time 17-21 min per task; Llama: profile 6 min, train 81 min (QQP), time 60-63 min per task (logs/llm_cls_status.log)|checkpoints/{qqp,mnli}_{qwen,llama}_cls.pt, traces/{qqp,mnli}_{qwen,llama}_val.parquet, results/local_cost_a6000_T128_{qwen,llama}.json, local_time_{qqp,mnli}_{qwen,llama}_a6000_T128.csv, local_check_*.json (Tables 15-16)|decoder/train_cls_exit_lora.py --task <t> --model Qwen/Qwen2.5-1.5B --tag qwen; ... --model meta-llama/Llama-3.1-8B-Instruct --tag llama --exits 8 12 16 20 24 28 32 --steps 8000 --bs 16; experiments/local_hw.py profile --task <t>_<tag> --name a6000; scripts/run_llm_cls_hw.sh <tag> mnli qqp (= scripts/run_llama_cls_queue.sh for Llama)
g9|A6000|est. 4 GPU-hours (docs/PLAN_LLM.md runbook)|traces/wikitext_{qlora,qfrozen}_val.parquet, results/train_wikitext_{qlora,qfrozen}_exit_accuracy.csv (5.7 decoding)|decoder/train_exit_lora.py --tag qlora --rank 16 --steps 6000; decoder/train_exit_lora.py --tag qfrozen --rank 0 --steps 6000
g10|A6000|est. 1 h per model|results/layerskip_{ls1b,ls8b}_layer_accuracy.csv (4.9), train_wikitext_llama_exit_accuracy.csv (Table 1, 53.6%)|decoder/layerskip_exits.py --model facebook/layerskip-llama3.2-1B / facebook/layerskip-llama3-8B; decoder/train_decoder_exits.py --model meta-llama/Llama-3.1-8B-Instruct --exits 12 16 20 24 28 32 --tag llama --bs 4
EOF
}

echo "GPU steps (id | GPU | time | expected output | command):"
plan | while IFS='|' read -r id gpu t out cmd; do
  printf "\n[%s] GPU: %s\n     time: %s\n     output: %s\n     command: %s\n" "$id" "$gpu" "$t" "$out" "$cmd"
done
[ "$RUN" = 1 ] || { echo; echo "Plan only. To run: RUN=1 GPU_ID=<id> STEPS=\"g5\" $0   (g5, g6, g7 are scripted; the others are listed for reference)"; exit 0; }

: "${GPU_ID:?set GPU_ID}"
if [ ! -d "$SH" ]; then           # shadow copy: the run scripts write to <root>/results and <root>/logs
  mkdir -p "$SH"
  for d in analysis attacks bandit configs decoder experiments runner scripts simulator; do cp -r "$REPO/$d" "$SH/$d"; done
  ln -s "$REPO/traces" "$SH/traces"; [ -d "$REPO/checkpoints" ] && ln -s "$REPO/checkpoints" "$SH/checkpoints"
  mkdir -p "$SH/logs" && cp -r "$REPO/results" "$SH/results"
fi
for s in $STEPS; do case $s in
  g5) ( cd "$SH" && rm -f results/local_cost_a6000_T{128,32}.json && GPU_ID=$GPU_ID scripts/run_server_a6000.sh ) ;;
  g6) ( cd "$SH" && CUDA_VISIBLE_DEVICES=$GPU_ID $PY experiments/check_graph_device.py --task qqp --name a6000 --seq-len 128 > logs/check_graph_device.log 2>&1 \
          && GPU_ID=$GPU_ID scripts/run_server_a6000_graph.sh ) ;;
  g7) ( cd "$SH" && rm -f results/local_cost_a6000_T197_vit.json && GPU_ID=$GPU_ID scripts/run_server_a6000_vit.sh ) ;;
  *) echo "step $s is not scripted here; run its command from the plan above inside $SH" ;;
esac; done
echo "outputs in $SH/results; compare with: $PY analysis/diff_repro.py --repro $SH/results --ref $REPO/results --files <group:file ...> --out ... --md ..."
