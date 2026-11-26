#!/usr/bin/env bash
# Remaining benchmark steps of 26 Nov, chained (waits by PID; kills only our own vLLM by its PID file).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
PY=~/venv/bin/python
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 15; done; }
( waitpid 42410; taskset -c 28-31 $PY bench/run_agent.py --llm none --budget 300 --specs "K(20,4)" "K(17,5)" "K(23,4)"; echo NONE_LARGE_DONE ) >> ../logs/kneser_agent_run_none_large.log 2>&1 &
( waitpid 40128; taskset -c 16-27 $PY bench/baselines.py run --limit 120 --par 3 "K(20,4)" "K(17,5)" "K(23,4)"; echo BASE_LARGE_DONE ) >> ../logs/kneser_agent_baselines_large.log 2>&1 &
waitpid 42409
taskset -c 32-35 $PY bench/run_agent.py --llm local --budget 300 --specs "K(20,4)" "K(17,5)" "K(23,4)" >> ../logs/kneser_agent_run_qwen32b_large.log 2>&1
echo "32B done $(date -u)"
kill "$(cat ../logs/minrank_vllm_8000.pid)"; sleep 20
cd .. && GPU=4 PORT=8001 MODEL=Qwen/Qwen3-4B-Instruct-2507 PY=$HOME/venv-vllm/bin/python LOG=logs/kneser_agent_vllm_4b.log bash scripts/minrank/05_vllm_serve.sh
cd kneser-agent
taskset -c 32-35 $PY bench/run_agent.py --llm "local:http://127.0.0.1:8001/v1|Qwen/Qwen3-4B-Instruct-2507" --budget 300 >> ../logs/kneser_agent_run_qwen4b.log 2>&1
echo "4B done $(date -u)"
wait
echo CHAIN_DONE
