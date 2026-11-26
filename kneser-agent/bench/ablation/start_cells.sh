#!/usr/bin/env bash
# Start the pipelines (chat backend) for one blind-agent configuration on the 9 new cells; prints the folders.
CFG=$1
cd "$(dirname "$0")"
for c in "K(16,4) 1" "K(16,4)-del0.01 1" "K(14,4) 1" "K(13,4) 0" "K(13,4) 1" "K(15,5) 0" "K(15,5) 1" "reg(1000,6) 0" "reg(1000,6) 1"; do
  set -- $c
  D="$PWD/chat/${CFG}_$(printf %s "$1" | tr -c 'A-Za-z0-9' _)s$2"
  PATH=$HOME/venv/bin:$PATH HF_HUB_OFFLINE=1 setsid nohup taskset -c 30-31 bash run.sh "$CFG" "$1" "$2" >/dev/null 2>&1 < /dev/null &
  echo "$D"
done
