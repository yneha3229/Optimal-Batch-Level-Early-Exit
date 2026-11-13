#!/bin/bash
# Joint exit training for the text tasks, sequentially on GPU 3.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
cd ~/ie624
for t in mnli qqp; do
  ~/venv/bin/python traces/train_exits.py configs/train_$t.yaml > logs/train_$t.log 2>&1
done
