#!/bin/bash
# After the text runs finish: ViT CIFAR-100 joint training, then the DeepSloth attack inputs (GPU 3).
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
cd ~/ie624
while pgrep -f scripts/train_text.sh > /dev/null; do sleep 30; done
~/venv/bin/python traces/train_exits.py configs/train_cifar100.yaml > logs/train_cifar100.log 2>&1 && \
~/venv/bin/python attacks/deepsloth_vit.py --n 500 > logs/deepsloth.log 2>&1
