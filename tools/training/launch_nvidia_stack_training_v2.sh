#!/bin/bash
# Launch NVIDIA Stack Training with venv

echo "🚀 Launching NVIDIA Stack Training"
pkill -f 'python.*train' 2>/dev/null
sleep 3

mkdir -p ~/organica-ai/models/neurolinux/{nemo,isaac}

# Use venv Python
PYTHON=~/organica-ai/venv/bin/python3

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup $PYTHON ~/organica-ai/training/train_nemo_asr_h100.py > ~/organica-ai/logs/nemo_asr.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON ~/organica-ai/training/train_isaac_rl_h100.py > ~/organica-ai/logs/isaac_nav.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON ~/organica-ai/training/train_isaac_rl_h100.py > ~/organica-ai/logs/isaac_avoid.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/pinn.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON ~/organica-ai/training/train_transformer.py > ~/organica-ai/logs/transformer.log 2>&1 &

sleep 3
echo "✅ Deployed!"
nvidia-smi
