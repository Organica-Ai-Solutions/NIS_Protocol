#!/bin/bash
# Launch diverse training workloads on H100

echo "🚀 Launching diverse training batch..."

# GPU 1 - PINN with larger network
CUDA_VISIBLE_DEVICES=1 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu1_pinn_large.log 2>&1 &

# GPU 2 - PINN different equation
CUDA_VISIBLE_DEVICES=2 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu2_pinn_wave.log 2>&1 &

# GPU 3 - PINN high batch
CUDA_VISIBLE_DEVICES=3 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu3_pinn_batch.log 2>&1 &

# GPU 4 - Transformer
CUDA_VISIBLE_DEVICES=4 nohup python3 ~/organica-ai/training/train_transformer.py > ~/organica-ai/logs/gpu4_transformer.log 2>&1 &

# GPU 5-7 - More PINN variants
CUDA_VISIBLE_DEVICES=5 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu5_pinn.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu6_pinn.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu7_pinn.log 2>&1 &

sleep 5
echo "✅ All 7 training jobs launched!"
