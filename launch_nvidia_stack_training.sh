#!/bin/bash
# Launch NVIDIA Stack Training on H100
# Phase B: NeMo ASR + Phase C: Isaac Lab RL

echo "🚀 Launching NVIDIA Stack Training Suite"
echo "=========================================="

# Stop current training jobs
echo "🛑 Stopping current training batch..."
pkill -f 'python3.*train' 2>/dev/null
sleep 3

# Clear GPU memory
echo "🧹 Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 2

# Create model directories
echo "📁 Creating NVIDIA model directories..."
mkdir -p ~/organica-ai/models/neurolinux/{nemo,isaac}

# Launch NVIDIA training jobs
echo ""
echo "🚀 Deploying NVIDIA Stack Training..."
echo ""

# GPUs 0-3: NeMo ASR Training (4-GPU data parallel)
echo "🎤 GPUs 0-3: NeMo ASR for NeuroLinux Voice Commands"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ~/organica-ai/training/train_nemo_asr_h100.py > ~/organica-ai/logs/nemo_asr.log 2>&1 &

# GPUs 4-5: Isaac Lab RL Training (2-GPU)
echo "🤖 GPUs 4-5: Isaac Lab Navigation + Obstacle Avoidance"
CUDA_VISIBLE_DEVICES=4 nohup python3 ~/organica-ai/training/train_isaac_rl_h100.py > ~/organica-ai/logs/isaac_navigation.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 ~/organica-ai/training/train_isaac_rl_h100.py > ~/organica-ai/logs/isaac_avoidance.log 2>&1 &

# GPU 6: Continue PINN training
echo "⚛️  GPU 6: PINN Physics Training (baseline)"
CUDA_VISIBLE_DEVICES=6 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/pinn_baseline.log 2>&1 &

# GPU 7: Continue Transformer training
echo "🧠 GPU 7: Transformer Training (baseline)"
CUDA_VISIBLE_DEVICES=7 nohup python3 ~/organica-ai/training/train_transformer.py > ~/organica-ai/logs/transformer_baseline.log 2>&1 &

sleep 5
echo ""
echo "✅ NVIDIA Stack Training Deployed!"
echo ""
echo "📊 Training Jobs:"
echo "  - NeMo ASR: GPUs 0-3 (4-GPU parallel)"
echo "  - Isaac Lab RL: GPUs 4-5 (navigation + avoidance)"
echo "  - PINN: GPU 6 (baseline)"
echo "  - Transformer: GPU 7 (baseline)"
echo ""
echo "Monitor with: nvidia-smi"
echo "Check logs: tail -f ~/organica-ai/logs/*.log"
echo ""
echo "Expected completion: 6-8 hours"
echo "Models will be saved to: ~/organica-ai/models/neurolinux/"
echo ""
echo "🎯 Phase B (NeMo ASR) + Phase C (Isaac Lab RL) in progress!"
