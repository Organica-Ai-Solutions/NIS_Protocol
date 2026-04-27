#!/bin/bash
# Phase 4 HEAVY: Real-Quality 500K-Step Training on 5 GPUs
# Uses real data (xArm, Aloha, PushT, CIFAR-100, LibriSpeech, BeaverTails)
# + large models that fill H100 VRAM (30-50GB each)
#
# GPU 0: Robotics CLIP (ViT-L/14 + 12L text transformer)
# GPU 1: VLA Heavy (ViT-B + 24L GPT action decoder)
# GPU 3: Sim2Real (ResNet-50 + DANN + contrastive)
# GPU 4: Safety Classifier (16L ViT + action + text fusion)
# GPU 6: Speech-to-Action (24L Whisper encoder + 12L decoder)
#
# Skip: GPU 2 (Cosmos Reason), GPU 5 (NIS-MoE 778K/1M), GPU 7 (Predict 2.5)

set -e
SCRIPT_DIR="/data/organica-ai/training/heavy"
VENV="/data/organica-ai/venv/bin/python3"
LOG="/data/organica-ai/logs"
DATE=$(date +%Y%m%d_%H%M)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  HEAVY Phase 4: Real Data + Large Models — 5 GPUs × ~48h   ║"
echo "║  Total: ~240 GPU-hours | Real xArm/Aloha/PushT/CIFAR/etc   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "[1/5] GPU 0 → Robotics CLIP Heavy (ViT-L/14, ~40GB VRAM)..."
tmux new-session -d -s heavy_clip "CUDA_VISIBLE_DEVICES=0 $VENV $SCRIPT_DIR/train_clip_heavy.py 2>&1 | tee $LOG/clip_heavy_gpu0_${DATE}.log"

echo "[2/5] GPU 1 → VLA Heavy (ViT-B + GPT-24L, ~50GB VRAM)..."
tmux new-session -d -s heavy_vla "CUDA_VISIBLE_DEVICES=1 $VENV $SCRIPT_DIR/train_vla_heavy.py 2>&1 | tee $LOG/vla_heavy_gpu1_${DATE}.log"

echo "[3/5] GPU 3 → Sim2Real Heavy (ResNet-50 + DANN, ~40GB VRAM)..."
tmux new-session -d -s heavy_sim2real "CUDA_VISIBLE_DEVICES=3 $VENV $SCRIPT_DIR/train_sim2real_heavy.py 2>&1 | tee $LOG/sim2real_heavy_gpu3_${DATE}.log"

echo "[4/5] GPU 4 → Safety Heavy (16L ViT + fusion, ~35GB VRAM)..."
tmux new-session -d -s heavy_safety "CUDA_VISIBLE_DEVICES=4 $VENV $SCRIPT_DIR/train_safety_heavy.py 2>&1 | tee $LOG/safety_heavy_gpu4_${DATE}.log"

echo "[5/5] GPU 6 → Speech2Action Heavy (Whisper-24L, ~40GB VRAM)..."
tmux new-session -d -s heavy_speech "CUDA_VISIBLE_DEVICES=6 $VENV $SCRIPT_DIR/train_speech2action_heavy.py 2>&1 | tee $LOG/speech_heavy_gpu6_${DATE}.log"

echo ""
echo "✅ All 5 heavy jobs launched!"
echo ""
echo "Monitor:"
echo "  tmux ls"
echo "  nvidia-smi"
echo "  tmux attach -t heavy_clip"
echo "  tail -f $LOG/clip_heavy_gpu0_${DATE}.log"
