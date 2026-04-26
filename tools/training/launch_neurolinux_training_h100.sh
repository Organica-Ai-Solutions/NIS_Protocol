#!/bin/bash
# Launch NeuroLinux Training Suite on H100
# BitNet Robotics + Vision + RL + Embeddings

echo "🤖 Launching NeuroLinux Training Suite"
echo "========================================"

# Stop current training
pkill -f 'python.*train' 2>/dev/null
sleep 3

# Create model directories
mkdir -p ~/organica-ai/models/neurolinux/{bitnet,vision,rl,embeddings}

# Use venv Python
PYTHON=~/organica-ai/venv/bin/python3

echo ""
echo "🚀 Deploying NeuroLinux Training..."
echo ""

# GPUs 0-3: BitNet Robotics (4-GPU parallel)
echo "🤖 GPUs 0-3: BitNet Robotics + CAN Bus Training"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup $PYTHON ~/organica-ai/training/train_bitnet_robotics_h100.py > ~/organica-ai/logs/bitnet_robotics.log 2>&1 &

# GPU 4: Vision model training
echo "👁️  GPU 4: Vision Model Training"
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON ~/organica-ai/training/train_vision_h100.py > ~/organica-ai/logs/vision.log 2>&1 &

# GPU 5: RL policy training
echo "🎮 GPU 5: RL Policy Training"
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON ~/organica-ai/training/train_rl_h100.py > ~/organica-ai/logs/rl.log 2>&1 &

# GPU 6: Embeddings training
echo "🧬 GPU 6: Embeddings Training"
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON ~/organica-ai/training/train_embeddings_h100.py > ~/organica-ai/logs/embeddings.log 2>&1 &

# GPU 7: Transformer baseline
echo "🧠 GPU 7: Transformer Baseline"
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON ~/organica-ai/training/train_transformer.py > ~/organica-ai/logs/transformer.log 2>&1 &

sleep 5
echo ""
echo "✅ NeuroLinux Training Deployed!"
echo ""
echo "📊 Training Jobs:"
echo "  - BitNet Robotics: GPUs 0-3 (4-GPU parallel)"
echo "  - Vision Model: GPU 4"
echo "  - RL Policy: GPU 5"
echo "  - Embeddings: GPU 6"
echo "  - Transformer: GPU 7 (baseline)"
echo ""
echo "Monitor: nvidia-smi"
echo "Logs: tail -f ~/organica-ai/logs/*.log"
echo ""
echo "Expected: 6-8 hours"
echo "Models: ~/organica-ai/models/neurolinux/"
echo ""
echo "🎯 NeuroLinux training in progress!"
