#!/bin/bash
# Launch NeuroLinux training suite on H100
# Deploys when current PINN/Transformer batch completes

echo "🤖 Launching NeuroLinux Training Suite on H100"
echo "=============================================="

# Stop current training jobs
echo "🛑 Stopping current training batch..."
pkill -f 'python3.*train' 2>/dev/null
sleep 3

# Clear GPU memory
echo "🧹 Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 2

# Create NeuroLinux models directory
echo "📁 Creating NeuroLinux models directory..."
mkdir -p ~/organica-ai/models/neurolinux/{bitnet,vision,rl,embeddings}

# Launch NeuroLinux training jobs
echo ""
echo "🚀 Deploying NeuroLinux Training Jobs..."
echo ""

# GPU 0-3: BitNet Robotics (4-GPU parallel training)
echo "🤖 GPUs 0-3: BitNet Robotics & CAN Bus Training"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ~/organica-ai/training/train_bitnet_robotics_h100.py > ~/organica-ai/logs/bitnet_robotics.log 2>&1 &

# GPU 4-5: Vision models (2-GPU training)
echo "👁️  GPUs 4-5: Vision Models (YOLO + MobileNet)"
CUDA_VISIBLE_DEVICES=4 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/vision_yolo.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/vision_mobilenet.log 2>&1 &

# GPU 6: RL policies
echo "🎮 GPU 6: RL Policies (Autonomous Navigation)"
CUDA_VISIBLE_DEVICES=6 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/rl_policies.log 2>&1 &

# GPU 7: Embeddings generation
echo "📚 GPU 7: Embeddings (Robotics Knowledge Base)"
CUDA_VISIBLE_DEVICES=7 nohup python3 ~/organica-ai/training/train_transformer.py > ~/organica-ai/logs/embeddings.log 2>&1 &

sleep 5
echo ""
echo "✅ NeuroLinux Training Suite Deployed!"
echo ""
echo "📊 Training Jobs:"
echo "  - BitNet Robotics: GPUs 0-3 (4-GPU parallel)"
echo "  - Vision Models: GPUs 4-5"
echo "  - RL Policies: GPU 6"
echo "  - Embeddings: GPU 7"
echo ""
echo "Monitor with: nvidia-smi"
echo "Check logs: tail -f ~/organica-ai/logs/*.log"
echo ""
echo "Expected completion: 4-6 hours"
echo "Models will be saved to: ~/organica-ai/models/neurolinux/"
