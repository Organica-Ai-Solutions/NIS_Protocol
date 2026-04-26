#!/bin/bash
# Launch 8 parallel training jobs on H100

echo "🚀 Launching 8 parallel training jobs..."

# GPU 0 - Already running PINN Heat
echo "✅ GPU 0: PINN Heat (already running)"

# GPU 1 - PINN Wave
echo "🔥 GPU 1: PINN Wave"
CUDA_VISIBLE_DEVICES=1 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu1_wave.log 2>&1 &

# GPU 2 - PINN Laplace
echo "🔥 GPU 2: PINN Laplace"
CUDA_VISIBLE_DEVICES=2 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu2_laplace.log 2>&1 &

# GPU 3 - PINN Burgers
echo "🔥 GPU 3: PINN Burgers"
CUDA_VISIBLE_DEVICES=3 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu3_burgers.log 2>&1 &

# GPU 4 - PINN Navier-Stokes
echo "🔥 GPU 4: PINN Navier-Stokes"
CUDA_VISIBLE_DEVICES=4 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu4_ns.log 2>&1 &

# GPU 5 - PINN Schrodinger
echo "🔥 GPU 5: PINN Schrodinger"
CUDA_VISIBLE_DEVICES=5 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu5_schrodinger.log 2>&1 &

# GPU 6 - PINN Maxwell
echo "🔥 GPU 6: PINN Maxwell"
CUDA_VISIBLE_DEVICES=6 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu6_maxwell.log 2>&1 &

# GPU 7 - PINN Diffusion
echo "🔥 GPU 7: PINN Diffusion"
CUDA_VISIBLE_DEVICES=7 nohup python3 ~/organica-ai/training/train_pinn_simple.py > ~/organica-ai/logs/gpu7_diffusion.log 2>&1 &

sleep 5
echo ""
echo "✅ All 8 training jobs launched!"
echo ""
echo "Monitor with: nvidia-smi"
echo "Check logs: tail -f ~/organica-ai/logs/*.log"
