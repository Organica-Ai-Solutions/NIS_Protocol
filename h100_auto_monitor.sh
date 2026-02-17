#!/bin/bash
# H100 Auto-Monitor and Restart Script
# Keeps all 8 GPUs training 24/7 while user is at work

LOG_DIR="$HOME/organica-ai/logs"
TRAINING_DIR="$HOME/organica-ai/training"
VENV_PYTHON="$HOME/organica-ai/venv/bin/python3"

# Check each GPU and restart if idle
for GPU in {0..7}; do
    # Check if training process exists for this GPU
    PROCESS_COUNT=$(ps aux | grep "CUDA_VISIBLE_DEVICES=$GPU" | grep "python.*train" | grep -v grep | wc -l)
    
    if [ $PROCESS_COUNT -eq 0 ]; then
        echo "[$(date)] GPU $GPU idle - restarting training"
        cd $TRAINING_DIR
        CUDA_VISIBLE_DEVICES=$GPU nohup $VENV_PYTHON train_pinn_simple.py > $LOG_DIR/gpu${GPU}_auto_$(date +%Y%m%d_%H%M).log 2>&1 &
        sleep 2
    fi
done

# Report status
echo "[$(date)] Auto-monitor check complete"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
