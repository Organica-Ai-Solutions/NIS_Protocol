#!/bin/bash
# H100 Auto-Monitor — Credit-Smart (DO NOT waste 80GB GPUs on tiny models)
# See docs/H100_CREDIT_AUDIT.md
#
# CREDIT RULE: Never restart with train_pinn_simple (2GB) — wastes credits.
# Instead: log idle GPUs and alert. Human launches launch_heavy.sh or heavy jobs.

LOG_DIR="${HOME}/organica-ai/logs"
HEAVY_DIR="${HOME}/organica-ai/scripts/h100_heavy"

# Check each GPU — report idle, DO NOT auto-launch tiny models
idle_gpus=()
for GPU in {0..7}; do
    PROCESS_COUNT=$(ps aux | grep "CUDA_VISIBLE_DEVICES=$GPU" | grep -E "python.*train|torchrun|accelerate" | grep -v grep | wc -l)
    if [ "$PROCESS_COUNT" -eq 0 ]; then
        idle_gpus+=($GPU)
    fi
done

if [ ${#idle_gpus[@]} -gt 0 ]; then
    echo "[$(date)] IDLE GPUs: ${idle_gpus[*]} — run launch_heavy.sh or queue a large job"
    echo "  cd ~/organica-ai && bash scripts/h100_heavy/launch_heavy.sh"
else
    echo "[$(date)] All 8 GPUs busy"
fi

# Report status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
