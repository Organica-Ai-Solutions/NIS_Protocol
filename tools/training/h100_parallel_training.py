#!/usr/bin/env python3
"""
H100 Parallel Training Launcher
Starts 8 independent training jobs, one per GPU.

CREDIT WARNING: The default TRAINING_JOBS reference small scripts (PINN, KAN, etc.)
that use ~2-5 GB VRAM each — wasteful on 80 GB H100s. Prefer:
  bash scripts/h100_heavy/launch_heavy.sh
which runs 40-50 GB models. See docs/H100_CREDIT_AUDIT.md.
"""

import subprocess
import time
import os

# Training jobs configuration
TRAINING_JOBS = [
    {
        "name": "PINN_Heat",
        "gpu": 0,
        "script": "train_pinn_heat.py",
        "log": "pinn_heat.log"
    },
    {
        "name": "PINN_Wave",
        "gpu": 1,
        "script": "train_pinn_wave.py",
        "log": "pinn_wave.log"
    },
    {
        "name": "PINN_Laplace",
        "gpu": 2,
        "script": "train_pinn_laplace.py",
        "log": "pinn_laplace.log"
    },
    {
        "name": "KAN_Network",
        "gpu": 3,
        "script": "train_kan.py",
        "log": "kan.log"
    },
    {
        "name": "Vision_YOLO",
        "gpu": 4,
        "script": "train_vision.py",
        "log": "vision.log"
    },
    {
        "name": "Voice_ASR",
        "gpu": 5,
        "script": "train_voice.py",
        "log": "voice.log"
    },
    {
        "name": "LLM_FineTune",
        "gpu": 6,
        "script": "train_llm.py",
        "log": "llm.log"
    },
    {
        "name": "Embeddings",
        "gpu": 7,
        "script": "train_embeddings.py",
        "log": "embeddings.log"
    }
]

def launch_training_job(job):
    """Launch a training job on a specific GPU"""
    gpu_id = job["gpu"]
    script = job["script"]
    log_file = f"~/organica-ai/logs/{job['log']}"
    
    # Set CUDA_VISIBLE_DEVICES to use specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = f"python3 ~/organica-ai/training/{script} > {log_file} 2>&1 &"
    
    print(f"🚀 Launching {job['name']} on GPU {gpu_id}")
    print(f"   Script: {script}")
    print(f"   Log: {log_file}")
    
    subprocess.Popen(cmd, shell=True, env=env)
    time.sleep(2)  # Small delay between launches

def main():
    print("="*60)
    print("H100 Parallel Training Launcher")
    print("="*60)
    print(f"Starting {len(TRAINING_JOBS)} training jobs...")
    print()
    
    for job in TRAINING_JOBS:
        launch_training_job(job)
    
    print()
    print("="*60)
    print("✅ All training jobs launched!")
    print("="*60)
    print()
    print("Monitor with:")
    print("  nvidia-smi  # GPU usage")
    print("  tail -f ~/organica-ai/logs/*.log  # Training logs")
    print("  ps aux | grep python  # Running processes")
    print()
    print("Remember: 600 GPU hours = ~75 hours with all 8 GPUs")
    print("Keep these running 24/7 to maximize your grant!")

if __name__ == "__main__":
    main()
