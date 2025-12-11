# DGX Cloud Preparation Scripts

Scripts to maximize value from NVIDIA DGX Cloud H100 credits ($100K grant).

## Overview

**Goal:** Don't waste GPU hours on experimentation. Have clear scripts ready to run.

| Script | Purpose | Priority |
|--------|---------|----------|
| `benchmark_cpu_baseline.py` | Measure current CPU performance | Run first |
| `prepare_training_data.py` | Export NIS-HUB telemetry for training | Before training |
| `train_bitnet_h100.py` | Fine-tune BitNet on H100 | Primary goal |
| `accelerate_pinn.py` | TensorRT optimization for physics | Secondary goal |
| `benchmark_multi_agent.py` | Test 50+ agent scaling | Tertiary goal |

## Pre-Approval Checklist

- [ ] Run `benchmark_cpu_baseline.py` to establish baselines
- [ ] Collect training data from NIS-HUB
- [ ] Verify model architecture is H100-compatible
- [ ] Test scripts locally in simulation mode

## Estimated GPU Hours

| Task | Hours | Cost @ $3/hr |
|------|-------|--------------|
| BitNet fine-tuning (30 epochs) | 500-1000 | $1,500-3,000 |
| PINN TensorRT optimization | 100-200 | $300-600 |
| Multi-agent benchmarks | 200-500 | $600-1,500 |
| Buffer for experimentation | 500 | $1,500 |
| **Total** | **1,300-2,200** | **$4,000-6,600** |

$100K credits = ~33,000 GPU hours. We have plenty of headroom.

## Usage

```bash
# 1. Establish baselines (run locally)
python scripts/dgx_cloud/benchmark_cpu_baseline.py

# 2. Prepare training data
python scripts/dgx_cloud/prepare_training_data.py

# 3. On DGX Cloud (after approval)
python scripts/dgx_cloud/train_bitnet_h100.py --epochs 30 --batch-size 32
python scripts/dgx_cloud/accelerate_pinn.py --export-tensorrt
python scripts/dgx_cloud/benchmark_multi_agent.py --agents 50
```
