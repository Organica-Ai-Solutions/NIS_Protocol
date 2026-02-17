# H100 Max Utilization Plan — Week 4+ (Feb 12-26)

**Created:** Feb 11, 2026  
**Goal:** Stop wasting 80GB H100s on 1-3GB models. Maximize every GPU-hour before credits expire.  
**Deadline:** Feb 26 (Cookoff submission) / TBD (grant expiry)

---

## The Problem

We've been running **1-3 GB models** on **80 GB GPUs**. Most jobs finish in minutes.
That's like renting a semi-truck to deliver a letter.

| Metric | Current | Target |
|--------|---------|--------|
| Avg VRAM per GPU | 3 GB (4%) | 60+ GB (75%+) |
| Avg job duration | 10 min - 2 hr | 12-72 hr |
| Model params trained | 1M - 276M | 8B - 72B |
| Multi-GPU jobs | 0 | 2-3 |

---

## Phase 0: Environment Setup (30 min)

Install missing packages needed for large-model fine-tuning:

```bash
# On H100 cluster
~/organica-ai/venv/bin/pip install \
  deepspeed \
  bitsandbytes \
  trl \
  flash-attn --no-build-isolation \
  wandb
```

---

## Phase 1: Large Model Fine-Tuning (Feb 12-18)

### Job A — Cosmos Reason2 8B Full Fine-Tune (1 GPU, ~40 GB)
**GPU:** 0 | **Duration:** 24-48h | **Priority:** CRITICAL (Cookoff)

Fine-tune the actual Cosmos Reason 2 model on our robotics data.
This is the #1 Cookoff differentiator — a Cosmos model fine-tuned on OUR domain.

```bash
# Single GPU, full fine-tune (8B model fits in 80GB with BF16)
CUDA_VISIBLE_DEVICES=0 python train_cosmos_reason2_finetune.py \
  --model_path /data/organica-ai/models/cosmos/cosmos-reason2-8b \
  --dataset /data/organica-ai/datasets/aloha,pusht,xarm,robotics \
  --output /data/organica-ai/models/cosmos/nis-cosmos-reason2-finetuned \
  --epochs 3 --batch_size 4 --gradient_accumulation 8 \
  --lr 2e-5 --bf16 --flash_attn
```

### Job B — Qwen 2.5 72B QLoRA (4 GPUs, ~280 GB)
**GPUs:** 0,1,4,7 | **Duration:** 48-72h | **Priority:** HIGH

Fine-tune the 72B model with QLoRA (4-bit quantized base + LoRA adapters).
This gives us a **production-grade 72B robotics/NIS LLM**.

```bash
# Multi-GPU QLoRA with DeepSpeed ZeRO-3
accelerate launch --num_processes 4 --config_file ds_zero3.yaml \
  train_qwen72b_qlora.py \
  --model_path /data/organica-ai/models/llm_base/qwen2.5-72b-instruct \
  --dataset /data/organica-ai/datasets/text,robotics \
  --output /data/organica-ai/models/qwen72b-nis-qlora \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --per_device_batch_size 2 --gradient_accumulation 16 \
  --epochs 1 --lr 1e-4 --bf16 --flash_attn
```

### Job C — CodeLlama 34B Full Fine-Tune (2 GPUs, ~140 GB)
**GPUs:** 1,4 | **Duration:** 24-36h | **Priority:** MEDIUM

Fine-tune on NIS Protocol codebase for code generation/completion.

```bash
accelerate launch --num_processes 2 \
  train_codellama34b_finetune.py \
  --model_path /data/organica-ai/models/llm_base/codellama-34b-instruct \
  --dataset /data/organica-ai/datasets/text \
  --output /data/organica-ai/models/codellama-34b-nis \
  --per_device_batch_size 1 --gradient_accumulation 32 \
  --epochs 2 --lr 5e-6 --bf16
```

### Job D — Mistral 7B Full Fine-Tune (1 GPU, ~28 GB)
**GPU:** 7 | **Duration:** 12-24h | **Priority:** MEDIUM

Smallest foundation model — good for edge deployment after quantization.

```bash
CUDA_VISIBLE_DEVICES=7 python train_mistral_finetune.py \
  --model_path /data/organica-ai/models/llm_base/mistral-7b-instruct \
  --output /data/organica-ai/models/mistral-7b-nis \
  --batch_size 8 --gradient_accumulation 4 \
  --epochs 3 --lr 2e-5 --bf16
```

---

## Phase 2: Massive Synthetic Data Generation (Feb 14-20)

Use Cosmos Predict 2.5 + Transfer 2.5 to generate millions of synthetic frames.

### Job E — Synthetic Robot Training Data (1-2 GPUs)
```bash
# Generate 1M synthetic frames using Cosmos Predict2.5
python generate_cosmos_synthetic.py \
  --source_datasets aloha,pusht,xarm \
  --output /data/organica-ai/datasets/synthetic_cosmos/ \
  --num_frames 1000000 \
  --augmentation_modes lighting,weather,style,time_of_day \
  --predict_futures 5 --fps 30
```

### Job F — Domain Augmentation via Transfer 2.5
```bash
# Transform existing data to new domains (factory, warehouse, outdoor)
python generate_domain_transfer.py \
  --source /data/organica-ai/datasets/ \
  --output /data/organica-ai/datasets/domain_augmented/ \
  --domains factory,warehouse,outdoor,underwater \
  --variations_per_image 10
```

---

## Phase 3: Multi-GPU Training Runs (Feb 18-26)

### Job G — Train Large VLA on Synthetic + Real Data (4 GPUs)
After generating synthetic data, train a much larger VLA model.

```bash
accelerate launch --num_processes 4 \
  train_vla_large.py \
  --dataset /data/organica-ai/datasets/synthetic_cosmos,aloha,pusht,xarm \
  --model_size large --hidden_dim 2048 --num_layers 24 \
  --batch_size 64 --epochs 10 \
  --output /data/organica-ai/models/vla_large_synthetic
```

### Job H — Retrain NIS-MoE v4 on Larger Scale (2 GPUs)
Use the new synthetic data to train a bigger MoE.

```bash
accelerate launch --num_processes 2 \
  train_nis_moe_v4.py \
  --hidden_dim 1024 --num_layers 24 --num_experts 32 \
  --dataset /data/organica-ai/datasets/synthetic_cosmos \
  --steps 2000000 --batch_size 128
```

---

## GPU Scheduling (Week 4)

### Option A: Sequential (Simpler)
```
Day 1 (Feb 12):  Install packages + Launch Cosmos 8B FT (GPU 0) + Mistral FT (GPU 7)
                  Keep: MoE v2 (GPU 3), MoE v3 (GPU 5), VLA v4 (GPU 6)
Day 2 (Feb 13):  MoE v2 finishes → Launch CodeLlama 34B FT (GPUs 3,4)
Day 3 (Feb 14):  Cosmos 8B FT done → Start synthetic data gen (GPU 0)
                  Mistral done → Start Yi 34B FT (GPU 7)
Day 4-6:         Launch Qwen 72B QLoRA (GPUs 0,1,4,7) — 48-72h run
Day 7+:          Train large VLA on synthetic data
```

### Option B: Aggressive Multi-GPU (Max Throughput)
```
Day 1:  Install packages
        GPUs 0,1,4,7 → Qwen 72B QLoRA (the biggest win, start immediately)
        GPU 3 → Keep MoE v2 (finishes Day 2)
        GPU 5 → Keep MoE v3 (finishes Day 7)
        GPU 6 → Keep VLA v4
Day 2:  GPU 3 freed → Cosmos 8B FT (single GPU, 24h)
Day 3:  Qwen 72B done → CodeLlama 34B (GPUs 0,1) + Synthetic gen (GPU 4) + Mistral (GPU 7)
Day 5+: Large VLA training on synthetic data (4 GPUs)
```

---

## Expected Outputs

| Model | Size | Use Case | Cookoff Value |
|-------|------|----------|---------------|
| NIS-Cosmos-Reason2-FT | 8B | Physics reasoning for robots | ⭐⭐⭐ Critical |
| Qwen-72B-NIS-QLoRA | 72B | General NIS intelligence | ⭐⭐ High |
| CodeLlama-34B-NIS | 34B | NIS code generation | ⭐ Medium |
| Mistral-7B-NIS | 7B | Edge deployment LLM | ⭐⭐ High |
| Synthetic dataset | 1M+ frames | Training data for everything | ⭐⭐⭐ Critical |
| VLA-Large | 500M+ | Robot control (bigger model) | ⭐⭐ High |

---

## Scripts to Create

1. `training/train_cosmos_reason2_finetune.py` — Full FT of Cosmos Reason 2 8B
2. `training/train_qwen72b_qlora.py` — QLoRA fine-tune with DeepSpeed
3. `training/train_codellama34b_finetune.py` — CodeLlama domain FT
4. `training/train_mistral_finetune.py` — Mistral 7B full FT
5. `training/generate_cosmos_synthetic.py` — Massive synthetic data pipeline
6. `training/train_vla_large.py` — Large VLA on synthetic data
7. `configs/ds_zero3.yaml` — DeepSpeed ZeRO-3 config for multi-GPU

---

## Assets Already on Cluster

### Models Ready to Fine-Tune
| Model | Path | Size | Downloaded |
|-------|------|------|-----------|
| Qwen 2.5 72B Instruct | `llm_base/qwen2.5-72b-instruct` | 136 GB | ✅ |
| CodeLlama 34B Instruct | `llm_base/codellama-34b-instruct` | 126 GB | ✅ |
| DeepSeek Coder 33B | `llm_base/deepseek-coder-33b` | 125 GB | ✅ |
| Yi 1.5 34B Chat | `llm_base/yi-1.5-34b-chat` | 65 GB | ✅ |
| Mistral 7B Instruct | `llm_base/mistral-7b-instruct` | 28 GB | ✅ |
| Cosmos Reason2 8B | `cosmos/cosmos-reason2-8b` | 17 GB | ✅ |
| Qwen3-VL 8B | `cosmos/qwen3-vl-8b-instruct` | 17 GB | ✅ |
| Llama 3.1 70B | `llm_base/llama-3.1-70b-instruct` | 68 KB ⚠️ | ❌ Stub only |

### Installed Packages
| Package | Installed | Needed |
|---------|-----------|--------|
| torch 2.5.1+cu121 | ✅ | - |
| transformers 4.57.5 | ✅ | - |
| accelerate 1.12.0 | ✅ | - |
| peft 0.18.1 | ✅ | - |
| datasets 4.5.0 | ✅ | - |
| deepspeed | ❌ | Multi-GPU training |
| bitsandbytes | ❌ | QLoRA quantization |
| trl | ❌ | RLHF / SFT Trainer |
| flash-attn | ❌ | 2x faster attention |
| wandb | ❌ | Experiment tracking |

### Disk
- `/data`: 53 TB total, 3.1 TB used, **50 TB free**
- Room for terabytes of synthetic data and model checkpoints

---

*This plan replaces the "cycle tiny models" approach with proper H100 utilization.*
*Every GPU should be using 40-80 GB of VRAM at all times.*
