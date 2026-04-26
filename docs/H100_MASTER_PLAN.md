# H100 DGX Cloud — Master Training Plan

> **Single source of truth.** This file replaces 6 former docs:
> `H100_TRAINING_PLAN.md`, `H100_TRAINING_PLAN_3WEEKS.md`, `H100_EXECUTION_PLAN.md`,
> `NEUROLINUX_H100_TRAINING_PLAN.md`, `H100_NEUROLINUX_READY.md`, `H100_DEPLOYMENT_VISION.md`
>
> Archived originals: `docs/archived/h100/`

**Last Updated:** Feb 7, 2026

---

## Quick Reference

| Field | Value |
|-------|-------|
| **Cluster** | NVIDIA H100 DGX — 8× H100 80 GB |
| **Access** | `ssh awesome-gpu-name` (24/7) |
| **Grant Period** | Jan 13 – Mar 14, 2026 |
| **Days Left** | ~35 |
| **Credits Remaining** | ~28,500 GPU-hrs |
| **Burn Rate** | 192 GPU-hrs/day (8 GPUs × 24 h) |
| **Extension** | 1–2 weeks requested Feb 5 (pending) |
| **Contact** | Christina Adams — dgxc-innovation-lab@nvidia.com |

---

## 1. Current GPU Allocation (Feb 6 snapshot)

| GPU | Job | Util | VRAM | Progress | Loss | ETA from Feb 6 |
|-----|-----|------|------|----------|------|-----------------|
| 0 | VLA-Max0 | 100 % | 25 GB | 202 K/500 K | 0.994 | ~Sun Feb 9 |
| 1 | VLA-Max1 | 100 % | 25 GB | 215 K/500 K | 0.994 | ~Sat Feb 8 |
| 2 | NIS-MoE v2 (16 experts) | 68 % | 21 GB | Starting | — | **Done** |
| 3 | VLA-Max3 | 100 % | 25 GB | 189 K/500 K | 0.999 | ~Mon Feb 10 |
| 4 | NIS-MoE Large (1024d, 24L) | 78 % | 17 GB | 2.9 K/100 K | 0.21 | **Done** |
| 5 | VLA-New5 | 100 % | 11 GB | 200/500 K | 1.01 | ~Sat Feb 8 |
| 6 | VLA-New6 | 100 % | 11 GB | 2.6 K/500 K | 1.02 | ~Sun Feb 9 |
| 7 | VLA-New7 | 100 % | 11 GB | 5.3 K/500 K | 1.01 | ~Sat Feb 8 |

> GPUs 2 & 4 likely idle now — MoE jobs finished. Check and reassign immediately.

---

## 2. Production Model Inventory — Post-Audit (Feb 24 2026)

> **Audit completed Feb 24 2026.** ~300 GB of intermediate checkpoints deleted.
> Storage: 3.6T → 3.3T. Only final/best models kept.

### Cosmos Inference Stack (Live on H100 — serving 24/7)

| Model | Size | Port | Status | NIS Route |
|-------|------|------|--------|-----------|
| cosmos-reason2-8b | 17 GB | :8100 | ✅ LIVE | `/cosmos/reason`, `/cookoff/robot-plan` |
| nis-cosmos-reason2-v2 (LoRA) | 3.7 GB | :8100 | ✅ ACTIVE (upgraded Feb 24) | adapter on reason2 |
| predict2/2B-Video2World | 249 GB | :8200 | ✅ LIVE | `/cosmos/predict`, `/cookoff/predict` |
| predict25-robotics-lora | 6.5 GB | :8200 | ✅ KEPT (final+best) | future: load on predict2 |
| transfer2.5/2B/general/edge | 48 GB | :8300 | ✅ LIVE | `/cookoff/transfer` |
| qwen3-vl-8b-instruct | 17 GB | :8200 | ✅ KEPT | Predict2.5 base |
| cosmos_demo_server | — | :8400 | ✅ LIVE | `/cookoff/status` |

### Large Language Models (H100 Fine-Tunes)

| Model | Size | Adapter | Status |
|-------|------|---------|--------|
| llm_base/qwen2.5-72b-instruct | 136 GB | qwen72b-nis-qlora-v2 (3.2 GB) | ✅ KEPT |
| llm_base/codellama-34b-instruct | 126 GB | codellama-34b-nis/best_lora (3.3 GB) | ✅ KEPT |
| llm_base/llama-3.1-70b-instruct | 68 KB* | — | ✅ KEPT |
| nis_llm/nis_llm_final.pt | 137 GB | — | ✅ KEPT |
| robotics_llm/robotics_llm_final.pt | 137 GB | — | ✅ KEPT |

*llama base is a symlink/stub — full weights in nis_llm

### VLA Models (Robot Action)

| Model | Size | Hardware | Status | NIS Route |
|-------|------|----------|--------|-----------|
| vla_xarm_v2 | 544 MB | xArm (our robot) | ✅ PRODUCTION | `/cookoff/execute` (pending wire) |
| vla_realdata_v4 | 186 MB | general | ✅ PRODUCTION | — |
| vla_pusht_real_final.pt | 17 MB | PushT | ✅ KEPT | — |
| vla_heavy_v2 | 14 GB | general | 🔄 TRAINING (ETA done) | — |

### Edge Models (Pi5 / Jetson Deployable)

| Category | Location | Size | Status | NIS Route |
|----------|----------|------|--------|-----------|
| NeMo ASR | neurolinux/nemo/ | 1.8 GB | ✅ PRODUCTION | `/voice/stt` |
| YOLO Vision | neurolinux/vision/ | 171 MB | ✅ PRODUCTION | `/vision/detect` |
| RL/PPO | neurolinux/rl/ | 35 MB | ✅ PRODUCTION | `/agents/rl` |
| Isaac Nav | neurolinux/isaac/ | 130 MB | ✅ PRODUCTION | `/agents/navigate` |
| SBERT Embeddings | neurolinux/embeddings/ | 2.3 GB | ✅ PRODUCTION | `/memory/embed` |
| World Model v4 | world_model_v4/final | 75 MB | ✅ KEPT | — |

### Archived / Deleted (Feb 24 2026)

| What | Why | Space Freed |
|------|-----|-------------|
| vla_maxbatch_step*.pt (50 files) | Intermediate checkpoints | ~210 GB |
| world_model_v4/epoch_* (500 dirs) | Intermediate epochs | ~37 GB |
| vla_pusht_real_step*.pt (20 files) | Intermediate checkpoints | ~340 MB |
| vla_realdata_unified_v3 steps (17) | Intermediate checkpoints | ~1.5 GB |
| cosmos_reason2_lora_v1/checkpoint-* | Superseded by v2 | ~500 MB |
| qwen72b_nis_qlora v1 | Superseded by v2 | ~3 GB |
| cosmos-reason2-8b-gguf | GGUF not used by server | ~5 GB |
| cosmos-aloha-policy | ALOHA robot, not xArm | ~2 GB |
| predict25-robotics-lora/checkpoint-* | Intermediate (kept final+best) | ~4 GB |
| **Total** | | **~263 GB** |

### Cluster Model Paths (Clean — Feb 24 2026)

```
/data/organica-ai/models/
├── cosmos/
│   ├── cosmos-reason2-8b/          # 17 GB — base model :8100
│   ├── nis-cosmos-reason2-v2/      # 3.7 GB — ACTIVE LoRA on :8100
│   ├── predict2/                   # 249 GB — base model :8200
│   ├── predict25-robotics-lora/    # 6.5 GB — robotics LoRA (final+best)
│   ├── transfer2.5/                # 48 GB — :8300
│   └── qwen3-vl-8b-instruct/       # 17 GB — Predict2.5 base
├── llm_base/
│   ├── qwen2.5-72b-instruct/       # 136 GB
│   ├── codellama-34b-instruct/     # 126 GB
│   └── llama-3.1-70b-instruct/     # stub
├── qwen72b-nis-qlora-v2/final_lora # 3.2 GB — best NIS LLM adapter
├── codellama-34b-nis/best_lora     # 3.3 GB — code adapter
├── nis_llm/nis_llm_final.pt        # 137 GB — NIS-LLM
├── robotics_llm/robotics_llm_final.pt # 137 GB — Robotics-LLM
├── vla_xarm_v2/                    # 544 MB — xArm VLA (our hardware)
├── vla_realdata_v4/                # 186 MB — real data VLA
├── vla_pusht_real_final.pt         # 17 MB
├── world_model_v4/final            # 75 MB
└── neurolinux/
    ├── nemo/                       # 1.8 GB — ASR
    ├── vision/                     # 171 MB — YOLO
    ├── rl/                         # 35 MB — PPO
    ├── isaac/                      # 130 MB — nav
    └── embeddings/                 # 2.3 GB — SBERT
```

---

## 3. Training Timeline

### Phase 1: PINN Foundation (Jan 15–17) ✅ COMPLETE

- 966 PINN models + 6 Transformer models
- ~280 GPU hours consumed

### Phase 2: NVIDIA Stack + NeuroLinux (Jan 18–28) ✅ COMPLETE

| GPUs | Task | Models |
|------|------|--------|
| 0–1 | NeMo ASR (voice commands) | 41 |
| 2–3 | Isaac Lab RL (nav + avoidance) | 10+ |
| 4 | Vision / YOLO (edge detection) | 32 |
| 5 | RL / PPO (autonomous nav) | 55 |
| 6 | Embeddings (offline knowledge) | 13 |
| 7 | Transformer (language) | 10 |

- 213 models total, ~70 GPU hours
- Quality retraining batch: 64 GPU hours (19 quality, 194 short)

### Phase 3: VLA + LLM + MoE (Jan 28 – Feb 10) 🔄 IN PROGRESS

**Week 1 (Jan 28 – Feb 3) ✅**
- VLA-Bimanual, VLA-Mobile, VLA-Surgical, VLA-Gripper, VLA-Industrial (all 500 K steps)

**Week 2 (Feb 4–10) 🔄**
- ✅ NIS-LLM (100 K steps, 76.16 h)
- ✅ Robotics-LLM (100 K steps, 53.34 h)
- ✅ NIS-MoE v1 (100 K steps, Loss 0.093)
- ✅ VLA-PushT Real (100 K steps, real data)
- ✅ VLA-4, VLA-6 (500 K steps each)
- 🔄 VLA-Max0, Max1, Max3 finishing Feb 8–10
- 🔄 NIS-MoE v2 (16 experts), NIS-MoE Large — likely done

### Phase 4: Specialized Models (Feb 11–17) — READY TO LAUNCH

**Script:** `launch_week3_training.sh`

| GPUs | Task | Epochs | ETA |
|------|------|--------|-----|
| 0–1 | Robotics CLIP (vision-language alignment) | 50 | ~6 h |
| 2–3 | Cosmos-VLA (reasoning + action for Cookoff) | 100 | ~12 h |
| 4–5 | Sim2Real domain adaptation | 80 | ~8 h |
| 6 | Safety classifier (action validation) | 60 | ~4 h |
| 7 | Speech-to-Action (voice → robot) | 80 | ~8 h |

### Phase 5: Export + Edge Deployment (Feb 18 – Mar 14)

- Full model export: rsync to local Mac + S3
- INT8/INT4 quantization for Pi5
- Edge deployment validation
- Cosmos Cookoff video (Feb 25)
- Documentation & benchmarks

---

## 4. Trainable Components Reference

| Component | Location | GPU Time | Status |
|-----------|----------|----------|--------|
| KAN Networks | `src/agents/signal_processing/kan_network.py` | Medium | Planned |
| Vision Agent (YOLO) | `src/agents/perception/vision_agent.py` | High | ✅ 32 models |
| Voice / ASR | `src/voice/vibevoice_realtime.py` | High | ✅ 41 models |
| BitNet Trainer | `src/agents/training/bitnet_online_trainer.py` | Very High | ✅ 4 models |
| PINN | `src/agents/physics/` | Medium-High | ✅ 966 models |
| Deep RL | `src/agents/learning/drl_foundation.py` | High | ✅ 55 models |
| LSTM Memory | `src/agents/memory/lstm_memory_core.py` | Medium | Planned |
| Embeddings | SBERT-based | Medium | ✅ 13 models |
| VLA (custom) | `scripts/train_vla_h100.py` | Very High | ✅ 8+ models |
| NIS-LLM | Llama 3.1 fine-tune | Very High | ✅ 1 model (76 h) |
| NIS-MoE | Mixture of Experts | High | ✅ 1 model |

---

## 5. NeuroLinux Edge Deployment

### Target Hardware

- **Raspberry Pi 5** (8 GB) — primary edge target
- **Jetson Orin Nano** — high-performance edge
- **Drones** (MAVLink/PX4), **Robot Arms** (CAN bus)

### Models Sized for Edge

| Model | Original | INT8 Quantized | Pi5 Compatible |
|-------|----------|----------------|----------------|
| VLA models | ~500 MB | ~125 MB | ✅ |
| Vision / YOLO | ~80 MB | ~20 MB | ✅ |
| NeMo ASR | ~540 MB | ~135 MB | ✅ |
| Embeddings | ~200 MB | ~50 MB | ✅ |
| BitNet | ~132 MB | ~33 MB | ✅ |
| Safety Classifier | ~50 MB | ~12 MB | ✅ |

### Deployment Steps

```bash
# 1. Sync from cluster to local Mac
rsync -avz --progress awesome-gpu-name:/data/organica-ai/models/ \
  ~/Desktop/Projects/NIS/NIS_Protocol/models/h100_trained/

# 2. Quantize for edge
python scripts/quantize_models.py --input models/h100_trained/ --output models/edge/ --format int8

# 3. Transfer to Pi5
scp -r models/edge/ pi@neurolinux.local:/opt/neurolinux/models/

# 4. Test on Pi5
curl -X POST 'http://neurolinux.local:8080/v4/nis/chat?message=pick%20up%20the%20red%20cube'
curl -X POST 'http://neurolinux.local:8080/v4/vision/detect'
```

---

## 6. Model Export Strategy

| Date | Action | Destination |
|------|--------|-------------|
| Feb 7 | **Sync now** — models at risk | Local Mac |
| Feb 10 | Week 2 full backup | Local Mac + S3 |
| Feb 18 | Week 3 full backup | Local Mac + S3 + GitHub LFS |
| Mar 10 | Final backup (4 days before grant ends) | All destinations |

```bash
# Sync to local Mac
rsync -avz --progress awesome-gpu-name:/data/organica-ai/models/ ~/NeuroLinux-Models/

# Create compressed archive
tar -czvf neurolinux-models-$(date +%Y%m%d).tar.gz ~/NeuroLinux-Models/

# Sync to S3 (if configured)
s5cmd sync ~/NeuroLinux-Models/ s3://organica-ai-models/
```

---

## 7. GPU Hour Tracking

| Batch | GPU Hours | Status |
|-------|-----------|--------|
| Phase 1: PINN + Transformer | ~280 | ✅ Complete |
| Phase 2: NVIDIA Stack | ~70 | ✅ Complete |
| Phase 2: Quality Retraining | ~64 | ✅ Complete |
| Phase 3: VLA + LLM + MoE | ~600 | 🔄 In progress |
| Phase 4: Week 3 Specialized | ~50 (est.) | ⏳ Pending |
| **Total Used** | **~1,500** | |
| **Remaining** | **~28,500** | |

**Grant Total:** 30,000 GPU hours (600-hour initial + extensions)
**Days Remaining:** ~35 (until Mar 14)

---

## 8. Scripts & Files Reference

### Launch Scripts

| Script | Purpose |
|--------|---------|
| `launch_week3_training.sh` | **Week 3 batch** — CLIP, Cosmos-VLA, Sim2Real, Safety, Speech |
| `launch_all_training.sh` | Original PINN 8-GPU launcher |
| `launch_neurolinux_training_h100.sh` | NeuroLinux-specific (BitNet, Vision, RL, Embeddings) |
| `launch_nvidia_stack_training_v2.sh` | NeMo + Isaac + PINN + Transformer |

### Training Scripts

| Script | Purpose |
|--------|---------|
| `h100_parallel_training.py` | 8-GPU parallel launcher (Python) |
| `train_nvidia_stack_unified_h100.py` | NeMo + Isaac + GR00T + Vision (534 lines) |
| `scripts/train_vla_h100.py` | VLA model training (555 lines) |

### Operations

| Script | Purpose |
|--------|---------|
| `h100_setup.sh` | Initial cluster setup (packages, venv, PyTorch) |
| `h100_auto_monitor.sh` | Auto-restart idle GPUs (cron) |
| `auto_restart_training.sh` | Watchdog loop — checks every 5 min |

### Inference

| File | Purpose |
|------|---------|
| `src/inference/h100_models.py` | `H100ModelInference` class (NeMo, Isaac, Vision) |
| `routes/h100_inference.py` | REST API at `/h100/*` (health, ASR, RL, tracking) |

### Local Model Directory

```
models/h100_trained/
├── nemo_asr/          # ← EMPTY — needs rsync
├── isaac_rl/          # ← EMPTY — needs rsync
├── vision/            # ← EMPTY — needs rsync
├── vision_tracking/   # ← EMPTY — needs rsync
└── embeddings/        # ← EMPTY — needs rsync
```

---

## 9. Key Commands

```bash
# SSH to cluster
ssh awesome-gpu-name

# GPU status
nvidia-smi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv

# tmux sessions
tmux ls
tmux attach -t <name>

# Training logs
tail -f /data/organica-ai/logs/<job>.log
ls -lhS /data/organica-ai/models/   # List models by size

# Launch Week 3
cd ~/organica-ai && bash launch_week3_training.sh

# Auto-monitor (run in tmux)
bash auto_restart_training.sh
```

---

## 10. Escalation Plan

| Issue | Action |
|-------|--------|
| GPU failure | Redistribute workload to remaining GPUs |
| Credit exhaustion | Complete surveys for more hours |
| Model quality issues | Prioritize VLA and core models |
| Export failures | Multiple backup methods (rsync, scp, s5cmd) |
| H100 temps > 85 °C | Reduce batch size or pause 1 GPU |
| Grant expiring | Extension requested (pending) |

---

## 11. Training Progress Log

| Date | Event | GPUs | Hours | Output |
|------|-------|------|-------|--------|
| Jan 15–17 | PINN + Transformer | 8 | ~280 | 972 models |
| Jan 18–22 | NVIDIA Stack + NeuroLinux | 8 | ~70 | 213 models |
| Jan 23 | Quality Retraining | 8 | ~64 | 19 quality models |
| Jan 28–Feb 3 | VLA suite (Week 1) | 8 | ~192 | 5 VLA models |
| Feb 4 | VLA max batch started | 8 | — | Running |
| Feb 4 | NIS-LLM fine-tuning started | 1 | 76 h | ✅ Completed |
| Feb 4 | Robotics-LLM started | 1 | 53 h | ✅ Completed |
| Feb 5 | NIS-MoE training (GPU 7) | 1 | 5.6 h | ✅ Completed |
| Feb 5 | Extension request sent to NVIDIA | — | — | Pending |
| Feb 6 | NIS-MoE v2 + Large started | 2 | ~12 h | Likely done |
| Feb 6 | VLA-New5/6/7 started | 3 | ~30 h | Running |
| **Feb 11** | **Week 3 launch** | **8** | **~50** | **Pending** |

---

## 12. Success Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Model count | 50+ | **160+** ✅ |
| Model diversity | 8+ types | **11 types** ✅ |
| GPU utilization | > 30 % | **68–100 %** ✅ |
| Training stability | No OOM | **Zero crashes** ✅ |
| Production-ready models | 3+ | **10+** ✅ |
| Edge-deployable models | 5+ | **All core models** ✅ |
| Total GPU hours | Maximize grant | **~1,500 of 28,500** 🔄 |

---

## 13. NVIDIA Extension Request

**Status:** Pending (sent Feb 5, 2026)
**Requested:** 1–2 week extension beyond Mar 14
**Contact:** Christina Adams (dgxc-innovation-lab@nvidia.com)
**Justification:** Active training, 160+ models produced, Cosmos Cookoff submission

---

**RULE: Never let an H100 sit idle. Check `nvidia-smi` daily. Run `auto_restart_training.sh` in a tmux session.**
