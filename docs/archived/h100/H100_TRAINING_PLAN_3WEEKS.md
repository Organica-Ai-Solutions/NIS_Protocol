# H100 DGX Cloud Training Plan - BURN RATE MAXIMIZATION

**Grant Period:** Jan 13 - Mar 14, 2026 (+ 1-2 week extension pending)  
**Current Date:** Feb 6, 2026  
**Days Remaining:** ~36 days (+ potential extension)  
**GPU Credits Remaining:** ~28,500 hours  
**Target Burn Rate:** 192 GPU-hrs/day (8 GPUs × 24h) ✅ ACHIEVED

---

## 📊 Current State (Feb 6, 2026) - UPDATED

### Completed Models (160+)
| Category | Count | Size | Status |
|----------|-------|------|--------|
| NeMo ASR | 41 | ~540 MB | ✅ Production ready |
| Vision/YOLO | 32 | 79 MB | ✅ Production ready |
| Embeddings | 10+ | ~200 MB | ✅ Production ready |
| Transformer/GPT | 10 | 1.98 GB | ✅ Production ready |
| RL/PPO | 55 | 18.5 MB | ✅ Production ready |
| Isaac Navigation | 10+ | ~20 MB | ✅ Production ready |
| BitNet | 4 | 132 MB | ✅ Production ready |
| **VLA Models** | **8** | **~2 GB** | ✅ **Completed** |
| **NIS-LLM** | **1** | **~14 GB** | ✅ **Completed** |
| **Robotics-LLM** | **1** | **~14 GB** | ✅ **Completed** |
| **NIS-MoE** | **1** | **~1 GB** | ✅ **Completed** |
| **VLA-PushT Real** | **1** | **~100 MB** | ✅ **Completed** |

### Current GPU Allocation (Feb 6, 2026 - 8/8 ACTIVE)
| GPU | Current Task | Utilization | Memory | Progress | ETA |
|-----|--------------|-------------|--------|----------|-----|
| 0 | VLA-Max0 | 100% | 25 GB | 202K/500K | ~70h |
| 1 | VLA-Max1 | 100% | 25 GB | 215K/500K | ~63h |
| 2 | NIS-MoE v2 (16 experts) | 68% | 21 GB | Starting | ~6h |
| 3 | VLA-Max3 | 100% | 25 GB | 189K/500K | ~79h |
| 4 | NIS-MoE Large (1024d) | 78% | 17 GB | 2.9K/100K | ~6h |
| 5 | VLA-New5 | 100% | 11 GB | 200/500K | ~28h |
| 6 | VLA-New6 | 100% | 11 GB | 2.6K/500K | ~35h |
| 7 | VLA-New7 | 100% | 11 GB | 5.3K/500K | ~17h |

### 🔥 GPU Credit Burn Rate Analysis
| Metric | Value |
|--------|-------|
| Credits Remaining | ~28,500 GPU-hrs |
| Current Burn Rate | 192 hrs/day (8 GPUs) ✅ |
| Days to Exhaust | ~148 days at max |
| Grant Ends | Mar 14, 2026 (~36 days) |
| **Projected Usage** | **~6,900 hrs** (if maintained) |
| **Extension Requested** | 1-2 weeks (pending NVIDIA response) |

✅ **STATUS**: All 8 GPUs running at 100% utilization 24/7!

---

## 🎯 Week 1: Jan 28 - Feb 3 (VLA Focus) ✅ COMPLETED

### Priority: Complete VLA Model Suite

**Goal:** Train production-ready VLA models for all robot types

### Week 1 Deliverables ✅
- [x] VLA-Bimanual model trained (500K steps)
- [x] VLA-Mobile model trained (500K steps)
- [x] VLA-Surgical model trained (500K steps)
- [x] VLA-Gripper model trained (500K steps)
- [x] VLA-Industrial model trained (500K steps)
- [x] Initial model export to cluster storage

---

## 🎯 Week 2: Feb 4 - Feb 10 (LLM Fine-Tuning + Foundation Models) 🔄 IN PROGRESS

### Priority: Fine-tune Open-Source LLMs for NIS Protocol

**Goal:** Create NIS-LLM - a specialized LLM for robot command parsing

### Week 2 Completed (Feb 6) ✅
- [x] **NIS-LLM** - 100K steps, 76.16 hours training ✅
- [x] **Robotics-LLM** - 100K steps, 53.34 hours training ✅
- [x] **NIS-MoE** - 100K steps, 276M params, Loss: 0.093 ✅
- [x] **VLA-PushT Real** - 100K steps on real robot data ✅
- [x] **VLA-4** - 500K steps completed ✅
- [x] **VLA-6** - 500K steps completed ✅

### Currently Training (Feb 6)
| GPU | Job | Progress | Loss | ETA |
|-----|-----|----------|------|-----|
| 0 | VLA-Max0 | 202K/500K | 0.994 | ~70h |
| 1 | VLA-Max1 | 215K/500K | 0.994 | ~63h |
| 2 | NIS-MoE v2 (16 experts) | Starting | - | ~6h |
| 3 | VLA-Max3 | 189K/500K | 0.999 | ~79h |
| 4 | NIS-MoE Large (1024d) | 2.9K/100K | 0.21 | ~6h |
| 5 | VLA-New5 | 200/500K | 1.01 | ~28h |
| 6 | VLA-New6 | 2.6K/500K | 1.02 | ~35h |
| 7 | VLA-New7 | 5.3K/500K | 1.01 | ~17h |

### Week 2 Remaining Deliverables
- [ ] Complete VLA-Max0, VLA-Max1, VLA-Max3 (500K steps each)
- [ ] Complete NIS-MoE v2 and NIS-MoE Large variants
- [ ] Export all models to local storage

---

## 🎯 Week 3: Feb 11 - Feb 18 (Specialized + Export)

### Priority: Specialized Models + Full Export

**Goal:** Train niche models and ensure all models are backed up

| GPU | Task | Model | Purpose |
|-----|------|-------|---------|
| 0-1 | Multi-modal | CLIP-style | Vision-language alignment |
| 2-3 | Speech-to-Action | Whisper+VLA | Voice-controlled robots |
| 4-5 | Sim2Real | Domain adaptation | Transfer learning |
| 6-7 | Safety Classifier | BERT-safety | Action validation |

### Training Scripts
```bash
# GPU 0-1: Multi-modal alignment
CUDA_VISIBLE_DEVICES=0,1 python train_multimodal.py \
  --model clip-vit-b32 \
  --dataset robotics_captions \
  --output ~/organica-ai/models/multimodal/clip_robotics.pt

# GPU 2-3: Speech-to-Action
CUDA_VISIBLE_DEVICES=2,3 python train_speech_action.py \
  --model whisper-small+vla \
  --dataset voice_commands_100k \
  --output ~/organica-ai/models/speech/whisper_vla.pt

# GPU 4-5: Sim2Real adaptation
CUDA_VISIBLE_DEVICES=4,5 python train_sim2real.py \
  --source isaac_sim \
  --target real_world \
  --output ~/organica-ai/models/sim2real/domain_adapt.pt

# GPU 6-7: Safety classifier
CUDA_VISIBLE_DEVICES=6,7 python train_safety.py \
  --model bert-base \
  --dataset safety_actions_50k \
  --output ~/organica-ai/models/safety/action_classifier.pt
```

### Week 3 Deliverables
- [ ] Multi-modal CLIP for robotics
- [ ] Speech-to-action model
- [ ] Sim2Real domain adaptation
- [ ] Safety action classifier
- [ ] **FULL MODEL EXPORT** to permanent storage

---

## 📦 Model Export Strategy

### Export Schedule
| Date | Action | Destination |
|------|--------|-------------|
| Feb 3 | Week 1 backup | Local Mac + S3 |
| Feb 10 | Week 2 backup | Local Mac + S3 |
| Feb 18 | Full export | Local Mac + S3 + GitHub LFS |
| Mar 10 | Final backup | All destinations (4 days before expiry) |

### Export Commands
```bash
# Sync to local Mac
rsync -avz --progress awesome-gpu-name:~/organica-ai/models/ ~/NeuroLinux-Models/

# Sync to S3 (if configured)
s5cmd sync ~/organica-ai/models/ s3://organica-ai-models/

# Create compressed archive
tar -czvf neurolinux-models-$(date +%Y%m%d).tar.gz ~/organica-ai/models/
```

---

## 📊 Expected Final Model Inventory (Feb 18)

### Core Models

| Model | Size | Purpose | Edge Compatible |
|-------|------|---------|-----------------|
| `vla_manipulation_v1.pt` | ~500MB | Arm control | ✅ Pi5/Jetson |
| `vla_navigation_v1.pt` | ~500MB | Mobile robots | ✅ Pi5/Jetson |
| `vla_drone_v1.pt` | ~500MB | Aerial robots | ✅ Pi5/Jetson |
| `vla_quadruped_v1.pt` | ~500MB | Legged robots | ✅ Pi5/Jetson |
| `vla_realdata_pusht.pt` | ~1.5GB | Real robot data | ✅ Jetson |
| `vla_realdata_aloha.pt` | ~1.5GB | Bimanual manipulation | ✅ Jetson |
| `vla_realdata_xarm.pt` | ~1.5GB | xArm manipulation | ✅ Jetson |
| `vision_yolov8x.pt` | ~200MB | Object detection | ✅ Pi5/Jetson |
| `clip_robotics.pt` | ~400MB | Vision-language | ✅ Pi5/Jetson |
| `whisper_vla.pt` | ~500MB | Voice control | ✅ Pi5/Jetson |
| `safety_classifier.pt` | ~400MB | Action safety | ✅ Pi5/Jetson |

### NIS-LLM Family (Fine-tuned LLMs)

| Model | Base | Size | Purpose | Deployment |
|-------|------|------|---------|------------|
| `nis-llm-70b/` | Llama 3.1 70B | ~140GB | Full capability | Cloud (NIM/vLLM) |
| `nis-llm-7b/` | Mistral 7B | ~14GB | Edge inference | Jetson Orin |
| `nis-coder-33b/` | DeepSeek-Coder | ~66GB | Code generation | Cloud |
| `nis-safety-7b/` | Llama 3.1 8B | ~16GB | Safety validation | Cloud/Edge |

### Quantized Versions (for Pi5)
| Model | Original | Quantized (INT8) | Speedup |
|-------|----------|------------------|---------|
| VLA models | 500MB | ~125MB | 2-4x |
| Vision | 200MB | ~50MB | 2-3x |
| Embeddings | 1.3GB | ~350MB | 2-3x |

---

## 🚨 Critical Reminders

1. **Grant expires March 14, 2026** - No extensions
2. **Export models weekly** - Don't lose 2 months of training
3. **Monitor GPU temps** - H100s running hot (87°C on some)
4. **Track credit usage** - Request top-ups if needed
5. **Document everything** - Training configs, hyperparameters

---

## 📞 Escalation Plan

If issues arise:
1. **GPU failures** → Redistribute workload to remaining GPUs
2. **Credit exhaustion** → Complete surveys for more hours
3. **Model quality issues** → Prioritize VLA and core models
4. **Export failures** → Use multiple backup methods (rsync, scp, s5cmd)

---

*Plan created: Jan 28, 2026*  
*Last updated: Feb 6, 2026*  
*Next review: Feb 10, 2026*

---

## 📈 Training Progress Log

### Feb 6, 2026 - Major Milestone Day
**Completed Jobs:**
- ✅ NIS-LLM (100K steps, 76.16h)
- ✅ Robotics-LLM (100K steps, 53.34h)
- ✅ NIS-MoE (100K steps, 276M params, Loss: 0.093)
- ✅ VLA-PushT Real (100K steps, real robot data)
- ✅ VLA-4 (500K steps)
- ✅ VLA-6 (500K steps)

**New Jobs Started:**
- NIS-MoE v2 (16 experts) on GPU 2
- NIS-MoE Large (1024d, 24L) on GPU 4
- VLA-New5 on GPU 5
- VLA-New6 on GPU 6
- VLA-New7 on GPU 7

**Models Saved:**
```
/data/organica-ai/models/nis_moe_h100_final.pt
/data/organica-ai/models/nis_moe_h100_step*.pt (20 checkpoints)
/data/organica-ai/models/vla_pusht_real_final.pt
/data/organica-ai/models/vla_max4_final.pt
/data/organica-ai/models/vla_max6_final.pt
```

### Feb 5, 2026
- Started NIS-MoE training on GPU 7
- Uploaded NIS-MoE repo to cluster (3.5GB)
- All 8 GPUs at 100% utilization
- Sent extension request to NVIDIA (1-2 weeks)

### Feb 4, 2026
- Started VLA max batch training on all GPUs
- NIS-LLM fine-tuning in progress
- Robotics-LLM training started
