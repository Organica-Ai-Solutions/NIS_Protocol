# H100 Execution Plan

**Node:** NVIDIA H100 8x GPU Cluster  
**Grant Period:** Jan 13 - Mar 14, 2026 (+ 1-2 week extension pending)  
**Days Left:** ~36 days  
**Access:** 24/7 via SSH (awesome-gpu-name)  
**Last Updated:** Feb 6, 2026

## 🚨 CURRENT STATUS: ALL 8 GPUs ACTIVE ✅

**Burn Rate:** 192 GPU-hrs/day (maximum)  
**Credits Used:** ~1,500 GPU-hrs  
**Credits Remaining:** ~28,500 GPU-hrs

---

## 📊 Current GPU Allocation (Feb 6, 2026)

| GPU | Job | Progress | ETA |
|-----|-----|----------|-----|
| 0 | VLA-Max0 | 202K/500K | ~70h |
| 1 | VLA-Max1 | 215K/500K | ~63h |
| 2 | NIS-MoE v2 (16 experts) | Starting | ~6h |
| 3 | VLA-Max3 | 189K/500K | ~79h |
| 4 | NIS-MoE Large (1024d) | 2.9K/100K | ~6h |
| 5 | VLA-New5 | 200/500K | ~28h |
| 6 | VLA-New6 | 2.6K/500K | ~35h |
| 7 | VLA-New7 | 5.3K/500K | ~17h |

---

## ✅ Completed Models (Feb 6)

| Model | Steps | Duration | Size |
|-------|-------|----------|------|
| NIS-LLM | 100K | 76.16h | ~14 GB |
| Robotics-LLM | 100K | 53.34h | ~14 GB |
| NIS-MoE | 100K | 5.6h | ~1 GB |
| VLA-PushT Real | 100K | 0.4h | ~100 MB |
| VLA-4 | 500K | ~24h | ~500 MB |
| VLA-6 | 500K | ~24h | ~500 MB |
| VLA-Bimanual | 500K | ~24h | ~500 MB |
| VLA-Mobile | 500K | ~24h | ~500 MB |
| + 150 earlier models | - | - | ~2 TB |

---

## 📅 Remaining Timeline

### Week 2 (Feb 4-10) - IN PROGRESS
- [x] NIS-LLM fine-tuning ✅
- [x] NIS-MoE training ✅
- [x] Real data VLA training ✅
- [ ] Complete VLA-Max0, Max1, Max3 (500K each)
- [ ] Complete NIS-MoE variants

### Week 3 (Feb 11-17) — READY TO LAUNCH
Script: `launch_week3_training.sh`
- [ ] GPU 0-1: Robotics CLIP (vision-language alignment, 50 epochs, ~6h)
- [ ] GPU 2-3: Cosmos-VLA (reasoning + action, 100 epochs, ~12h)
- [ ] GPU 4-5: Sim2Real domain adaptation (80 epochs, ~8h)
- [ ] GPU 6: Safety classifier (action validation, 60 epochs, ~4h)
- [ ] GPU 7: Speech-to-Action (voice → robot, 80 epochs, ~8h)
- [ ] Model quantization (INT8) for Pi5 edge deployment

### Week 4+ (Feb 18 - Mar 14+)
- [ ] Full model export (rsync to local + S3)
- [ ] Edge deployment validation on Pi5
- [ ] Cosmos Cookoff video recording (Feb 25)
- [ ] Documentation & benchmarks
- [ ] INT8/INT4 quantization for all VLA models

---

## 🔧 Key Commands

```bash
# SSH to cluster
ssh awesome-gpu-name

# Check GPU status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv

# List tmux sessions
tmux ls

# View training logs
tail -f /data/organica-ai/logs/<job>.log

# Start new training
tmux new-session -d -s <name> 'CUDA_VISIBLE_DEVICES=<gpu> python train.py'
```

---

## 📞 NVIDIA Extension Request

**Status:** Pending (sent Feb 5, 2026)  
**Requested:** 1-2 week extension  
**Contact:** Christina Adams (dgxc-innovation-lab@nvidia.com)

**Never let H100 sit idle!** ✅
