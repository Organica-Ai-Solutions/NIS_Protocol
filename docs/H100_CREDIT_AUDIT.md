# H100 Credit Audit — Smart Usage & Crash Recovery

**Audit Date:** Feb 17, 2026  
**Purpose:** Maximize value from ~28,500 GPU-hr remaining; prevent waste; fix crash recovery.

---

## 1. Crash Diagnosis

### Two Separate NIS Systems (Often Confused)

| System | Host | Port | Status | Restart Command |
|--------|------|------|--------|------------------|
| **Pi NIS** | 192.168.1.163 (Raspberry Pi) | 8000 | Often DOWN | `ssh neurolinux@192.168.1.163` → `sudo systemctl restart nis-protocol` |
| **H100 NIS** | awesome-gpu-name (DGX cluster) | 8000 | On cluster | `ssh awesome-gpu-name` → see `check_h100_status.sh` (auto-restarts) |

**Pi NIS crash:** Not controlled by H100. Must be restarted manually on the Pi. Run `python pi_status.py` to check.

**H100 NIS crash:** `check_h100_status.sh` (lines 67–73) auto-restarts `uvicorn main:app` when down. Run on cluster:
```bash
ssh awesome-gpu-name "bash -s" < check_h100_status.sh
```

### Cosmos / Reason2 URLs (Inconsistent — Fix Required)

| File | URL Used | Issue |
|------|----------|-------|
| `routes/cosmos_dance.py` | `192.168.1.160:8101` | Hardcoded; 8101 is relay, not Reason2 |
| `routes/cookoff.py` | `localhost:8100` | Assumes tunnel; OK when tunneled |
| `calibrate_and_pick.py` | `192.168.1.100:8100` | Wrong IP (100 vs 160) |
| `src/core/tool_executor.py` | `localhost:8100` + `8101` | Two endpoints; clarify roles |

**Recommendation:** Use `H100_REASON_URL` env var everywhere. Default `http://localhost:8100` (tunnel from H100).

---

## 2. Credit Waste — Critical Issues

### Issue A: Tiny Models on 80GB GPUs

From `H100_MAX_UTILIZATION_PLAN.md`:
- **Current:** 1–3 GB models → ~4% VRAM → jobs finish in 10 min–2 hr
- **Target:** 40–80 GB models → 75%+ VRAM → 12–72 hr runs

| Script | VRAM Est. | Problem |
|--------|-----------|---------|
| `h100_auto_monitor.sh` | ~2 GB | Restarts idle GPUs with `train_pinn_simple.py` — **wastes credits** |
| `launch_week3_training.sh` CLIP | ~5 GB | Small model; 2 GPUs for 6h = 12 GPU-hr for tiny model |
| `launch_week3_training.sh` Safety | ~1 GB | 512 batch, 256 hidden — finishes in ~1 hr |
| `h100_parallel_training.py` | varies | References scripts that may not exist |

### Issue B: Wrong Fallback on Idle GPUs

`h100_auto_monitor.sh` line 16:
```bash
CUDA_VISIBLE_DEVICES=$GPU nohup $VENV_PYTHON train_pinn_simple.py ...
```
This launches a **tiny** PINN model whenever a GPU goes idle. Better: launch `scripts/h100_heavy/*.py` jobs instead.

### Issue C: No Checkpoint/Resume Strategy

Some training scripts don't save intermediate checkpoints. Wasted GPU-hr if process dies.

---

## 3. Credit-Smart Recommendations

### Immediate (Do First)

1. **Stop `h100_auto_monitor.sh` from using train_pinn_simple**  
   Edit to use `launch_heavy.sh` or a queue of heavy jobs instead.

2. **Prioritize H100_MAX_UTILIZATION_PLAN jobs:**
   - Cosmos Reason2 8B full fine-tune (40 GB, 24–48h) — **Cookoff critical**
   - Qwen 72B QLoRA (4 GPUs, 48–72h)
   - Heavy VLA (50 GB, 500K steps)

3. **Run `check_h100_status.sh` daily** on the cluster to catch NIS crashes and idle GPUs.

### Short-Term

4. **Consolidate small Week 3 jobs** — Run CLIP + Safety + Speech on 1 GPU together (sequential or small multi-task) instead of 3 GPUs.

5. **Add checkpoint every N steps** to all heavy scripts:
   ```python
   if step % 50000 == 0:
       torch.save({"model": model.state_dict(), "step": step}, f"ckpt_{step}.pt")
   ```

6. **Reduce intermediate saves** — Keep only last + best. Delete epoch_1, epoch_2, etc. (already done for some models per H100_MASTER_PLAN).

### Medium-Term

7. **Use DeepSpeed ZeRO-3** for multi-GPU to fit larger models.

8. **Synthetic data gen** — Use Predict2.5 + Transfer2.5 during inference slots to generate training data (doesn't burn training credits).

---

## 4. File Audit Summary

| Category | Files | Action |
|----------|-------|--------|
| Launch | `launch_week3_training.sh`, `launch_heavy.sh` | Use heavy; avoid small models |
| Monitor | `h100_auto_monitor.sh` | **FIX:** remove train_pinn_simple fallback |
| Parallel | `h100_parallel_training.py` | Update job list to heavy scripts |
| Status | `check_h100_status.sh` | Good; use for crash recovery |
| Inference | `routes/cookoff.py`, `cosmos_dance.py` | Unify H100_REASON_URL |
| Calibration | `calibrate_and_pick.py` | Fix IP 192.168.1.100 → env var |

---

## 5. Quick Commands

```bash
# Check Pi (local network)
python pi_status.py

# Check H100 cluster
ssh awesome-gpu-name "bash -s" < check_h100_status.sh

# Or copy script and run there
scp check_h100_status.sh awesome-gpu-name:/tmp/
ssh awesome-gpu-name "bash /tmp/check_h100_status.sh"

# Launch HEAVY training (credit-efficient)
ssh awesome-gpu-name
cd ~/organica-ai && bash scripts/h100_heavy/launch_heavy.sh
```

---

**RULE:** Never run small models (train_pinn_simple, tiny CLIP, etc.) on idle H100s. Use `launch_heavy.sh` or queue large jobs.
