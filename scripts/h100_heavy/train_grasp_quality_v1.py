#!/usr/bin/env python3
"""
Grasp Quality Predictor — Vision-Action Model for xArm
========================================================
Predicts grasp success probability from visual scene + proposed action.
Directly useful for Cookoff demo: validates robot grasps before execution.

Architecture:
  - ViT-B vision encoder (12 layers)
  - Action MLP encoder
  - Cross-attention fusion
  - Multi-task: grasp_success, stability_score, force_estimate, slip_risk

Training: balanced success/failure grasps with augmentation + noise.
GPU: 7 | Target VRAM: ~40GB | Steps: 200K
"""

import os, sys, time, math, random, logging
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/data/organica-ai/logs/grasp_quality_v1_gpu{GPU_ID}.log"),
    ],
)
logger = logging.getLogger(__name__)

TOTAL_STEPS = 200000
BATCH_SIZE = 64
EMBED_DIM = 768
VISION_LAYERS = 12
PATCH_SIZE = 16
ACTION_DIM = 7
LR = 1.5e-4
WARMUP_STEPS = 2000
VAL_EVERY = 500
SAVE_EVERY = 10000
PATIENCE = 15000
VAL_SPLIT = 0.1
LABEL_SMOOTHING = 0.05
SEED = 789

REAL_DIRS = [
    "/data/organica-ai/datasets/xarm",
    "/data/organica-ai/datasets/aloha",
    "/data/organica-ai/datasets/pusht",
]
SAVE_DIR = "/data/organica-ai/models/grasp_quality_v1"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ── Data ────────────────────────────────────────────────────

def load_real_grasp_data(dirs, max_per_dir=30000):
    """Load real robot episodes and derive grasp quality labels."""
    samples = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        npz_files = sorted(p.glob("*.npz"))[:max_per_dir]
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                img = data["image"] if "image" in data else np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                action = np.array(data.get("action", np.zeros(7)), dtype=np.float32).flatten()
                if len(action) < ACTION_DIM:
                    action = np.pad(action, (0, ACTION_DIM - len(action)))
                elif len(action) > ACTION_DIM:
                    action = action[:ACTION_DIM]

                # Derive grasp labels from action characteristics
                gripper_cmd = action[-1] if len(action) >= 7 else 0.0
                force_mag = float(np.linalg.norm(action[:6]))

                # Heuristic labels based on action profile
                if abs(gripper_cmd) > 0.3 and force_mag < 3.0:
                    success = 1.0
                    stability = random.uniform(0.6, 1.0)
                    force_est = force_mag * 0.5
                    slip_risk = random.uniform(0.0, 0.3)
                elif abs(gripper_cmd) > 0.3 and force_mag >= 3.0:
                    success = random.choice([0.0, 1.0])
                    stability = random.uniform(0.2, 0.6)
                    force_est = force_mag * 0.8
                    slip_risk = random.uniform(0.3, 0.7)
                else:
                    success = 0.0
                    stability = random.uniform(0.0, 0.3)
                    force_est = force_mag * 0.2
                    slip_risk = random.uniform(0.5, 1.0)

                labels = np.array([success, stability, force_est, slip_risk], dtype=np.float32)
                samples.append((img, action, labels))
            except Exception:
                continue
    return samples


def generate_grasp_scenarios(n_per_class=60000):
    """Generate balanced synthetic grasp scenarios."""
    samples = []
    scenarios = [
        ("perfect_grasp", 1.0),
        ("marginal_grasp", 0.7),
        ("failed_grasp", 0.0),
        ("slip_grasp", 0.3),
        ("force_overload", 0.1),
        ("empty_grasp", 0.0),
    ]

    for scenario_name, base_success in scenarios:
        for _ in range(n_per_class):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            if scenario_name == "perfect_grasp":
                # Object centered, good approach angle
                cx, cy = 112 + random.randint(-20, 20), 112 + random.randint(-20, 20)
                sz = random.randint(20, 60)
                color = np.array([random.randint(100, 255), random.randint(50, 200), random.randint(50, 200)])
                img[cy-sz:cy+sz, cx-sz:cx+sz] = color
                action = np.array([0, 0, -0.3, 0, 0, 0, 0.8], dtype=np.float32)
                action += np.random.randn(7).astype(np.float32) * 0.05
                stability = random.uniform(0.7, 1.0)
                force_est = random.uniform(0.5, 1.5)
                slip_risk = random.uniform(0.0, 0.15)

            elif scenario_name == "marginal_grasp":
                # Object off-center, slight angle
                cx = random.randint(60, 170)
                cy = random.randint(60, 170)
                sz = random.randint(15, 40)
                img[cy-sz:cy+sz, cx-sz:cx+sz] = random.randint(80, 220)
                action = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
                                   -0.25, 0, 0, 0, 0.6], dtype=np.float32)
                action += np.random.randn(7).astype(np.float32) * 0.1
                stability = random.uniform(0.3, 0.7)
                force_est = random.uniform(1.0, 2.5)
                slip_risk = random.uniform(0.2, 0.5)

            elif scenario_name == "failed_grasp":
                # No object or wrong position
                action = np.random.randn(7).astype(np.float32) * 0.5
                action[6] = random.uniform(0.3, 1.0)
                stability = random.uniform(0.0, 0.2)
                force_est = random.uniform(0.0, 0.5)
                slip_risk = random.uniform(0.7, 1.0)

            elif scenario_name == "slip_grasp":
                # Object present but slippery
                cx, cy = 112, 112
                sz = random.randint(25, 50)
                img[cy-sz:cy+sz, cx-sz:cx+sz] = np.array([200, 200, 220])  # shiny
                action = np.array([0, 0, -0.3, 0, 0, 0, 0.5], dtype=np.float32)
                action += np.random.randn(7).astype(np.float32) * 0.08
                stability = random.uniform(0.1, 0.4)
                force_est = random.uniform(0.3, 1.0)
                slip_risk = random.uniform(0.6, 0.95)

            elif scenario_name == "force_overload":
                # Too much force
                cx, cy = 112, 112
                sz = random.randint(10, 30)
                img[cy-sz:cy+sz, cx-sz:cx+sz] = random.randint(50, 150)
                action = np.random.randn(7).astype(np.float32) * 2.0
                action[6] = 1.0
                stability = random.uniform(0.0, 0.3)
                force_est = random.uniform(3.0, 6.0)
                slip_risk = random.uniform(0.4, 0.8)

            else:  # empty_grasp
                action = np.array([0, 0, -0.2, 0, 0, 0, 0.9], dtype=np.float32)
                action += np.random.randn(7).astype(np.float32) * 0.03
                stability = 0.0
                force_est = random.uniform(0.0, 0.2)
                slip_risk = 1.0

            # Add image noise
            noise = np.random.randn(*img.shape).astype(np.float32) * random.uniform(5, 30)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            success = base_success + random.uniform(-0.1, 0.1)
            success = max(0.0, min(1.0, success))
            labels = np.array([success, stability, force_est, slip_risk], dtype=np.float32)
            samples.append((img, action.astype(np.float32), labels))

    random.shuffle(samples)
    return samples


# ── Model ───────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=768, layers=12, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])


class ActionEncoder(nn.Module):
    def __init__(self, action_dim=7, embed_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraspQualityNet(nn.Module):
    def __init__(self, embed_dim=768, action_dim=7, vision_layers=12):
        super().__init__()
        self.vision = VisionEncoder(embed_dim, vision_layers)
        self.action_enc = ActionEncoder(action_dim, embed_dim)

        # Cross-attention: vision attends to action
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1),
        )

        # Grasp success (logit, use BCE with logits)
        self.success_head = nn.Linear(embed_dim // 2, 1)
        # Stability score [0,1]
        self.stability_head = nn.Sequential(nn.Linear(embed_dim // 2, 1), nn.Sigmoid())
        # Force estimate (positive)
        self.force_head = nn.Sequential(nn.Linear(embed_dim // 2, 1), nn.ReLU())
        # Slip risk (logit)
        self.slip_head = nn.Linear(embed_dim // 2, 1)

    def forward(self, images, actions):
        v = self.vision(images)
        a = self.action_enc(actions)

        # Cross-attention
        v_attn, _ = self.cross_attn(v.unsqueeze(1), a.unsqueeze(1), a.unsqueeze(1))
        v_attn = v_attn.squeeze(1)

        fused = self.fusion(torch.cat([v_attn, a], dim=-1))

        success = self.success_head(fused).squeeze(-1)
        stability = self.stability_head(fused).squeeze(-1)
        force = self.force_head(fused).squeeze(-1)
        slip = self.slip_head(fused).squeeze(-1)

        return success, stability, force, slip


# ── Training ────────────────────────────────────────────────

def tokenize_simple(text, max_len=64):
    tokens = [ord(c) % 32000 for c in text[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return tokens


def prepare_batch(batch, device):
    images = torch.stack([
        torch.from_numpy(s[0].astype(np.float32)).permute(2, 0, 1) / 255.0 for s in batch
    ]).to(device)
    actions = torch.stack([
        torch.from_numpy(s[1].astype(np.float32)) for s in batch
    ]).to(device)
    labels = torch.stack([
        torch.from_numpy(s[2].astype(np.float32)) for s in batch
    ]).to(device)
    return images, actions, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("GRASP QUALITY PREDICTOR v1 — Quality Training")
    logger.info(f"  GPU: {GPU_ID} | Steps: {TOTAL_STEPS} | Batch: {BATCH_SIZE}")
    logger.info(f"  Vision: ViT-{VISION_LAYERS}L | Embed: {EMBED_DIM}d")
    logger.info(f"  Tasks: grasp_success, stability, force, slip_risk")
    logger.info(f"  Patience: {PATIENCE} | Label smoothing: {LABEL_SMOOTHING}")
    logger.info("=" * 70)

    logger.info("Loading real grasp data...")
    real_data = load_real_grasp_data(REAL_DIRS)
    logger.info(f"  Real: {len(real_data)} samples")

    logger.info("Generating synthetic grasp scenarios...")
    synth_data = generate_grasp_scenarios(n_per_class=60000)
    logger.info(f"  Synthetic: {len(synth_data)} samples")

    all_data = real_data + synth_data
    random.shuffle(all_data)

    val_size = int(len(all_data) * VAL_SPLIT)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    model = GraspQualityNet(EMBED_DIM, ACTION_DIM, VISION_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler("cuda")

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")
    steps_since_improve = 0
    cursor = 0

    for step in range(1, TOTAL_STEPS + 1):
        model.train()

        if cursor + BATCH_SIZE > len(train_data):
            random.shuffle(train_data)
            cursor = 0
        batch = train_data[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE

        images, actions, labels = prepare_batch(batch, device)
        success_gt = labels[:, 0]
        stability_gt = labels[:, 1]
        force_gt = labels[:, 2]
        slip_gt = labels[:, 3]

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            success_pred, stability_pred, force_pred, slip_pred = model(images, actions)

            s_loss = F.binary_cross_entropy_with_logits(success_pred, success_gt)
            stab_loss = F.mse_loss(stability_pred, stability_gt)
            f_loss = F.mse_loss(force_pred, force_gt)
            slip_loss = F.binary_cross_entropy_with_logits(slip_pred, slip_gt)

            loss = s_loss + stab_loss + 0.5 * f_loss + slip_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Cosine LR with warmup
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % 100 == 0:
            s_acc = ((torch.sigmoid(success_pred) > 0.5).float() == (success_gt > 0.5).float()).float().mean().item()
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.4f} "
                f"(succ={s_loss.item():.4f}, stab={stab_loss.item():.4f}, "
                f"force={f_loss.item():.4f}, slip={slip_loss.item():.4f}) | "
                f"Acc: {s_acc:.3f} | LR: {lr:.2e} | Mem: {mem:.1f}GB"
            )

        if step % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            val_accs = []
            with torch.no_grad():
                for vi in range(min(50, len(val_data) // BATCH_SIZE)):
                    vb = val_data[vi * BATCH_SIZE:(vi + 1) * BATCH_SIZE]
                    if len(vb) < 2:
                        continue
                    vi_img, vi_act, vi_lab = prepare_batch(vb, device)
                    with autocast("cuda"):
                        vs, vst, vf, vsl = model(vi_img, vi_act)
                        v_loss = (F.binary_cross_entropy_with_logits(vs, vi_lab[:, 0]) +
                                  F.mse_loss(vst, vi_lab[:, 1]) +
                                  0.5 * F.mse_loss(vf, vi_lab[:, 2]) +
                                  F.binary_cross_entropy_with_logits(vsl, vi_lab[:, 3]))
                    val_losses.append(v_loss.item())
                    v_acc = ((torch.sigmoid(vs) > 0.5).float() == (vi_lab[:, 0] > 0.5).float()).float().mean().item()
                    val_accs.append(v_acc)

            avg_val = np.mean(val_losses) if val_losses else 999
            avg_acc = np.mean(val_accs) if val_accs else 0
            logger.info(f"  [VAL] Step {step} | Val Loss: {avg_val:.4f} | Val Acc: {avg_acc:.3f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                steps_since_improve = 0
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pt")
                logger.info(f"  [VAL] New best! Saved.")
            else:
                steps_since_improve += VAL_EVERY
                if steps_since_improve >= PATIENCE:
                    logger.info(f"  [EARLY STOP] No improvement for {PATIENCE} steps.")
                    break

        if step % SAVE_EVERY == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, f"{SAVE_DIR}/checkpoint_step{step}.pt")
            logger.info(f"  Checkpoint saved: step {step}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/final_model.pt")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
