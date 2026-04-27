#!/usr/bin/env python3
"""
Safety Classifier v2 — Quality-Focused Training
================================================
Fixes from v1:
  - BeaverTails was loading 0 samples (wrong key). Now uses proper parsing.
  - Balanced sampling: equal safe/unsafe per batch (no class collapse).
  - Label smoothing (0.1) to prevent overconfident predictions.
  - Validation split (10%) with early stopping (patience=5000 steps).
  - Cosine LR with warmup, gradient clipping.
  - Logs val_loss and val_acc every 500 steps.

GPU: 4 | Target VRAM: ~30GB | Steps: 200K (quality > quantity)
"""

import os
import sys
import time
import json
import logging
import math
import random
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/data/organica-ai/logs/safety_v2_gpu{GPU_ID}.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
TOTAL_STEPS = 300000
BATCH_SIZE = 64          # smaller batch = better generalization
EMBED_DIM = 768
NUM_CLASSES = 8          # safe, 6 unsafe types, warning
LR = 2e-4
WARMUP_STEPS = 2000
LABEL_SMOOTHING = 0.1
VAL_EVERY = 500
SAVE_EVERY = 10000
PATIENCE = 20000         # early stopping patience (in steps)
VAL_SPLIT = 0.1
SEED = 42

CLASSES = [
    "safe", "unsafe_speed", "unsafe_force", "unsafe_collision",
    "unsafe_workspace", "unsafe_human", "unsafe_singularity", "warning",
]

REAL_DIRS = [
    "/data/organica-ai/datasets/xarm",
    "/data/organica-ai/datasets/aloha",
    "/data/organica-ai/datasets/pusht",
]
SAFETY_DIR = "/data/organica-ai/datasets/safety"
SAVE_DIR = "/data/organica-ai/models/safety_v2"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ── Data ────────────────────────────────────────────────────

def load_real_robot_episodes(dirs, max_per_dir=30000):
    """Load real robot episodes and assign safety labels based on action magnitudes."""
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
                action = data["action"] if "action" in data else np.zeros(7)
                instruction = str(data["instruction"]) if "instruction" in data else "robot action"

                # Derive safety label from action magnitude
                action_mag = np.linalg.norm(action)
                if action_mag > 5.0:
                    label = random.choice([1, 2, 3])  # unsafe speed/force/collision
                elif action_mag > 3.0:
                    label = 7  # warning
                else:
                    label = 0  # safe
                samples.append((img, action, instruction, label))
            except Exception:
                continue
    return samples


def load_beavertails(safety_dir, max_samples=50000):
    """Load BeaverTails safety dataset with proper parsing."""
    samples = []
    p = Path(safety_dir)
    if not p.exists():
        logger.warning(f"Safety dir not found: {safety_dir}")
        return samples

    # Try multiple file patterns
    for pattern in ["*.jsonl", "*.json", "*.txt", "*.csv"]:
        for f in sorted(p.glob(pattern))[:10]:
            try:
                with open(f, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # BeaverTails format: {"prompt": ..., "response": ..., "is_safe": bool, "category": ...}
                        text = entry.get("prompt", entry.get("text", entry.get("response", "")))
                        is_safe = entry.get("is_safe", entry.get("safe", True))
                        cat = entry.get("category", "")

                        if is_safe:
                            label = 0
                        elif "collision" in str(cat).lower():
                            label = 3
                        elif "force" in str(cat).lower() or "harm" in str(cat).lower():
                            label = 2
                        elif "speed" in str(cat).lower():
                            label = 1
                        else:
                            label = random.choice([1, 2, 3, 4, 5, 6])

                        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        action = np.random.randn(7).astype(np.float32) * (0.5 if is_safe else 3.0)
                        samples.append((img, action, str(text)[:200], label))

                        if len(samples) >= max_samples:
                            return samples
            except Exception:
                continue

    # Also try NPZ files
    for f in sorted(p.glob("*.npz"))[:max_samples]:
        try:
            data = np.load(f, allow_pickle=True)
            img = data["image"] if "image" in data else np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            action = data["action"] if "action" in data else np.zeros(7)
            label = int(data["label"]) if "label" in data else 0
            text = str(data.get("instruction", "safety sample"))
            samples.append((img, action, text, label))
        except Exception:
            continue

    return samples


def generate_balanced_synthetic(n_per_class=25000):
    """Generate synthetic data with guaranteed class balance."""
    samples = []
    for cls_id in range(NUM_CLASSES):
        for _ in range(n_per_class):
            # Create class-distinctive features
            if cls_id == 0:  # safe
                img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 0.3
            elif cls_id == 1:  # unsafe_speed
                img = np.random.randint(50, 255, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 5.0
            elif cls_id == 2:  # unsafe_force
                img = np.random.randint(0, 150, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 4.0
                action[0] *= 3  # high force on first joint
            elif cls_id == 3:  # unsafe_collision
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                # Add "obstacle" pattern
                img[80:140, 80:140] = 255
                action = np.random.randn(7).astype(np.float32) * 2.0
            elif cls_id == 4:  # unsafe_workspace
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 2.5
                action += 3.0  # offset = out of workspace
            elif cls_id == 5:  # unsafe_human
                img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                # Add "human" blob
                img[60:180, 90:130] = np.random.randint(180, 230, (120, 40, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 1.5
            elif cls_id == 6:  # unsafe_singularity
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 0.1
                action[2] = 0.0  # near-zero = singularity
                action[3] = 0.0
            else:  # warning
                img = np.random.randint(80, 220, (224, 224, 3), dtype=np.uint8)
                action = np.random.randn(7).astype(np.float32) * 2.0

            # Add noise augmentation
            noise = np.random.randn(*img.shape) * random.uniform(5, 25)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            text = f"safety scenario class {CLASSES[cls_id]}"
            samples.append((img, action, text, cls_id))

    random.shuffle(samples)
    return samples


class BalancedSampler:
    """Ensures each batch has balanced class representation."""

    def __init__(self, samples, batch_size):
        self.batch_size = batch_size
        self.by_class = {i: [] for i in range(NUM_CLASSES)}
        for s in samples:
            self.by_class[s[3]].append(s)
        for k in self.by_class:
            random.shuffle(self.by_class[k])
        self.cursors = {i: 0 for i in range(NUM_CLASSES)}
        self.per_class = max(1, batch_size // NUM_CLASSES)

    def get_batch(self):
        batch = []
        for cls_id in range(NUM_CLASSES):
            pool = self.by_class[cls_id]
            if not pool:
                continue
            for _ in range(self.per_class):
                idx = self.cursors[cls_id] % len(pool)
                batch.append(pool[idx])
                self.cursors[cls_id] += 1
                if self.cursors[cls_id] >= len(pool):
                    random.shuffle(pool)
                    self.cursors[cls_id] = 0
        random.shuffle(batch)
        return batch[:self.batch_size]


# ── Model ───────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=768, layers=12, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
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


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=768, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(256, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos(pos)
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))


class SafetyClassifierV2(nn.Module):
    def __init__(self, embed_dim=768, num_classes=8):
        super().__init__()
        self.vision = VisionEncoder(embed_dim, layers=12)
        self.action_enc = ActionEncoder(7, embed_dim)
        self.text_enc = TextEncoder(32000, embed_dim, layers=4)

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1),
        )
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        # Binary safety score head (logits, no sigmoid)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, images, actions, token_ids):
        v = self.vision(images)
        a = self.action_enc(actions)
        t = self.text_enc(token_ids)

        # Cross-attention: vision attends to action+text
        kv = torch.stack([a, t], dim=1)
        v_attn, _ = self.cross_attn(v.unsqueeze(1), kv, kv)
        v_attn = v_attn.squeeze(1)

        fused = self.fusion(torch.cat([v_attn, a, t], dim=-1))
        logits = self.classifier(fused)
        score = self.score_head(fused).squeeze(-1)
        return logits, score


# ── Training ────────────────────────────────────────────────

def tokenize_simple(text, max_len=64):
    tokens = [ord(c) % 32000 for c in text[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return tokens


def prepare_batch(batch, device):
    images = torch.stack([
        torch.from_numpy(s[0]).permute(2, 0, 1).float() / 255.0 for s in batch
    ]).to(device)
    actions = torch.stack([
        torch.from_numpy(s[1].astype(np.float32)) if isinstance(s[1], np.ndarray)
        else torch.tensor(s[1], dtype=torch.float32) for s in batch
    ]).to(device)
    # Pad actions to 7 dims
    if actions.shape[-1] < 7:
        actions = F.pad(actions, (0, 7 - actions.shape[-1]))
    elif actions.shape[-1] > 7:
        actions = actions[:, :7]
    tokens = torch.tensor(
        [tokenize_simple(s[2]) for s in batch], dtype=torch.long
    ).to(device)
    labels = torch.tensor([s[3] for s in batch], dtype=torch.long).to(device)
    scores = (labels == 0).float().to(device)  # 1.0 = safe, 0.0 = unsafe
    return images, actions, tokens, labels, scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("SAFETY CLASSIFIER v2 — QUALITY TRAINING")
    logger.info(f"  GPU: {GPU_ID} | Steps: {TOTAL_STEPS} | Batch: {BATCH_SIZE}")
    logger.info(f"  Label smoothing: {LABEL_SMOOTHING} | Val split: {VAL_SPLIT}")
    logger.info(f"  Early stopping patience: {PATIENCE} steps")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading real robot episodes...")
    real_samples = load_real_robot_episodes(REAL_DIRS)
    logger.info(f"  Real robot: {len(real_samples)} samples")

    logger.info("Loading BeaverTails safety data...")
    bt_samples = load_beavertails(SAFETY_DIR)
    logger.info(f"  BeaverTails: {len(bt_samples)} samples")

    logger.info("Generating balanced synthetic data...")
    synth_samples = generate_balanced_synthetic(n_per_class=25000)
    logger.info(f"  Synthetic: {len(synth_samples)} samples")

    all_samples = real_samples + bt_samples + synth_samples
    random.shuffle(all_samples)

    # Verify class balance
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for s in all_samples:
        class_counts[s[3]] += 1
    logger.info("Class distribution:")
    for i, name in enumerate(CLASSES):
        logger.info(f"  {name}: {class_counts[i]}")

    # Train/val split
    val_size = int(len(all_samples) * VAL_SPLIT)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    logger.info(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Model
    model = SafetyClassifierV2(EMBED_DIM, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    train_sampler = BalancedSampler(train_samples, BATCH_SIZE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_val_loss = float("inf")
    steps_since_improve = 0

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        batch = train_sampler.get_batch()
        images, actions, tokens, labels, scores = prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, score_pred = model(images, actions, tokens)
            cls_loss = ce_loss_fn(logits, labels)
            score_loss = F.binary_cross_entropy_with_logits(score_pred, scores)
            loss = cls_loss + 0.5 * score_loss

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
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            mem = torch.cuda.memory_allocated() / 1e9
            elapsed = time.time()
            logger.info(
                f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.4f} "
                f"(cls={cls_loss.item():.4f}, score={score_loss.item():.4f}) | "
                f"Acc: {acc:.3f} | LR: {lr:.2e} | Mem: {mem:.1f}GB"
            )

        # Validation
        if step % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            val_accs = []
            val_batches = min(50, len(val_samples) // BATCH_SIZE)
            with torch.no_grad():
                for vi in range(val_batches):
                    vb = val_samples[vi * BATCH_SIZE:(vi + 1) * BATCH_SIZE]
                    if len(vb) < 2:
                        continue
                    vi_img, vi_act, vi_tok, vi_lab, vi_sc = prepare_batch(vb, device)
                    with autocast():
                        vl, vs = model(vi_img, vi_act, vi_tok)
                        v_cls = ce_loss_fn(vl, vi_lab)
                        v_score = F.binary_cross_entropy_with_logits(vs, vi_sc)
                        v_loss = v_cls + 0.5 * v_score
                    val_losses.append(v_loss.item())
                    val_accs.append((vl.argmax(dim=-1) == vi_lab).float().mean().item())

            avg_val_loss = np.mean(val_losses) if val_losses else 999
            avg_val_acc = np.mean(val_accs) if val_accs else 0
            logger.info(
                f"  [VAL] Step {step} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {avg_val_acc:.3f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                steps_since_improve = 0
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pt")
                logger.info(f"  [VAL] New best! Saved to {SAVE_DIR}/best_model.pt")
            else:
                steps_since_improve += VAL_EVERY
                if steps_since_improve >= PATIENCE:
                    logger.info(f"  [EARLY STOP] No improvement for {PATIENCE} steps. Stopping.")
                    break

        if step % SAVE_EVERY == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, f"{SAVE_DIR}/checkpoint_step{step}.pt")
            logger.info(f"  Checkpoint saved: step {step}")

    # Final save
    torch.save(model.state_dict(), f"{SAVE_DIR}/final_model.pt")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
