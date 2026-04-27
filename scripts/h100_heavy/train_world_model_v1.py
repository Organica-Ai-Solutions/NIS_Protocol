#!/usr/bin/env python3
"""
World Model — Trajectory Prediction for Sim-Before-Act
=======================================================
Predicts next robot state given current state + action.
Core for Cookoff "Sim-Before-Act" pipeline stage.

Architecture: Transformer sequence model with state-action tokens.
Input:  (state_t, action_t) pairs over a window of 16 steps
Output: predicted state_{t+1}, reward, done flag

GPU: 7 | Target VRAM: ~35GB | Steps: 150K (quality-focused)
"""

import os, sys, time, math, random, logging
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

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
        logging.FileHandler(f"/data/organica-ai/logs/world_model_v1_gpu{GPU_ID}.log"),
    ],
)
logger = logging.getLogger(__name__)

TOTAL_STEPS = 150000
BATCH_SIZE = 128
SEQ_LEN = 16
STATE_DIM = 14       # q(6) + dq(6) + gripper(1) + time(1)
ACTION_DIM = 7       # dq_cmd(6) + gripper_cmd(1)
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 8
LR = 2e-4
WARMUP_STEPS = 1500
VAL_EVERY = 500
SAVE_EVERY = 10000
PATIENCE = 5000
VAL_SPLIT = 0.1
SEED = 456

REAL_DIRS = [
    "/data/organica-ai/datasets/xarm",
    "/data/organica-ai/datasets/aloha",
    "/data/organica-ai/datasets/pusht",
]
SAVE_DIR = "/data/organica-ai/models/world_model_v1"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ── Data ────────────────────────────────────────────────────

def build_trajectories_from_episodes(dirs, max_per_dir=30000):
    """Build trajectory sequences from real robot episodes."""
    trajectories = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        npz_files = sorted(p.glob("*.npz"))[:max_per_dir]
        episode_actions = []
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                action = np.array(data.get("action", np.zeros(7)), dtype=np.float32).flatten()
                if len(action) < ACTION_DIM:
                    action = np.pad(action, (0, ACTION_DIM - len(action)))
                elif len(action) > ACTION_DIM:
                    action = action[:ACTION_DIM]
                episode_actions.append(action)
            except Exception:
                continue

        if len(episode_actions) < SEQ_LEN + 1:
            continue

        # Integrate actions into states
        q = np.zeros(6, dtype=np.float32)
        dq = np.zeros(6, dtype=np.float32)
        gripper = np.array([0.0], dtype=np.float32)
        states, actions = [], []

        for i, act in enumerate(episode_actions):
            t = np.array([i * 0.05], dtype=np.float32)
            states.append(np.concatenate([q, dq, gripper, t]))
            actions.append(act)
            dq = 0.9 * dq + 0.1 * act[:6]
            q = q + dq * 0.05
            gripper = np.clip(gripper + act[6:7] * 0.1, 0, 1)

        # Slice into windows
        for start in range(0, len(states) - SEQ_LEN - 1, SEQ_LEN // 2):
            s_seq = np.stack(states[start:start + SEQ_LEN + 1])
            a_seq = np.stack(actions[start:start + SEQ_LEN])
            trajectories.append((s_seq, a_seq))

    return trajectories


def generate_synthetic_trajectories(n=200000):
    """Generate synthetic trajectories with diverse dynamics."""
    trajectories = []
    traj_types = [
        "smooth_reach", "pick_place", "oscillate", "random_walk",
        "fast_move", "slow_precise", "spiral", "home_return",
    ]

    for _ in range(n):
        ttype = random.choice(traj_types)
        q = np.random.randn(6).astype(np.float32) * 0.5
        dq = np.zeros(6, dtype=np.float32)
        gripper = np.array([0.0], dtype=np.float32)
        states, actions = [], []

        target = np.random.randn(6).astype(np.float32)

        for step in range(SEQ_LEN + 1):
            t = np.array([step * 0.05], dtype=np.float32)
            states.append(np.concatenate([q.copy(), dq.copy(), gripper.copy(), t]))

            if step < SEQ_LEN:
                if ttype == "smooth_reach":
                    act6 = 0.3 * (target - q) + np.random.randn(6).astype(np.float32) * 0.02
                    grip = np.array([0.0])
                elif ttype == "pick_place":
                    phase = step / SEQ_LEN
                    if phase < 0.3:
                        act6 = np.array([0, 0, -0.5, 0, 0, 0], dtype=np.float32)
                        grip = np.array([0.0])
                    elif phase < 0.4:
                        act6 = np.zeros(6, dtype=np.float32)
                        grip = np.array([1.0])
                    elif phase < 0.7:
                        act6 = np.array([0.3, 0.2, 0.3, 0, 0, 0], dtype=np.float32)
                        grip = np.array([0.0])
                    else:
                        act6 = np.zeros(6, dtype=np.float32)
                        grip = np.array([-1.0])
                elif ttype == "oscillate":
                    freq = random.uniform(0.5, 2.0)
                    act6 = (np.sin(step * freq + np.arange(6)) * 0.5).astype(np.float32)
                    grip = np.array([0.0])
                elif ttype == "fast_move":
                    act6 = np.random.randn(6).astype(np.float32) * 2.0
                    grip = np.array([0.0])
                elif ttype == "slow_precise":
                    act6 = 0.05 * (target - q) + np.random.randn(6).astype(np.float32) * 0.005
                    grip = np.array([0.0])
                elif ttype == "spiral":
                    angle = step * 0.4
                    act6 = np.array([
                        np.cos(angle) * 0.3, np.sin(angle) * 0.3,
                        0.02, 0, 0, angle * 0.1
                    ], dtype=np.float32)
                    grip = np.array([0.0])
                elif ttype == "home_return":
                    act6 = -0.2 * q + np.random.randn(6).astype(np.float32) * 0.01
                    grip = np.array([-gripper[0]])
                else:
                    act6 = np.random.randn(6).astype(np.float32) * 0.5
                    grip = np.array([0.0])

                act = np.concatenate([act6, grip]).astype(np.float32)
                actions.append(act)
                dq = 0.9 * dq + 0.1 * act6
                q = q + dq * 0.05
                gripper = np.clip(gripper + grip * 0.1, 0, 1)

        trajectories.append((np.stack(states), np.stack(actions)))

    return trajectories


# ── Model ───────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Transformer world model: predicts next state from (state, action) history.
    """
    def __init__(self, state_dim=14, action_dim=7, embed_dim=768,
                 num_heads=12, num_layers=8, seq_len=16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Embeddings
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.token_type = nn.Embedding(2, embed_dim)  # 0=state, 1=action
        self.pos_embed = nn.Embedding(seq_len * 2, embed_dim)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction heads
        self.state_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, state_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):
        """
        states: (B, T+1, state_dim) — includes target at T+1
        actions: (B, T, action_dim)
        Returns: predicted next state, reward, done logit
        """
        B, T = actions.shape[:2]
        device = states.device

        # Interleave: s0, a0, s1, a1, ..., s_{T-1}, a_{T-1}
        s_emb = self.state_proj(states[:, :T])   # (B, T, E)
        a_emb = self.action_proj(actions)          # (B, T, E)

        # Build interleaved sequence
        tokens = torch.zeros(B, T * 2, self.embed_dim, device=device)
        tokens[:, 0::2] = s_emb
        tokens[:, 1::2] = a_emb

        # Add type and position embeddings
        type_ids = torch.zeros(T * 2, dtype=torch.long, device=device)
        type_ids[1::2] = 1
        pos_ids = torch.arange(T * 2, device=device)
        tokens = tokens + self.token_type(type_ids) + self.pos_embed(pos_ids)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T * 2, device=device)

        out = self.transformer(tokens, mask=mask)
        out = self.norm(out)

        # Predict from last token
        last = out[:, -1]
        pred_state = self.state_head(last)
        pred_reward = self.reward_head(last).squeeze(-1)
        pred_done = self.done_head(last).squeeze(-1)

        return pred_state, pred_reward, pred_done


# ── Training ────────────────────────────────────────────────

def prepare_batch(batch, device):
    states = torch.stack([torch.from_numpy(s.astype(np.float32)) for s, a in batch]).float().to(device)
    actions = torch.stack([torch.from_numpy(a.astype(np.float32)) for s, a in batch]).float().to(device)
    return states, actions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("WORLD MODEL v1 — Trajectory Prediction")
    logger.info(f"  GPU: {GPU_ID} | Steps: {TOTAL_STEPS} | Batch: {BATCH_SIZE}")
    logger.info(f"  Arch: {NUM_LAYERS}L Transformer, {EMBED_DIM}d, {NUM_HEADS} heads")
    logger.info(f"  Seq: {SEQ_LEN} steps | State: {STATE_DIM}d | Action: {ACTION_DIM}d")
    logger.info("=" * 70)

    logger.info("Loading real trajectories...")
    real_trajs = build_trajectories_from_episodes(REAL_DIRS)
    logger.info(f"  Real: {len(real_trajs)} trajectory windows")

    logger.info("Generating synthetic trajectories...")
    synth_trajs = generate_synthetic_trajectories(200000)
    logger.info(f"  Synthetic: {len(synth_trajs)} trajectories")

    all_data = real_trajs + synth_trajs
    random.shuffle(all_data)

    val_size = int(len(all_data) * VAL_SPLIT)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    model = WorldModel(STATE_DIM, ACTION_DIM, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler()
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

        states, actions = prepare_batch(batch, device)
        target_state = states[:, -1]  # ground truth next state

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred_state, pred_reward, pred_done = model(states, actions)
            state_loss = F.mse_loss(pred_state, target_state)
            # Reward: closer to target = higher reward
            reward_target = -torch.norm(pred_state - target_state, dim=-1).detach()
            reward_loss = F.mse_loss(pred_reward, reward_target)
            loss = state_loss + 0.1 * reward_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # LR schedule
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % 100 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.4f} "
                f"(state={state_loss.item():.4f}, reward={reward_loss.item():.4f}) | "
                f"LR: {lr:.2e} | Mem: {mem:.1f}GB"
            )

        if step % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi in range(min(50, len(val_data) // BATCH_SIZE)):
                    vb = val_data[vi * BATCH_SIZE:(vi + 1) * BATCH_SIZE]
                    vs, va = prepare_batch(vb, device)
                    vt = vs[:, -1]
                    with autocast():
                        vps, vpr, vpd = model(vs, va)
                        vl = F.mse_loss(vps, vt)
                    val_losses.append(vl.item())

            avg_val = np.mean(val_losses) if val_losses else 999
            logger.info(f"  [VAL] Step {step} | Val Loss: {avg_val:.4f}")

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

    torch.save(model.state_dict(), f"{SAVE_DIR}/final_model.pt")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
