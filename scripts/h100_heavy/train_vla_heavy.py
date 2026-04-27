#!/usr/bin/env python3
"""
Heavy VLA — Multi-Dataset Robot Policy (xArm + Aloha + PushT)
Target: ~50GB VRAM, 500K steps, ~48h on H100
ViT-B vision encoder + 24-layer GPT action decoder + cross-attention
"""
import os, sys, time, signal, math, gc, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from pathlib import Path

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
TOTAL_STEPS = 500000
BATCH_SIZE = 64
EMBED_DIM = 1024
VISION_LAYERS = 12
ACTION_LAYERS = 24
NUM_HEADS = 16
IMAGE_SIZE = 224
MAX_ACTION_DIM = 14
ACTION_HORIZON = 16
LR = 1e-4
WARMUP = 5000
SAVE_DIR = Path("/data/organica-ai/models/vla_heavy_v1")
LOG_DIR = Path("/data/organica-ai/logs")
DATA_DIR = Path("/data/organica-ai/datasets")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f'vla_heavy_gpu{GPU_ID}.log')])
logger = logging.getLogger(__name__)
device = torch.device("cuda")
shutdown = False
def handler(s, f):
    global shutdown; shutdown = True; logger.info("Shutdown requested...")
signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGTERM, handler)

class RobotEpisodeDataset(Dataset):
    """Load real robot episodes with windowed action chunks"""
    def __init__(self, data_dir, dataset_name, window=16):
        self.samples = []
        self.window = window
        ds_dir = data_dir / dataset_name
        if not ds_dir.exists():
            logger.warning(f"Dataset {dataset_name} not found at {ds_dir}")
            return
        episodes = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
        for ep_dir in episodes:
            steps = sorted(ep_dir.glob("step_*.npz"))
            if len(steps) < window:
                # Use what we have, pad later
                self.samples.append([str(s) for s in steps])
            else:
                for i in range(0, len(steps) - window + 1, max(1, window // 4)):
                    self.samples.append([str(s) for s in steps[i:i+window]])
        logger.info(f"  {dataset_name}: {len(episodes)} episodes -> {len(self.samples)} windows")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if not self.samples:
            return (torch.randn(3, IMAGE_SIZE, IMAGE_SIZE),
                    "robot moving", torch.zeros(ACTION_HORIZON, MAX_ACTION_DIM))
        idx = idx % len(self.samples)
        window_files = self.samples[idx]
        images, actions, instruction = [], [], ""
        for f in window_files:
            try:
                d = np.load(f, allow_pickle=True)
                img = torch.from_numpy(d['image'].copy()).float().permute(2, 0, 1) / 255.0
                act = torch.from_numpy(d['action'].copy()).float()
                instruction = str(d['instruction'])
                images.append(img)
                # Pad action to MAX_ACTION_DIM
                padded = torch.zeros(MAX_ACTION_DIM)
                padded[:len(act)] = act
                actions.append(padded)
            except Exception:
                images.append(torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.5 + 0.5)
                actions.append(torch.zeros(MAX_ACTION_DIM))
        # Pad to ACTION_HORIZON
        while len(actions) < ACTION_HORIZON:
            actions.append(actions[-1].clone() if actions else torch.zeros(MAX_ACTION_DIM))
        actions = actions[:ACTION_HORIZON]
        # Use first image as observation
        obs_img = images[0] if images else torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
        # Augment: noise + brightness
        if random.random() < 0.5:
            obs_img = obs_img + torch.randn_like(obs_img) * random.uniform(0.01, 0.1)
        if random.random() < 0.4:
            obs_img = obs_img * random.uniform(0.7, 1.3)
        obs_img = obs_img.clamp(0, 1)
        action_tensor = torch.stack(actions)
        # Add action noise for robustness
        if random.random() < 0.3:
            action_tensor = action_tensor + torch.randn_like(action_tensor) * 0.02
        return obs_img, instruction, action_tensor

class SyntheticVLADataset(Dataset):
    """Large synthetic dataset to fill training"""
    def __init__(self, num_samples=2000000):
        self.num_samples = num_samples
        self.instructions = [
            "pick up the red cube", "place the blue sphere on the shelf",
            "push the block forward", "rotate the cylinder 90 degrees",
            "stack the blocks", "sort objects by color", "grasp the tool",
            "move arm to home position", "wave greeting gesture",
            "lift the heavy box carefully", "slide object to the left",
            "align the parts together", "inspect the surface quality",
            "pour liquid into container", "open the drawer slowly",
            "close the gripper on the bolt", "reach for the target",
            "retract arm to safe position", "calibrate joint angles",
            "follow the trajectory path", "execute pick and place cycle",
        ]
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        bg = random.choice([(0.6,0.5,0.4),(0.7,0.7,0.7),(0.3,0.3,0.3),(0.9,0.9,0.85)])
        for c in range(3): img[c] = bg[c]
        # Add objects
        for _ in range(random.randint(1, 4)):
            cx, cy = random.randint(20, 204), random.randint(20, 204)
            sz = random.randint(8, 40)
            col = (random.random(), random.random(), random.random())
            y1, y2 = max(0, cy-sz), min(224, cy+sz)
            x1, x2 = max(0, cx-sz), min(224, cx+sz)
            for c in range(3): img[c, y1:y2, x1:x2] = col[c]
        img = img + torch.randn_like(img) * 0.05
        img = img.clamp(0, 1)
        instruction = random.choice(self.instructions)
        # Synthetic smooth trajectory
        t = torch.linspace(0, 1, ACTION_HORIZON).unsqueeze(1)
        start = torch.randn(1, MAX_ACTION_DIM) * 0.3
        end = torch.randn(1, MAX_ACTION_DIM) * 0.3
        actions = start * (1 - t) + end * t + torch.randn(ACTION_HORIZON, MAX_ACTION_DIM) * 0.01
        return img, instruction, actions

# ── Model ──
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, embed_dim=1024):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=12, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbed(IMAGE_SIZE, 14, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.ln(x)

class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=1024, depth=6, num_heads=16, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(embed_dim)
        self.vocab = {}
        words = ["pick","up","the","red","blue","green","cube","sphere","block",
                 "place","push","pull","rotate","lift","lower","grasp","release",
                 "move","arm","to","home","position","on","shelf","forward","left",
                 "right","slowly","carefully","sort","stack","objects","by","color",
                 "heavy","box","tool","gripper","close","open","drawer","reach",
                 "target","retract","safe","calibrate","joint","follow","trajectory",
                 "execute","and","cycle","a","an","from","into","degrees","90"]
        for i, w in enumerate(words):
            self.vocab[w] = i + 1
    def tokenize(self, text, max_len=64):
        tokens = [self.vocab.get(w, 0) for w in text.lower().split()]
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:, :tokens.shape[1]]
        x = self.encoder(x)
        return self.ln(x)

class ActionDecoder(nn.Module):
    """GPT-style action decoder with cross-attention to vision+language"""
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, action_dim=14, horizon=16):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, horizon, embed_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True),
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(0.1)),
                'ln1': nn.LayerNorm(embed_dim),
                'ln2': nn.LayerNorm(embed_dim),
                'ln3': nn.LayerNorm(embed_dim),
            }))
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, action_dim)
    def forward(self, action_tokens, context):
        x = self.action_embed(action_tokens) + self.pos[:, :action_tokens.shape[1]]
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), 1).bool()
        for layer in self.layers:
            # Self attention (causal)
            res = x
            x = layer['ln1'](x)
            x, _ = layer['self_attn'](x, x, x, attn_mask=causal)
            x = x + res
            # Cross attention to vision+language
            res = x
            x = layer['ln2'](x)
            x, _ = layer['cross_attn'](x, context, context)
            x = x + res
            # FFN
            res = x
            x = layer['ln3'](x)
            x = layer['ffn'](x) + res
        return self.head(self.ln_out(x))

class HeavyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder(EMBED_DIM, VISION_LAYERS, NUM_HEADS)
        self.language = InstructionEncoder(10000, EMBED_DIM, 6, NUM_HEADS)
        self.decoder = ActionDecoder(EMBED_DIM, ACTION_LAYERS, NUM_HEADS, MAX_ACTION_DIM, ACTION_HORIZON)
    def forward(self, images, tokens, actions_in):
        vis = self.vision(images)
        lang = self.language(tokens)
        context = torch.cat([vis, lang], dim=1)
        pred = self.decoder(actions_in, context)
        return pred

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("HEAVY VLA TRAINING — Multi-Dataset Robot Policy")
    logger.info(f"  Vision: ViT-B/{VISION_LAYERS}L, Decoder: GPT-{ACTION_LAYERS}L")
    logger.info(f"  Embed: {EMBED_DIM}, Heads: {NUM_HEADS}, Horizon: {ACTION_HORIZON}")
    logger.info(f"  Steps: {TOTAL_STEPS}, Batch: {BATCH_SIZE}, LR: {LR}")
    logger.info("=" * 70)

    # Datasets
    datasets = []
    for name in ["xarm", "aloha", "pusht"]:
        ds = RobotEpisodeDataset(DATA_DIR, name, ACTION_HORIZON)
        if len(ds) > 1:
            datasets.append(ds)
    synth = SyntheticVLADataset(2000000)
    datasets.append(synth)
    combined = ConcatDataset(datasets)
    logger.info(f"  Total dataset: {len(combined)} samples")

    lang_enc = InstructionEncoder(10000, EMBED_DIM, 6, NUM_HEADS)
    def collate(batch):
        imgs, insts, acts = zip(*batch)
        imgs = torch.stack(imgs)
        tokens = torch.stack([lang_enc.tokenize(i) for i in insts])
        acts = torch.stack(acts)
        return imgs, tokens, acts

    loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=True)

    model = HeavyVLA().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05, betas=(0.9, 0.95))
    scaler = GradScaler()
    def lr_fn(step):
        if step < WARMUP: return step / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model.train()
    step = 0; best = float('inf'); t0 = time.time(); rloss = 0.0

    while step < TOTAL_STEPS and not shutdown:
        for imgs, tokens, actions in loader:
            if step >= TOTAL_STEPS or shutdown: break
            imgs = imgs.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            # Teacher forcing: input is shifted actions
            actions_in = torch.zeros_like(actions)
            actions_in[:, 1:] = actions[:, :-1]

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(imgs, tokens, actions_in)
                loss = F.mse_loss(pred, actions) + F.l1_loss(pred, actions) * 0.1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); sched.step()
            step += 1; rloss += loss.item()

            if step % 100 == 0:
                avg = rloss / 100; elapsed = time.time() - t0
                eta = (TOTAL_STEPS - step) * (elapsed / step) / 3600
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"Step {step}/{TOTAL_STEPS} | Loss: {avg:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Mem: {mem:.1f}GB | ETA: {eta:.1f}h")
                if avg < best: best = avg
                rloss = 0.0
            if step % 25000 == 0:
                torch.save({'step': step, 'model': model.state_dict(), 'best': best},
                    SAVE_DIR / f"vla_heavy_step{step}.pt")
                logger.info(f"Checkpoint: step {step}")

    torch.save({'step': step, 'model': model.state_dict(), 'best': best},
        SAVE_DIR / "vla_heavy_final.pt")
    h = (time.time() - t0) / 3600
    logger.info(f"COMPLETE | Steps: {step} | Best: {best:.6f} | Duration: {h:.1f}h")

if __name__ == "__main__":
    train()
