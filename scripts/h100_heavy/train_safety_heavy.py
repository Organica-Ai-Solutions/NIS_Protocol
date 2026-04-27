#!/usr/bin/env python3
"""
Heavy Safety Classifier — Multi-Modal Action Safety Validation
Uses real BeaverTails safety dataset + real robot episodes + synthetic unsafe scenarios
Target: ~35GB VRAM, 500K steps, ~48h on H100

Architecture: Large vision encoder + action encoder + language encoder -> safety fusion head
Categories: safe, unsafe_speed, unsafe_force, unsafe_collision, unsafe_workspace, unsafe_human
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
TOTAL_STEPS = 500000
BATCH_SIZE = 128
EMBED_DIM = 1024
ENCODER_LAYERS = 16
NUM_HEADS = 16
IMAGE_SIZE = 224
NUM_SAFETY_CLASSES = 8
LR = 2e-4
WARMUP = 5000
SAVE_DIR = Path("/data/organica-ai/models/safety_heavy_v1")
LOG_DIR = Path("/data/organica-ai/logs")
DATA_DIR = Path("/data/organica-ai/datasets")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f'safety_heavy_gpu{GPU_ID}.log')])
logger = logging.getLogger(__name__)
device = torch.device("cuda")
shutdown = False
def handler(s, f):
    global shutdown; shutdown = True
signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGTERM, handler)

SAFETY_LABELS = [
    "safe",              # 0 - normal operation
    "unsafe_speed",      # 1 - joint velocity too high
    "unsafe_force",      # 2 - excessive force/torque
    "unsafe_collision",  # 3 - predicted collision
    "unsafe_workspace",  # 4 - out of workspace bounds
    "unsafe_human",      # 5 - too close to human
    "unsafe_singularity",# 6 - near kinematic singularity
    "warning",           # 7 - borderline, needs monitoring
]


class RealBeaverTailsDataset(Dataset):
    """Load real BeaverTails safety dataset — maps text safety labels to our categories"""
    def __init__(self, data_dir):
        self.samples = []
        cache_path = data_dir / "safety" / "beaver_tails_cache.pt"
        if cache_path.exists():
            try:
                data = torch.load(cache_path, map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    # Map BeaverTails categories to our safety labels
                    texts = data.get('texts', data.get('prompts', []))
                    labels = data.get('labels', data.get('is_safe', []))
                    for i in range(len(texts)):
                        text = str(texts[i]) if i < len(texts) else ""
                        label = int(labels[i]) if i < len(labels) else 0
                        # Map: 0=safe, 1=unsafe -> expand to our categories
                        if label == 0:
                            safety_label = 0  # safe
                        else:
                            safety_label = random.choice([1, 2, 3, 4, 5, 6])  # various unsafe
                        self.samples.append((text, safety_label))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text = item.get('prompt', item.get('text', ''))
                            is_safe = item.get('is_safe', True)
                            label = 0 if is_safe else random.choice([1, 2, 3, 4, 5, 6])
                            self.samples.append((str(text), label))
            except Exception as e:
                logger.warning(f"Error loading BeaverTails: {e}")
        logger.info(f"BeaverTails: {len(self.samples)} safety samples")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if not self.samples:
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(14), "safe action", 0
        idx = idx % len(self.samples)
        text, label = self.samples[idx]
        # Generate corresponding image (text-conditioned synthetic)
        img = self._text_to_scene(text, label)
        action = self._label_to_action(label)
        return img, action, text, label

    def _text_to_scene(self, text, label):
        """Generate scene image based on safety context"""
        img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if label == 0:  # safe - normal scene
            img[1] = 0.3  # greenish tint
            img += torch.randn_like(img) * 0.1 + 0.4
        elif label in [1, 2]:  # speed/force - red warning
            img[0] = 0.6
            img += torch.randn_like(img) * 0.15 + 0.2
        elif label in [3, 4]:  # collision/workspace
            img[0] = 0.5; img[1] = 0.3
            img += torch.randn_like(img) * 0.1 + 0.3
        else:  # human proximity / singularity
            img[2] = 0.5
            img += torch.randn_like(img) * 0.1 + 0.3
        # Add objects
        for _ in range(random.randint(1, 3)):
            cx, cy = random.randint(20, 204), random.randint(20, 204)
            sz = random.randint(10, 40)
            y1, y2 = max(0, cy-sz), min(224, cy+sz)
            x1, x2 = max(0, cx-sz), min(224, cx+sz)
            img[:, y1:y2, x1:x2] = torch.rand(3, 1, 1)
        return img.clamp(0, 1)

    def _label_to_action(self, label):
        """Generate action vector matching safety label"""
        action = torch.randn(14) * 0.1
        if label == 1:  # unsafe speed
            action *= 5.0  # high velocity
        elif label == 2:  # unsafe force
            action[6:] = torch.randn(8) * 3.0  # high torque
        elif label == 4:  # workspace violation
            action[:3] = torch.randn(3) * 2.0 + 2.0  # out of bounds
        return action


class RealRobotSafetyDataset(Dataset):
    """Real robot episodes labeled for safety based on action magnitudes"""
    def __init__(self, data_dir):
        self.samples = []
        for ds_name in ["xarm", "aloha", "pusht"]:
            ds_dir = data_dir / ds_name
            if not ds_dir.exists():
                continue
            for ep_dir in sorted(ds_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                for step_file in sorted(ep_dir.glob("step_*.npz")):
                    self.samples.append(str(step_file))
        logger.info(f"RealRobotSafety: {len(self.samples)} samples")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        idx = idx % len(self.samples) if self.samples else 0
        try:
            d = np.load(self.samples[idx], allow_pickle=True)
            img = torch.from_numpy(d['image'].copy()).float().permute(2, 0, 1) / 255.0
            action = torch.from_numpy(d['action'].copy()).float()
            instruction = str(d['instruction'])
            # Pad action
            padded = torch.zeros(14)
            padded[:len(action)] = action

            # Label based on action properties (real data is mostly safe)
            speed = action.abs().max().item()
            if speed > 2.0:
                label = 1  # unsafe speed
            elif speed > 1.5:
                label = 7  # warning
            else:
                label = 0  # safe

            # Augment
            if random.random() < 0.5:
                img = img + torch.randn_like(img) * random.uniform(0.01, 0.08)
            img = img.clamp(0, 1)

            return img, padded, instruction, label
        except Exception:
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.5 + 0.5, torch.zeros(14), "unknown", 0


class SyntheticSafetyDataset(Dataset):
    """Large synthetic dataset with balanced safety categories"""
    def __init__(self, num_samples=2000000):
        self.num_samples = num_samples
        self.scenarios = {
            0: ["robot moving slowly to target", "gentle grasp on soft object",
                "arm at rest in home position", "slow approach to pick location"],
            1: ["arm moving at maximum speed", "rapid joint acceleration detected",
                "fast swing toward target", "emergency speed violation"],
            2: ["excessive grip force on fragile item", "high torque on stuck joint",
                "force limit exceeded during push", "crushing force detected"],
            3: ["predicted collision with obstacle", "arm trajectory intersects table",
                "gripper path blocked by object", "imminent impact detected"],
            4: ["arm extended beyond workspace limit", "joint angle at mechanical stop",
                "end effector outside safe zone", "reaching beyond maximum radius"],
            5: ["human hand detected in workspace", "person too close to robot",
                "operator in danger zone", "human-robot proximity violation"],
            6: ["joint configuration near singularity", "wrist alignment degenerate",
                "kinematic singularity approaching", "manipulability index critical"],
            7: ["action within tolerance but borderline", "speed approaching limit",
                "workspace boundary nearby", "monitoring recommended"],
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Balanced sampling across categories
        label = idx % NUM_SAFETY_CLASSES
        scenario = random.choice(self.scenarios[label])

        # Generate scene
        img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        bg = random.uniform(0.2, 0.7)
        img += bg
        # Add visual cues based on safety level
        if label >= 1 and label <= 6:  # unsafe
            # Red warning overlay intensity proportional to danger
            danger = random.uniform(0.1, 0.4)
            img[0] += danger
        if label == 0:  # safe - green tint
            img[1] += random.uniform(0.05, 0.15)
        # Objects
        for _ in range(random.randint(1, 5)):
            cx, cy = random.randint(15, 209), random.randint(15, 209)
            sz = random.randint(8, 45)
            y1, y2 = max(0, cy-sz), min(224, cy+sz)
            x1, x2 = max(0, cx-sz), min(224, cx+sz)
            img[:, y1:y2, x1:x2] = torch.rand(3, 1, 1)
        img = img + torch.randn_like(img) * 0.05
        img = img.clamp(0, 1)

        # Action vector matching label
        action = torch.randn(14) * 0.1
        if label == 1: action *= random.uniform(3.0, 8.0)
        elif label == 2: action[6:] *= random.uniform(3.0, 6.0)
        elif label == 4: action[:3] += random.uniform(1.5, 3.0)

        return img, action, scenario, label


# ── Model ──

class VisionSafetyEncoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=16, num_heads=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
        n_patches = (IMAGE_SIZE // 14) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.ln(x[:, 0])

class ActionSafetyEncoder(nn.Module):
    def __init__(self, action_dim=14, embed_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, x):
        return self.net(x)

class TextSafetyEncoder(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=1024, depth=4, num_heads=8, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(embed_dim)
        self.vocab = {}
        words = ["robot","arm","moving","slow","fast","speed","force","grip","collision",
                 "obstacle","workspace","limit","human","hand","person","close","safe",
                 "unsafe","danger","warning","emergency","stop","joint","angle","torque",
                 "maximum","exceeded","detected","violation","approaching","critical",
                 "gentle","careful","rapid","excessive","predicted","blocked","beyond",
                 "singularity","degenerate","borderline","tolerance","monitoring",
                 "the","a","at","to","on","in","of","for","with","is","and","near"]
        for i, w in enumerate(words):
            self.vocab[w] = i + 1

    def tokenize(self, text, max_len=64):
        tokens = [self.vocab.get(w, 0) for w in text.lower().split()]
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:, :tokens.shape[1]]
        x = self.encoder(x)
        return self.ln(x.mean(dim=1))

class SafetyFusionHead(nn.Module):
    """Multi-modal fusion for safety classification"""
    def __init__(self, embed_dim=1024, num_classes=8):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 512), nn.GELU(),
            nn.Linear(512, num_classes))
        # Also predict safety score (regression)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 3, 512), nn.GELU(),
            nn.Linear(512, 1))

    def forward(self, vision_feat, action_feat, text_feat):
        fused = torch.cat([vision_feat, action_feat, text_feat], dim=-1)
        logits = self.fusion(fused)
        score = self.score_head(fused)
        return logits, score

class HeavySafetyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionSafetyEncoder(EMBED_DIM, ENCODER_LAYERS, NUM_HEADS)
        self.action = ActionSafetyEncoder(14, EMBED_DIM)
        self.text = TextSafetyEncoder(5000, EMBED_DIM, 4, 8)
        self.head = SafetyFusionHead(EMBED_DIM, NUM_SAFETY_CLASSES)

    def forward(self, images, actions, tokens):
        v = self.vision(images)
        a = self.action(actions)
        t = self.text(tokens)
        return self.head(v, a, t)


def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("HEAVY SAFETY CLASSIFIER TRAINING")
    logger.info(f"  Encoder: ViT-{ENCODER_LAYERS}L + Action MLP + Text Transformer")
    logger.info(f"  Classes: {NUM_SAFETY_CLASSES} ({', '.join(SAFETY_LABELS)})")
    logger.info(f"  Steps: {TOTAL_STEPS}, Batch: {BATCH_SIZE}")
    logger.info(f"  Data: BeaverTails + Real robot + 2M synthetic")
    logger.info("=" * 70)

    datasets = []
    bt_ds = RealBeaverTailsDataset(DATA_DIR)
    if len(bt_ds) > 1: datasets.append(bt_ds)
    robot_ds = RealRobotSafetyDataset(DATA_DIR)
    if len(robot_ds) > 1: datasets.append(robot_ds)
    synth_ds = SyntheticSafetyDataset(2000000)
    datasets.append(synth_ds)
    combined = ConcatDataset(datasets)
    logger.info(f"  Total: {len(combined)} samples")

    text_enc = TextSafetyEncoder(5000, EMBED_DIM, 4, 8)
    def collate(batch):
        imgs, acts, texts, labels = zip(*batch)
        imgs = torch.stack(imgs)
        acts = torch.stack(acts)
        tokens = torch.stack([text_enc.tokenize(t) for t in texts])
        labels = torch.tensor(labels, dtype=torch.long)
        return imgs, acts, tokens, labels

    loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=True)

    model = HeavySafetyModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    # Class weights for imbalanced data (unsafe categories are rarer in real data)
    weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()
    def lr_fn(step):
        if step < WARMUP: return step / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model.train()
    step = 0; best = float('inf'); t0 = time.time(); rloss = 0.0; rcorrect = 0; rtotal = 0

    while step < TOTAL_STEPS and not shutdown:
        for imgs, acts, tokens, labels in loader:
            if step >= TOTAL_STEPS or shutdown: break
            imgs = imgs.to(device, non_blocking=True)
            acts = acts.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Augment images on GPU
            if random.random() < 0.5:
                imgs = imgs + torch.randn_like(imgs) * random.uniform(0.01, 0.1)
                imgs = imgs.clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, score = model(imgs, acts, tokens)
                cls_loss = criterion(logits, labels)
                # Safety score target: 1.0 for safe, 0.0 for unsafe
                score_target = (labels == 0).float().unsqueeze(1)
                score_loss = F.binary_cross_entropy_with_logits(score, score_target)
                loss = cls_loss + score_loss * 0.5

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); sched.step()

            pred = logits.argmax(dim=1)
            rcorrect += (pred == labels).sum().item()
            rtotal += labels.shape[0]
            step += 1; rloss += loss.item()

            if step % 100 == 0:
                avg = rloss / 100; acc = rcorrect / max(rtotal, 1)
                elapsed = time.time() - t0
                eta = (TOTAL_STEPS - step) * (elapsed / step) / 3600
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"Step {step}/{TOTAL_STEPS} | Loss: {avg:.4f} | Acc: {acc:.3f} | "
                    f"Mem: {mem:.1f}GB | ETA: {eta:.1f}h")
                if avg < best: best = avg
                rloss = 0.0; rcorrect = 0; rtotal = 0

            if step % 25000 == 0:
                torch.save({'step': step, 'model': model.state_dict(), 'best': best,
                    'labels': SAFETY_LABELS},
                    SAVE_DIR / f"safety_heavy_step{step}.pt")

    torch.save({'step': step, 'model': model.state_dict(), 'best': best,
        'labels': SAFETY_LABELS}, SAVE_DIR / "safety_heavy_final.pt")
    h = (time.time() - t0) / 3600
    logger.info(f"COMPLETE | Steps: {step} | Best: {best:.4f} | Duration: {h:.1f}h")

if __name__ == "__main__":
    train()
