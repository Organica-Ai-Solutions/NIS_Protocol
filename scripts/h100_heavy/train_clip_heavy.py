#!/usr/bin/env python3
"""
Heavy Robotics CLIP — Vision-Language Alignment for Robot Manipulation
Uses CIFAR-100 real images + xArm episode images + rich synthetic augmentation
Target: ~40GB VRAM, 500K steps, ~48h on H100

Architecture: ViT-L/14 vision encoder + 12-layer text transformer
Data: Real CIFAR-100 + real xArm images + synthetic augmentation (noise, blur, crop, color jitter)
"""
import os, sys, time, signal, math, gc, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from pathlib import Path

# ── Config ──
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
TOTAL_STEPS = 500000
BATCH_SIZE = 64
EMBED_DIM = 1024
VISION_LAYERS = 16
TEXT_LAYERS = 12
NUM_HEADS = 16
IMAGE_SIZE = 224
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 77
LR = 3e-4
WARMUP_STEPS = 5000
SAVE_DIR = Path("/data/organica-ai/models/clip_heavy_v1")
LOG_DIR = Path("/data/organica-ai/logs")
DATA_DIR = Path("/data/organica-ai/datasets")
CHECKPOINT_EVERY = 25000

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f'clip_heavy_gpu{GPU_ID}.log')
    ]
)
logger = logging.getLogger(__name__)
device = torch.device("cuda")

shutdown = False
def handler(sig, frame):
    global shutdown
    shutdown = True
    logger.info("Graceful shutdown requested...")
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


# ══════════════════════════════════════════════════════════════════════
# DATA: Real + Synthetic with Heavy Augmentation
# ══════════════════════════════════════════════════════════════════════

def apply_augmentation(image_tensor):
    """Apply realistic augmentation: noise, blur, crop, color jitter, cutout"""
    B, C, H, W = image_tensor.shape

    # Random Gaussian noise (sensor noise simulation)
    if random.random() < 0.5:
        noise_std = random.uniform(0.01, 0.15)
        image_tensor = image_tensor + torch.randn_like(image_tensor) * noise_std

    # Random brightness/contrast (lighting variation)
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.7, 1.3)
        mean = image_tensor.mean(dim=(2, 3), keepdim=True)
        image_tensor = (image_tensor - mean) * contrast + mean * brightness

    # Random horizontal flip
    if random.random() < 0.5:
        image_tensor = image_tensor.flip(3)

    # Random erasing / cutout (occlusion simulation)
    if random.random() < 0.3:
        eh = random.randint(H // 8, H // 3)
        ew = random.randint(W // 8, W // 3)
        y = random.randint(0, H - eh)
        x = random.randint(0, W - ew)
        image_tensor[:, :, y:y+eh, x:x+ew] = torch.randn(B, C, eh, ew, device=image_tensor.device) * 0.5

    # Gaussian blur (camera defocus)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5, 7])
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image_tensor.device) / (kernel_size ** 2)
        for c in range(C):
            image_tensor[:, c:c+1] = F.conv2d(image_tensor[:, c:c+1], kernel, padding=pad)

    return image_tensor.clamp(0, 1)


class RealXArmDataset(Dataset):
    """Load real xArm episode images with instructions"""
    def __init__(self, data_dir, image_size=224):
        self.samples = []
        self.image_size = image_size
        xarm_dir = data_dir / "xarm"
        if xarm_dir.exists():
            episodes = sorted(xarm_dir.iterdir())
            for ep_dir in episodes:
                if not ep_dir.is_dir():
                    continue
                steps = sorted(ep_dir.glob("step_*.npz"))
                for step_file in steps:
                    self.samples.append(str(step_file))
        logger.info(f"RealXArmDataset: {len(self.samples)} samples from {xarm_dir}")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        idx = idx % len(self.samples) if self.samples else 0
        try:
            data = np.load(self.samples[idx], allow_pickle=True)
            image = data['image']  # (224, 224, 3) uint8
            instruction = str(data['instruction'])

            # Normalize to [0, 1] float
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            # Resize if needed
            if image.shape[1] != self.image_size:
                image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode='bilinear').squeeze(0)

            return image, instruction
        except Exception:
            # Fallback to synthetic
            return torch.randn(3, self.image_size, self.image_size) * 0.5 + 0.5, "robot arm moving"


class RealCIFAR100Dataset(Dataset):
    """Load real CIFAR-100 images with class labels as captions"""
    def __init__(self, data_dir, image_size=224):
        self.image_size = image_size
        self.images = []
        self.labels = []

        cifar_dir = data_dir / "clip" / "cifar-100-python"
        if cifar_dir.exists():
            import pickle
            for split in ['train', 'test']:
                fpath = cifar_dir / split
                if fpath.exists():
                    with open(fpath, 'rb') as f:
                        d = pickle.load(f, encoding='bytes')
                    imgs = d[b'data']  # (N, 3072)
                    lbls = d.get(b'fine_labels', d.get(b'coarse_labels', [0]*len(imgs)))
                    self.images.append(imgs)
                    self.labels.extend(lbls)
            if self.images:
                self.images = np.concatenate(self.images, axis=0)
            else:
                self.images = np.zeros((1, 3072), dtype=np.uint8)

        # CIFAR-100 fine label names (subset)
        self.label_names = [
            "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
            "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
            "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
            "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
            "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
            "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree",
            "pear", "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
            "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
            "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
            "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank",
            "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
            "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
        ]
        # Robotics-relevant caption templates
        self.templates = [
            "a photo of a {}",
            "a robot observing a {}",
            "robot camera view of a {}",
            "manipulation target: {}",
            "scene containing a {}",
            "a {} on a table in a robotics lab",
            "robot arm reaching for a {}",
        ]
        logger.info(f"RealCIFAR100Dataset: {len(self.images)} images loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(3, 32, 32)
        img = torch.from_numpy(img.copy()).float() / 255.0
        # Upscale to 224x224
        img = F.interpolate(img.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)

        label_idx = self.labels[idx] if idx < len(self.labels) else 0
        label_name = self.label_names[label_idx % len(self.label_names)]
        template = random.choice(self.templates)
        caption = template.format(label_name)

        return img, caption


class SyntheticRoboticsDataset(Dataset):
    """Large synthetic dataset with realistic robot scene generation"""
    def __init__(self, num_samples=2000000, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size

        # Rich robotics vocabulary
        self.objects = [
            "red cube", "blue sphere", "green cylinder", "yellow cone", "orange block",
            "purple ring", "white box", "black rod", "silver bolt", "gold nut",
            "wooden block", "plastic bottle", "metal can", "rubber ball", "glass jar",
            "foam pad", "cardboard box", "steel plate", "copper wire", "ceramic mug",
            "screwdriver", "wrench", "pliers", "hammer", "tape measure",
        ]
        self.surfaces = [
            "wooden table", "metal workbench", "conveyor belt", "foam mat",
            "glass surface", "rubber pad", "steel shelf", "plastic tray",
        ]
        self.actions = [
            "picking up", "placing down", "pushing", "pulling", "rotating",
            "stacking", "sorting", "inspecting", "grasping", "releasing",
            "lifting", "lowering", "sliding", "flipping", "aligning",
        ]
        self.robots = [
            "xArm 1S", "robot arm", "6-DOF manipulator", "gripper",
            "robotic hand", "servo arm", "articulated arm",
        ]
        self.conditions = [
            "under bright lighting", "in dim conditions", "with shadows",
            "from overhead view", "from side angle", "close-up view",
            "with motion blur", "in cluttered scene", "on clean surface",
        ]

    def __len__(self):
        return self.num_samples

    def _generate_scene_image(self):
        """Generate a synthetic scene with geometric shapes on backgrounds"""
        img = torch.zeros(3, self.image_size, self.image_size)

        # Random background (simulate table/surface colors)
        bg_colors = [
            (0.6, 0.5, 0.4),  # wood
            (0.7, 0.7, 0.7),  # metal
            (0.3, 0.3, 0.3),  # dark
            (0.9, 0.9, 0.85), # white
            (0.4, 0.5, 0.4),  # green mat
        ]
        bg = random.choice(bg_colors)
        for c in range(3):
            img[c] = bg[c]

        # Add gradient (lighting simulation)
        gradient = torch.linspace(0.8, 1.2, self.image_size).unsqueeze(0).expand(self.image_size, -1)
        img = img * gradient.unsqueeze(0)

        # Add 1-5 random objects (colored rectangles/circles)
        num_objects = random.randint(1, 5)
        for _ in range(num_objects):
            obj_color = (random.random(), random.random(), random.random())
            cx = random.randint(30, self.image_size - 30)
            cy = random.randint(30, self.image_size - 30)
            size = random.randint(10, 50)

            if random.random() < 0.5:
                # Rectangle
                x1 = max(0, cx - size)
                x2 = min(self.image_size, cx + size)
                y1 = max(0, cy - size)
                y2 = min(self.image_size, cy + size)
                for c in range(3):
                    img[c, y1:y2, x1:x2] = obj_color[c]
            else:
                # Circle
                yy, xx = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < size ** 2
                for c in range(3):
                    img[c][mask] = obj_color[c]

        # Add texture noise (sensor noise)
        img = img + torch.randn_like(img) * 0.03

        return img.clamp(0, 1)

    def __getitem__(self, idx):
        image = self._generate_scene_image()

        # Generate rich caption
        action = random.choice(self.actions)
        obj = random.choice(self.objects)
        surface = random.choice(self.surfaces)
        robot = random.choice(self.robots)
        condition = random.choice(self.conditions)

        templates = [
            f"{robot} {action} the {obj} from the {surface}",
            f"{action} {obj} {condition}",
            f"{robot} {action} {obj} and placing on {surface}",
            f"the {obj} on {surface} {condition}",
            f"{robot} performing {action} task with {obj}",
            f"scene: {obj} on {surface}, {robot} {action} {condition}",
        ]
        caption = random.choice(templates)

        return image, caption


class SimpleTokenizer:
    """Byte-pair-style tokenizer with robotics vocabulary"""
    def __init__(self, vocab_size=32000, max_len=77):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self._build_vocab()

    def _build_vocab(self):
        # Build vocab from common robotics + general words
        words = set()
        robotics_words = [
            "robot", "arm", "gripper", "servo", "motor", "joint", "end", "effector",
            "pick", "place", "push", "pull", "rotate", "lift", "lower", "grasp",
            "release", "move", "reach", "approach", "retract", "home", "calibrate",
            "red", "blue", "green", "yellow", "orange", "purple", "white", "black",
            "cube", "sphere", "cylinder", "cone", "block", "ring", "box", "rod",
            "table", "surface", "shelf", "tray", "mat", "bench", "conveyor",
            "left", "right", "up", "down", "forward", "backward", "slow", "fast",
            "camera", "sensor", "view", "scene", "target", "object", "position",
            "the", "a", "an", "on", "in", "from", "to", "with", "and", "of",
            "is", "are", "was", "for", "at", "by", "this", "that", "it",
            "photo", "image", "picture", "observation", "frame", "manipulation",
            "performing", "task", "action", "command", "instruction", "step",
        ]
        for w in robotics_words:
            words.add(w)

        # Add character-level fallback
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            words.add(c)

        for i, w in enumerate(sorted(words)):
            if len(self.word2idx) < self.vocab_size:
                self.word2idx[w] = len(self.word2idx)

    def encode(self, text):
        tokens = [self.word2idx.get("<sos>", 1)]
        for word in text.lower().split():
            tokens.append(self.word2idx.get(word, self.word2idx["<unk>"]))
        tokens.append(self.word2idx.get("<eos>", 2))
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
# MODEL: ViT-L/14 Vision Encoder + 12-Layer Text Transformer
# ══════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """ViT-L/14 style vision encoder"""
    def __init__(self, img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.ln_pre = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0])
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class TextTransformer(nn.Module):
    """12-layer text transformer encoder"""
    def __init__(self, vocab_size=32000, embed_dim=1024, depth=12, num_heads=16, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.ln_pre = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, tokens):
        x = self.token_embed(tokens)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.ln_pre(x)

        # Causal mask for autoregressive text
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        # Use EOS token position (last non-pad)
        x = self.ln_post(x)
        # Take features at the last position
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class RoboticsCLIP(nn.Module):
    """Full CLIP model with learned temperature"""
    def __init__(self, embed_dim=1024, vision_layers=24, text_layers=12, num_heads=16,
                 vocab_size=32000, image_size=224, max_len=77):
        super().__init__()
        self.visual = VisionTransformer(image_size, 14, embed_dim, vision_layers, num_heads)
        self.text = TextTransformer(vocab_size, embed_dim, text_layers, num_heads, max_len)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, tokens):
        image_features = self.visual(images)
        text_features = self.text(tokens)
        return image_features, text_features, self.logit_scale.exp()


def clip_loss(image_features, text_features, logit_scale):
    """Symmetric contrastive loss (InfoNCE)"""
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(image_features), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HEAVY ROBOTICS CLIP TRAINING")
    logger.info(f"  Model: ViT-L/14 ({VISION_LAYERS}L vision + {TEXT_LAYERS}L text)")
    logger.info(f"  Embed dim: {EMBED_DIM}, Heads: {NUM_HEADS}")
    logger.info(f"  Steps: {TOTAL_STEPS}, Batch: {BATCH_SIZE}, LR: {LR}")
    logger.info(f"  Data: Real xArm + CIFAR-100 + 2M synthetic")
    logger.info("=" * 70)

    # Build datasets
    logger.info("Loading datasets...")
    datasets = []

    xarm_ds = RealXArmDataset(DATA_DIR, IMAGE_SIZE)
    if len(xarm_ds) > 0:
        datasets.append(xarm_ds)
        logger.info(f"  Real xArm: {len(xarm_ds)} samples")

    cifar_ds = RealCIFAR100Dataset(DATA_DIR, IMAGE_SIZE)
    if len(cifar_ds) > 0:
        datasets.append(cifar_ds)
        logger.info(f"  Real CIFAR-100: {len(cifar_ds)} samples")

    synth_ds = SyntheticRoboticsDataset(2000000, IMAGE_SIZE)
    datasets.append(synth_ds)
    logger.info(f"  Synthetic: {len(synth_ds)} samples")

    combined = ConcatDataset(datasets)
    logger.info(f"  Total: {len(combined)} samples")

    tokenizer = SimpleTokenizer(VOCAB_SIZE, MAX_SEQ_LEN)

    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        tokens = torch.stack([tokenizer.encode(c) for c in captions])
        return images, tokens

    loader = DataLoader(
        combined, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=True
    )

    # Build model
    model = RoboticsCLIP(
        embed_dim=EMBED_DIM, vision_layers=VISION_LAYERS, text_layers=TEXT_LAYERS,
        num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE, image_size=IMAGE_SIZE, max_len=MAX_SEQ_LEN
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    logger.info(f"  VRAM estimate: ~{param_count * 4 / 1e9 * 3:.1f} GB (params + grads + optimizer)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.98))
    scaler = GradScaler()

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    start_time = time.time()
    running_loss = 0.0

    logger.info("Starting training...")

    while step < TOTAL_STEPS and not shutdown:
        for images, tokens in loader:
            if step >= TOTAL_STEPS or shutdown:
                break

            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            # Apply augmentation on GPU
            images = apply_augmentation(images)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                image_feat, text_feat, logit_scale = model(images, tokens)
                loss = clip_loss(image_feat, text_feat, logit_scale)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running_loss += loss.item()

            if step % 100 == 0:
                avg_loss = running_loss / 100
                elapsed = time.time() - start_time
                eta = (TOTAL_STEPS - step) * (elapsed / step) / 3600
                lr = optimizer.param_groups[0]['lr']
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"Step {step}/{TOTAL_STEPS} | Loss: {avg_loss:.4f} | "
                    f"Scale: {logit_scale.item():.2f} | LR: {lr:.2e} | "
                    f"Mem: {mem:.1f}GB | ETA: {eta:.1f}h"
                )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                running_loss = 0.0

            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = SAVE_DIR / f"clip_heavy_step{step}.pt"
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_loss': best_loss,
                    'config': {
                        'embed_dim': EMBED_DIM, 'vision_layers': VISION_LAYERS,
                        'text_layers': TEXT_LAYERS, 'num_heads': NUM_HEADS,
                        'vocab_size': VOCAB_SIZE, 'image_size': IMAGE_SIZE,
                    }
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = SAVE_DIR / "clip_heavy_final.pt"
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'best_loss': best_loss,
        'config': {
            'embed_dim': EMBED_DIM, 'vision_layers': VISION_LAYERS,
            'text_layers': TEXT_LAYERS, 'num_heads': NUM_HEADS,
            'vocab_size': VOCAB_SIZE, 'image_size': IMAGE_SIZE,
        }
    }, final_path)

    elapsed_h = (time.time() - start_time) / 3600
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Steps: {step}/{TOTAL_STEPS}")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"  Duration: {elapsed_h:.1f}h")
    logger.info(f"  Model: {final_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    train()
