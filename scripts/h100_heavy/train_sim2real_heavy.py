#!/usr/bin/env python3
"""
Heavy Sim2Real Domain Adaptation — Adversarial + Contrastive
Uses real xArm/Aloha/PushT images as "real" domain, synthetic scenes as "sim" domain
Target: ~40GB VRAM, 500K steps, ~48h on H100

Architecture: Large ResNet-50 feature extractor + domain discriminator + task head
Method: DANN (Domain-Adversarial Neural Network) + contrastive alignment
"""
import os, sys, time, signal, math, gc, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Function
import numpy as np
import logging
from pathlib import Path

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
TOTAL_STEPS = 500000
BATCH_SIZE = 128
EMBED_DIM = 2048
BACKBONE_DIM = 1024
NUM_CLASSES = 25  # object categories
IMAGE_SIZE = 224
LR = 3e-4
WARMUP = 5000
SAVE_DIR = Path("/data/organica-ai/models/sim2real_heavy_v1")
LOG_DIR = Path("/data/organica-ai/logs")
DATA_DIR = Path("/data/organica-ai/datasets")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f'sim2real_heavy_gpu{GPU_ID}.log')])
logger = logging.getLogger(__name__)
device = torch.device("cuda")
shutdown = False
def handler(s, f):
    global shutdown; shutdown = True
signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGTERM, handler)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class RealDomainDataset(Dataset):
    """Real robot images from xArm + Aloha + PushT episodes"""
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
                    self.samples.append((str(step_file), ds_name))
        logger.info(f"RealDomain: {len(self.samples)} images from xarm/aloha/pusht")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        idx = idx % len(self.samples) if self.samples else 0
        try:
            path, ds_name = self.samples[idx]
            d = np.load(path, allow_pickle=True)
            img = torch.from_numpy(d['image'].copy()).float().permute(2, 0, 1) / 255.0
            action = torch.from_numpy(d['action'].copy()).float()
            # Augment real images (mild — preserve distribution)
            if random.random() < 0.3:
                img = img + torch.randn_like(img) * 0.02
            if random.random() < 0.3:
                img = img * random.uniform(0.9, 1.1)
            img = img.clamp(0, 1)
            # Task label from action magnitude
            label = min(int(action.abs().sum().item() * 5), NUM_CLASSES - 1)
            return img, label, 1  # domain=1 for real
        except Exception:
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.5 + 0.5, 0, 1


class SimDomainDataset(Dataset):
    """Synthetic simulation images with heavy augmentation"""
    def __init__(self, num_samples=2000000):
        self.num_samples = num_samples
        self.object_colors = [
            (0.9, 0.2, 0.2), (0.2, 0.2, 0.9), (0.2, 0.8, 0.2), (0.9, 0.9, 0.2),
            (0.9, 0.5, 0.1), (0.7, 0.2, 0.9), (0.9, 0.9, 0.9), (0.1, 0.1, 0.1),
        ]
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        # Sim-style backgrounds (flat colors, gradients, checkerboard)
        bg_type = random.choice(['flat', 'gradient', 'checker', 'noise'])
        if bg_type == 'flat':
            c = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
            for ch in range(3): img[ch] = c[ch]
        elif bg_type == 'gradient':
            for ch in range(3):
                img[ch] = torch.linspace(random.uniform(0.1, 0.5), random.uniform(0.5, 0.9), IMAGE_SIZE).unsqueeze(0)
        elif bg_type == 'checker':
            sz = random.choice([8, 16, 32])
            for y in range(0, IMAGE_SIZE, sz):
                for x in range(0, IMAGE_SIZE, sz):
                    if ((y // sz) + (x // sz)) % 2 == 0:
                        c = random.uniform(0.3, 0.7)
                        img[:, y:y+sz, x:x+sz] = c
        else:
            img = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE) * 0.3 + 0.3

        # Add sim objects (perfect geometric shapes — the "sim gap")
        num_obj = random.randint(1, 6)
        label = 0
        for i in range(num_obj):
            col = random.choice(self.object_colors)
            cx, cy = random.randint(20, 204), random.randint(20, 204)
            sz = random.randint(10, 50)
            shape = random.choice(['rect', 'circle', 'triangle'])
            if shape == 'rect':
                y1, y2 = max(0, cy-sz), min(224, cy+sz)
                x1, x2 = max(0, cx-sz), min(224, cx+sz)
                for ch in range(3): img[ch, y1:y2, x1:x2] = col[ch]
            elif shape == 'circle':
                yy, xx = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
                mask = ((xx - cx)**2 + (yy - cy)**2) < sz**2
                for ch in range(3): img[ch][mask] = col[ch]
            else:  # triangle
                for dy in range(sz*2):
                    w = int(sz * (1 - dy / (sz*2)))
                    y = cy - sz + dy
                    if 0 <= y < 224:
                        x1 = max(0, cx - w)
                        x2 = min(224, cx + w)
                        for ch in range(3): img[ch, y, x1:x2] = col[ch]
            label = max(label, i)

        # Sim-specific artifacts: perfect edges, no noise, uniform lighting
        # (This IS the domain gap we want to bridge)
        label = min(label, NUM_CLASSES - 1)
        return img.clamp(0, 1), label, 0  # domain=0 for sim


# ── Model: Large Feature Extractor + Domain Discriminator + Task Head ──

class LargeResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.gelu(out + self.shortcut(x))

class FeatureExtractor(nn.Module):
    """ResNet-50 style backbone"""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(3, 2, 1))
        # ResNet-50 stages
        self.layer1 = self._make_layer(128, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(2048, BACKBONE_DIM), nn.LayerNorm(BACKBONE_DIM))

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [LargeResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(LargeResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

class DomainDiscriminator(nn.Module):
    """Multi-layer domain classifier with gradient reversal"""
    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(2048, 2048), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 1))
    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.net(x)

class TaskHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, num_classes))
    def forward(self, x):
        return self.net(x)

class Sim2RealModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FeatureExtractor()
        self.domain_disc = DomainDiscriminator(BACKBONE_DIM)
        self.task_head = TaskHead(BACKBONE_DIM, NUM_CLASSES)
        # Contrastive projection
        self.contrastive_proj = nn.Sequential(
            nn.Linear(BACKBONE_DIM, 512), nn.GELU(), nn.Linear(512, 256))
    def forward(self, x, alpha=1.0):
        feat = self.backbone(x)
        domain_out = self.domain_disc(feat, alpha)
        task_out = self.task_head(feat)
        contrast = F.normalize(self.contrastive_proj(feat), dim=-1)
        return feat, domain_out, task_out, contrast


def contrastive_loss(sim_feat, real_feat, temperature=0.1):
    """NT-Xent contrastive loss to align sim and real features"""
    batch = min(sim_feat.shape[0], real_feat.shape[0])
    sim_feat = sim_feat[:batch]
    real_feat = real_feat[:batch]
    # Positive pairs: same index sim-real
    logits = sim_feat @ real_feat.t() / temperature
    labels = torch.arange(batch, device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("HEAVY SIM2REAL DOMAIN ADAPTATION")
    logger.info(f"  Backbone: ResNet-50 ({BACKBONE_DIM}D)")
    logger.info(f"  Method: DANN + Contrastive alignment")
    logger.info(f"  Steps: {TOTAL_STEPS}, Batch: {BATCH_SIZE}, LR: {LR}")
    logger.info(f"  Data: Real robot images + 2M synthetic sim images")
    logger.info("=" * 70)

    real_ds = RealDomainDataset(DATA_DIR)
    sim_ds = SimDomainDataset(2000000)
    logger.info(f"  Real: {len(real_ds)}, Sim: {len(sim_ds)}")

    real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE//2, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    sim_loader = DataLoader(sim_ds, batch_size=BATCH_SIZE//2, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    model = Sim2RealModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()
    def lr_fn(step):
        if step < WARMUP: return step / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model.train()
    step = 0; best = float('inf'); t0 = time.time(); rloss = 0.0
    real_iter = iter(real_loader)

    while step < TOTAL_STEPS and not shutdown:
        for sim_imgs, sim_labels, sim_domains in sim_loader:
            if step >= TOTAL_STEPS or shutdown: break

            # Get real batch (cycle)
            try:
                real_imgs, real_labels, real_domains = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_imgs, real_labels, real_domains = next(real_iter)

            # DANN alpha schedule: ramp from 0 to 1
            p = step / TOTAL_STEPS
            alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

            sim_imgs = sim_imgs.to(device, non_blocking=True)
            real_imgs = real_imgs.to(device, non_blocking=True)
            sim_labels = sim_labels.to(device, non_blocking=True)
            real_labels = real_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                # Forward both domains
                _, sim_dom, sim_task, sim_cont = model(sim_imgs, alpha)
                _, real_dom, real_task, real_cont = model(real_imgs, alpha)

                # Task loss (both domains)
                task_loss = (F.cross_entropy(sim_task, sim_labels) +
                           F.cross_entropy(real_task, real_labels)) / 2

                # Domain loss (binary: sim=0, real=1)
                dom_labels_sim = torch.zeros(sim_dom.shape[0], 1, device=device)
                dom_labels_real = torch.ones(real_dom.shape[0], 1, device=device)
                domain_loss = (F.binary_cross_entropy_with_logits(sim_dom, dom_labels_sim) +
                             F.binary_cross_entropy_with_logits(real_dom, dom_labels_real)) / 2

                # Contrastive alignment loss
                cont_loss = contrastive_loss(sim_cont, real_cont)

                loss = task_loss + domain_loss * 0.5 + cont_loss * 0.3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); sched.step()
            step += 1; rloss += loss.item()

            if step % 100 == 0:
                avg = rloss / 100; elapsed = time.time() - t0
                eta = (TOTAL_STEPS - step) * (elapsed / step) / 3600
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"Step {step}/{TOTAL_STEPS} | Loss: {avg:.4f} | Task: {task_loss.item():.4f} | "
                    f"Domain: {domain_loss.item():.4f} | Contrast: {cont_loss.item():.4f} | "
                    f"Alpha: {alpha:.3f} | Mem: {mem:.1f}GB | ETA: {eta:.1f}h")
                if avg < best: best = avg
                rloss = 0.0
            if step % 25000 == 0:
                torch.save({'step': step, 'model': model.state_dict(), 'best': best},
                    SAVE_DIR / f"sim2real_heavy_step{step}.pt")

    torch.save({'step': step, 'model': model.state_dict(), 'best': best},
        SAVE_DIR / "sim2real_heavy_final.pt")
    h = (time.time() - t0) / 3600
    logger.info(f"COMPLETE | Steps: {step} | Best: {best:.4f} | Duration: {h:.1f}h")

if __name__ == "__main__":
    train()
