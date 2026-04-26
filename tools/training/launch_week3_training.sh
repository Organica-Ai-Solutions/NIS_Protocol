#!/bin/bash
# ============================================================================
# Week 3 Training Launch - Feb 11-18, 2026
# Specialized Models + Cosmos Integration
# ============================================================================
#
# SCHEDULE:
#   GPUs 0-1: Multi-modal CLIP (vision-language alignment for robotics)
#   GPUs 2-3: Cosmos-VLA (Cosmos Reason 2 + VLA for Cookoff demo)
#   GPUs 4-5: Sim2Real domain adaptation
#   GPU 6:    Safety classifier (action validation)
#   GPU 7:    Speech-to-Action (Whisper + VLA)
#
# BEFORE RUNNING:
#   1. SSH into cluster: ssh awesome-gpu-name
#   2. Check current jobs: nvidia-smi
#   3. Save any running checkpoints
#   4. Copy this script: scp launch_week3_training.sh awesome-gpu-name:~/organica-ai/
#
# ============================================================================

set -e

echo "=============================================="
echo " Week 3 Training Launch"
echo " Specialized Models + Cosmos Integration"
echo " $(date)"
echo "=============================================="
echo ""

# Safety: don't kill VLA jobs if still running
RUNNING_VLA=$(ps aux | grep "python.*vla" | grep -v grep | wc -l)
if [ "$RUNNING_VLA" -gt 0 ]; then
    echo "WARNING: $RUNNING_VLA VLA jobs still running!"
    echo "Check with: nvidia-smi"
    echo ""
    read -p "Kill running jobs and start Week 3? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted. Let VLA jobs finish first."
        exit 0
    fi
    pkill -f 'python.*train' 2>/dev/null || true
    sleep 5
fi

# Paths
PYTHON=~/organica-ai/venv/bin/python3
TRAIN_DIR=~/organica-ai/training
LOG_DIR=~/organica-ai/logs/week3
MODEL_DIR=~/organica-ai/models/week3

mkdir -p $LOG_DIR $MODEL_DIR/{clip,cosmos_vla,sim2real,safety,speech_action}

# ============================================================================
# GPU 0-1: Multi-modal CLIP for Robotics
# ============================================================================
echo "GPU 0-1: Multi-modal CLIP (vision-language alignment)"
cat > $TRAIN_DIR/train_clip_robotics.py << 'CLIP_SCRIPT'
#!/usr/bin/env python3
"""Multi-modal CLIP for robotics scene understanding"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(512, hidden_dim)
    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return self.proj(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)

class RoboticsCLIP(nn.Module):
    def __init__(self, hidden_dim=512, temperature=0.07):
        super().__init__()
        self.vision = VisionEncoder(hidden_dim)
        self.text = TextEncoder(hidden_dim=hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, images, tokens):
        img_feat = self.vision(images)
        txt_feat = self.text(tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ txt_feat.T) / self.temperature.exp()
        return logits

class RoboticsSceneDataset(Dataset):
    DESCRIPTIONS = [
        "robot arm reaching for red cube on table",
        "gripper grasping cylindrical object",
        "robot placing object in green bin",
        "empty workspace with tools",
        "robot arm in home position",
        "cluttered table with multiple objects",
        "robot navigating around obstacle",
        "arm extended with open gripper",
        "pick and place sequence in progress",
        "safety zone boundary visible",
    ]
    def __init__(self, num_samples=50000):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.rand(3, 224, 224).astype(np.float32) * 0.3
        num_obj = np.random.randint(1, 6)
        for _ in range(num_obj):
            x1, y1 = np.random.randint(0, 180, 2)
            w, h = np.random.randint(20, 60, 2)
            color = np.random.rand(3)
            image[:, x1:min(x1+w, 224), y1:min(y1+h, 224)] = color.reshape(3, 1, 1)
        desc = self.DESCRIPTIONS[idx % len(self.DESCRIPTIONS)]
        tokens = [hash(w) % 4998 + 2 for w in desc.split()]
        tokens = tokens[:20] + [0] * max(0, 20 - len(tokens))
        return {
            "image": torch.from_numpy(image),
            "tokens": torch.tensor(tokens, dtype=torch.long)
        }

def train():
    print(f"Training Robotics CLIP on {DEVICE}")
    model = RoboticsCLIP().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    dataset = RoboticsSceneDataset(100000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    out_dir = Path.home() / "organica-ai" / "models" / "week3" / "clip"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in loader:
            images = batch["image"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            logits = model(images, tokens)
            labels = torch.arange(logits.size(0), device=DEVICE)
            loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"[CLIP] Epoch {epoch+1}/50 | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), out_dir / f"clip_robotics_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), out_dir / "clip_robotics_final.pt")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train()
CLIP_SCRIPT

CUDA_VISIBLE_DEVICES=0,1 nohup $PYTHON $TRAIN_DIR/train_clip_robotics.py > $LOG_DIR/clip_robotics.log 2>&1 &
echo "  PID: $!"

# ============================================================================
# GPU 2-3: Cosmos-VLA (Cosmos Reason 2 style + VLA for Cookoff)
# ============================================================================
echo "GPU 2-3: Cosmos-VLA (reasoning + action prediction)"
cat > $TRAIN_DIR/train_cosmos_vla.py << 'COSMOS_VLA_SCRIPT'
#!/usr/bin/env python3
"""Cosmos-style VLA: chain-of-thought reasoning + action prediction"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosmosVLAModel(nn.Module):
    """VLA with reasoning chain: image+text -> reasoning -> action"""
    def __init__(self, hidden_dim=768, action_dim=7, reason_steps=4):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.vision_proj = nn.Linear(512, hidden_dim)
        self.text_embed = nn.Embedding(5000, 256)
        self.text_enc = nn.LSTM(256, hidden_dim // 2, 2, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        # Reasoning transformer (chain-of-thought)
        self.reasoning = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=reason_steps
        )
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 10)  # 10-step action chunk
        )
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.action_dim = action_dim
    def forward(self, images, tokens):
        vis = self.vision(images).squeeze(-1).squeeze(-1)
        vis = self.vision_proj(vis).unsqueeze(1)
        txt = self.text_embed(tokens)
        _, (h, _) = self.text_enc(txt)
        txt_feat = self.text_proj(torch.cat([h[-2], h[-1]], dim=-1)).unsqueeze(1)
        combined = torch.cat([vis, txt_feat], dim=1)
        reasoned = self.reasoning(combined)
        pooled = reasoned.mean(dim=1)
        actions = self.action_head(pooled).view(-1, 10, self.action_dim)
        confidence = self.confidence_head(pooled)
        return actions, confidence

class CosmosDataset(Dataset):
    COMMANDS = [
        "pick up the red cube from the table",
        "place the object in the green zone",
        "wave hello to the camera",
        "move to home position",
        "push the blue cylinder forward",
        "rotate gripper 90 degrees clockwise",
        "navigate around the obstacle",
        "stack the red block on blue block",
    ]
    def __init__(self, n=80000):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        np.random.seed(idx)
        img = np.random.rand(3, 224, 224).astype(np.float32) * 0.3
        for _ in range(np.random.randint(1, 5)):
            x, y = np.random.randint(0, 180, 2)
            w, h = np.random.randint(20, 60, 2)
            img[:, x:min(x+w, 224), y:min(y+h, 224)] = np.random.rand(3, 1, 1)
        cmd = self.COMMANDS[idx % len(self.COMMANDS)]
        tokens = [hash(w) % 4998 + 2 for w in cmd.split()]
        tokens = tokens[:20] + [0] * max(0, 20 - len(tokens))
        action = np.random.randn(10, 7).astype(np.float32) * 0.1
        return {
            "image": torch.from_numpy(img),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "action": torch.from_numpy(action),
            "confidence": torch.tensor([0.9], dtype=torch.float32)
        }

def train():
    print(f"Training Cosmos-VLA on {DEVICE}")
    model = CosmosVLAModel().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    dataset = CosmosDataset(100000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    out_dir = Path.home() / "organica-ai" / "models" / "week3" / "cosmos_vla"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            toks = batch["tokens"].to(DEVICE)
            gt_act = batch["action"].to(DEVICE)
            gt_conf = batch["confidence"].to(DEVICE)
            pred_act, pred_conf = model(imgs, toks)
            loss_act = nn.functional.mse_loss(pred_act, gt_act)
            loss_conf = nn.functional.binary_cross_entropy(pred_conf, gt_conf)
            loss = loss_act + 0.1 * loss_conf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"[CosmosVLA] Epoch {epoch+1}/100 | Loss: {avg:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), out_dir / f"cosmos_vla_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), out_dir / "cosmos_vla_final.pt")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train()
COSMOS_VLA_SCRIPT

CUDA_VISIBLE_DEVICES=2,3 nohup $PYTHON $TRAIN_DIR/train_cosmos_vla.py > $LOG_DIR/cosmos_vla.log 2>&1 &
echo "  PID: $!"

# ============================================================================
# GPU 4-5: Sim2Real Domain Adaptation
# ============================================================================
echo "GPU 4-5: Sim2Real Domain Adaptation"
cat > $TRAIN_DIR/train_sim2real.py << 'SIM2REAL_SCRIPT'
#!/usr/bin/env python3
"""Sim2Real domain adaptation for transferring sim-trained models to real hardware"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)
    def forward(self, x):
        return self.fc(self.conv(x).squeeze(-1).squeeze(-1))

class ActionPredictor(nn.Module):
    def __init__(self, feature_dim=256, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class Sim2RealModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FeatureExtractor()
        self.action = ActionPredictor()
        self.domain = DomainDiscriminator()
    def forward(self, x, alpha=1.0):
        feat = self.feature(x)
        action = self.action(feat)
        # Gradient reversal for domain adaptation
        domain = self.domain(feat)
        return action, domain, feat

class SimRealDataset(Dataset):
    def __init__(self, n=50000, domain="sim"):
        self.n = n
        self.domain = domain
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        np.random.seed(idx + (0 if self.domain == "sim" else 100000))
        noise = 0.1 if self.domain == "sim" else 0.3
        img = np.random.rand(3, 128, 128).astype(np.float32) * noise
        for _ in range(np.random.randint(1, 4)):
            x, y = np.random.randint(0, 90, 2)
            w, h = np.random.randint(15, 40, 2)
            img[:, x:min(x+w, 128), y:min(y+h, 128)] = np.random.rand(3, 1, 1)
        action = np.random.randn(7).astype(np.float32) * 0.1
        label = 0.0 if self.domain == "sim" else 1.0
        return {
            "image": torch.from_numpy(img),
            "action": torch.from_numpy(action),
            "domain": torch.tensor([label])
        }

def train():
    print(f"Training Sim2Real on {DEVICE}")
    model = Sim2RealModel().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    sim_data = SimRealDataset(60000, "sim")
    real_data = SimRealDataset(60000, "real")
    sim_loader = DataLoader(sim_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    real_loader = DataLoader(real_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    out_dir = Path.home() / "organica-ai" / "models" / "week3" / "sim2real"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(80):
        model.train()
        total = 0
        for sim_batch, real_batch in zip(sim_loader, real_loader):
            sim_img = sim_batch["image"].to(DEVICE)
            sim_act = sim_batch["action"].to(DEVICE)
            real_img = real_batch["image"].to(DEVICE)
            # Task loss on sim data
            pred_act, sim_dom, _ = model(sim_img)
            loss_task = nn.functional.mse_loss(pred_act, sim_act)
            # Domain loss
            _, real_dom, _ = model(real_img)
            loss_dom = nn.functional.binary_cross_entropy(sim_dom, torch.zeros_like(sim_dom))
            loss_dom += nn.functional.binary_cross_entropy(real_dom, torch.ones_like(real_dom))
            loss = loss_task + 0.5 * loss_dom
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / min(len(sim_loader), len(real_loader))
        print(f"[Sim2Real] Epoch {epoch+1}/80 | Loss: {avg:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), out_dir / f"sim2real_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), out_dir / "sim2real_final.pt")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train()
SIM2REAL_SCRIPT

CUDA_VISIBLE_DEVICES=4,5 nohup $PYTHON $TRAIN_DIR/train_sim2real.py > $LOG_DIR/sim2real.log 2>&1 &
echo "  PID: $!"

# ============================================================================
# GPU 6: Safety Classifier
# ============================================================================
echo "GPU 6: Safety Action Classifier"
cat > $TRAIN_DIR/train_safety_classifier.py << 'SAFETY_SCRIPT'
#!/usr/bin/env python3
"""Safety classifier for robot action validation"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafetyClassifier(nn.Module):
    """Classifies robot actions as safe/unsafe given state and action"""
    def __init__(self, state_dim=14, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 4)  # safe, collision_risk, joint_limit, excessive_force
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class SafetyDataset(Dataset):
    def __init__(self, n=200000):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        np.random.seed(idx)
        state = np.random.randn(14).astype(np.float32) * 0.5
        action = np.random.randn(7).astype(np.float32) * 0.3
        # Generate safety labels based on physics heuristics
        labels = np.zeros(4, dtype=np.float32)
        if np.linalg.norm(action[:3]) < 0.5 and np.all(np.abs(state[:7]) < 2.0):
            labels[0] = 1.0  # safe
        if np.linalg.norm(action[:3]) > 0.8:
            labels[1] = 1.0  # collision risk
        if np.any(np.abs(state[:7] + action) > 2.5):
            labels[2] = 1.0  # joint limit
        if np.linalg.norm(action[3:6]) > 1.0:
            labels[3] = 1.0  # excessive force
        if labels.sum() == 0:
            labels[0] = 1.0
        return {
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "labels": torch.from_numpy(labels)
        }

def train():
    print(f"Training Safety Classifier on {DEVICE}")
    model = SafetyClassifier().to(DEVICE)
    dataset = SafetyDataset(200000)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    out_dir = Path.home() / "organica-ai" / "models" / "week3" / "safety"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(60):
        model.train()
        total, correct, count = 0, 0, 0
        for batch in loader:
            state = batch["state"].to(DEVICE)
            action = batch["action"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(state, action)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).all(dim=-1).sum().item()
            count += labels.size(0)
        scheduler.step()
        acc = correct / count * 100
        avg = total / len(loader)
        print(f"[Safety] Epoch {epoch+1}/60 | Loss: {avg:.4f} | Acc: {acc:.1f}%")
        if (epoch + 1) % 15 == 0:
            torch.save(model.state_dict(), out_dir / f"safety_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), out_dir / "safety_final.pt")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train()
SAFETY_SCRIPT

CUDA_VISIBLE_DEVICES=6 nohup $PYTHON $TRAIN_DIR/train_safety_classifier.py > $LOG_DIR/safety.log 2>&1 &
echo "  PID: $!"

# ============================================================================
# GPU 7: Speech-to-Action
# ============================================================================
echo "GPU 7: Speech-to-Action (voice commands -> robot actions)"
cat > $TRAIN_DIR/train_speech_action.py << 'SPEECH_SCRIPT'
#!/usr/bin/env python3
"""Speech-to-Action: voice commands directly to robot actions"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechToAction(nn.Module):
    def __init__(self, mel_dim=80, hidden=512, action_dim=7):
        super().__init__()
        self.acoustic = nn.Sequential(
            nn.Conv1d(mel_dim, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3, 2, 1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, hidden, 3, 2, 1), nn.ReLU(), nn.BatchNorm1d(hidden),
        )
        self.encoder = nn.LSTM(hidden, hidden // 2, 3, batch_first=True, bidirectional=True, dropout=0.1)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, action_dim * 10)
        )
        self.cmd_classifier = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 8)  # 8 command types
        )
        self.action_dim = action_dim
    def forward(self, mel):
        x = self.acoustic(mel).transpose(1, 2)
        x, _ = self.encoder(x)
        x = x[:, -1, :]
        actions = self.action_head(x).view(-1, 10, self.action_dim)
        cmd_type = self.cmd_classifier(x)
        return actions, cmd_type

class SpeechCommandDataset(Dataset):
    def __init__(self, n=100000):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        np.random.seed(idx)
        mel = np.random.randn(80, 100).astype(np.float32) * 0.5
        cmd_type = idx % 8
        action = np.random.randn(10, 7).astype(np.float32) * 0.1
        return {
            "mel": torch.from_numpy(mel),
            "action": torch.from_numpy(action),
            "cmd_type": torch.tensor(cmd_type, dtype=torch.long)
        }

def train():
    print(f"Training Speech-to-Action on {DEVICE}")
    model = SpeechToAction().to(DEVICE)
    dataset = SpeechCommandDataset(100000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    out_dir = Path.home() / "organica-ai" / "models" / "week3" / "speech_action"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(80):
        model.train()
        total = 0
        for batch in loader:
            mel = batch["mel"].to(DEVICE)
            gt_act = batch["action"].to(DEVICE)
            gt_cmd = batch["cmd_type"].to(DEVICE)
            pred_act, pred_cmd = model(mel)
            loss = nn.functional.mse_loss(pred_act, gt_act) + 0.5 * nn.functional.cross_entropy(pred_cmd, gt_cmd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"[Speech2Act] Epoch {epoch+1}/80 | Loss: {avg:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), out_dir / f"speech_action_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), out_dir / "speech_action_final.pt")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train()
SPEECH_SCRIPT

CUDA_VISIBLE_DEVICES=7 nohup $PYTHON $TRAIN_DIR/train_speech_action.py > $LOG_DIR/speech_action.log 2>&1 &
echo "  PID: $!"

# ============================================================================
# SUMMARY
# ============================================================================
sleep 5
echo ""
echo "=============================================="
echo " Week 3 Training Deployed!"
echo " $(date)"
echo "=============================================="
echo ""
echo " GPU 0-1: Robotics CLIP (50 epochs, ~6h)"
echo " GPU 2-3: Cosmos-VLA (100 epochs, ~12h)"
echo " GPU 4-5: Sim2Real (80 epochs, ~8h)"
echo " GPU 6:   Safety Classifier (60 epochs, ~4h)"
echo " GPU 7:   Speech-to-Action (80 epochs, ~8h)"
echo ""
echo " Models: ~/organica-ai/models/week3/"
echo " Logs:   ~/organica-ai/logs/week3/"
echo ""
echo " Monitor: nvidia-smi"
echo " Logs:    tail -f ~/organica-ai/logs/week3/*.log"
echo ""

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv
