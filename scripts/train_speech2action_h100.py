#!/usr/bin/env python3
"""
Speech-to-Action Training (Whisper encoder + VLA decoder)
Maps speech commands directly to robot actions without text intermediate.
H100 GPU optimized.
"""
import os, sys, time, signal, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Graceful shutdown
shutdown = False
def handler(sig, frame):
    global shutdown
    shutdown = True
    print("\nGraceful shutdown requested...")
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

# ============================================================
# Model: Speech encoder (Whisper-style) + Action decoder (VLA)
# ============================================================

class SpeechEncoder(nn.Module):
    """Whisper-style speech encoder with mel spectrogram input"""
    def __init__(self, n_mels=80, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.positional = nn.Parameter(torch.randn(1, 1500, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, mel):
        # mel: (B, n_mels, T)
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T/2, d_model)
        T = x.size(1)
        x = x + self.positional[:, :T, :]
        x = self.transformer(x)
        x = self.ln(x)
        return x  # (B, T/2, d_model)


class ActionDecoder(nn.Module):
    """VLA-style action decoder — predicts joint positions from speech embeddings"""
    def __init__(self, d_model=512, action_dim=7, n_heads=8, n_layers=4):
        super().__init__()
        self.action_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh()
        )

    def forward(self, speech_emb):
        B = speech_emb.size(0)
        query = self.action_query.expand(B, -1, -1)
        x = self.transformer(query, speech_emb)
        x = self.ln(x)
        actions = self.head(x.squeeze(1))
        return actions  # (B, action_dim)


class Speech2ActionModel(nn.Module):
    def __init__(self, n_mels=80, d_model=512, action_dim=7):
        super().__init__()
        self.encoder = SpeechEncoder(n_mels=n_mels, d_model=d_model)
        self.decoder = ActionDecoder(d_model=d_model, action_dim=action_dim)

    def forward(self, mel):
        speech_emb = self.encoder(mel)
        actions = self.decoder(speech_emb)
        return actions


# ============================================================
# Synthetic dataset (speech command -> robot action pairs)
# ============================================================

COMMANDS = [
    ("pick up the red block", [0.3, 0.5, 0.2, -0.1, 0.0, 0.8, 1.0]),
    ("move left", [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("move right", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("go up", [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
    ("go down", [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
    ("open gripper", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ("close gripper", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ("push forward", [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("pull back", [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("rotate wrist", [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]),
    ("place on table", [0.2, 0.3, -0.3, 0.0, -0.1, 0.0, -1.0]),
    ("wave hello", [0.0, 0.0, 0.5, 0.3, -0.3, 0.3, 0.0]),
    ("home position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("stack blocks", [0.2, 0.4, 0.3, -0.1, 0.0, 0.5, 1.0]),
    ("pour water", [0.1, 0.3, 0.4, 0.0, -0.8, 0.0, 1.0]),
    ("wipe surface", [0.3, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0]),
]

class SpeechActionDataset(Dataset):
    def __init__(self, n_samples=10000, n_mels=80, max_len=300):
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.max_len = max_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cmd_idx = idx % len(COMMANDS)
        _, action = COMMANDS[cmd_idx]
        # Synthetic mel spectrogram with command-specific patterns
        torch.manual_seed(idx)
        mel = torch.randn(self.n_mels, self.max_len) * 0.3
        # Embed command index as frequency pattern
        freq = cmd_idx * 5
        t = torch.arange(self.max_len).float()
        for i in range(min(5, self.n_mels)):
            mel[freq % self.n_mels + i] += 0.5 * torch.sin(2 * math.pi * (cmd_idx + 1) * t / self.max_len)
        action_t = torch.tensor(action, dtype=torch.float32)
        # Add small noise to actions
        action_t = action_t + torch.randn_like(action_t) * 0.02
        action_t = action_t.clamp(-1, 1)
        return mel, action_t


# ============================================================
# Training
# ============================================================

def train():
    # Config
    TOTAL_STEPS = 200000
    BATCH_SIZE = 64
    LR_MAX = 3e-4
    LR_MIN = 1e-6
    WARMUP = 5000
    SAVE_EVERY = 25000
    LOG_EVERY = 1000
    SAVE_DIR = "/data/organica-ai/models/speech2action"
    CKPT_DIR = "/home/nvidia/organica-ai/checkpoints/speech2action"

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Model
    model = Speech2ActionModel(n_mels=80, d_model=512, action_dim=7).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Speech2Action model: {params/1e6:.1f}M params")

    # Data
    dataset = SpeechActionDataset(n_samples=50000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01)

    # Training loop
    step = 0
    best_loss = float('inf')
    start_time = time.time()

    while step < TOTAL_STEPS and not shutdown:
        for mel, action in loader:
            if step >= TOTAL_STEPS or shutdown:
                break

            mel = mel.to(device)
            action = action.to(device)

            # LR schedule: warmup + cosine decay
            if step < WARMUP:
                lr = LR_MAX * step / WARMUP
            else:
                progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
                lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            pred = model(mel)
            loss = F.mse_loss(pred, action)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1

            if step % LOG_EVERY == 0:
                elapsed = (time.time() - start_time) / 60
                try:
                    import subprocess
                    r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=5)
                    temps = r.stdout.strip().split('\n')
                    gpu_idx = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
                    temp = temps[0].strip() if gpu_idx >= len(temps) else temps[0].strip()
                except:
                    temp = "0"
                print(f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.6f} | Best: {best_loss:.6f} | "
                      f"LR: {lr:.2e} | Temp: {temp}C | Time: {elapsed:.1f}min")

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % SAVE_EVERY == 0:
                path = f"{SAVE_DIR}/speech2action_step{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                }, path)
                print(f"  Saved: {path}")

                ckpt_path = f"{CKPT_DIR}/speech2action_latest.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                }, ckpt_path)

    # Final save
    final_path = f"{SAVE_DIR}/speech2action_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'loss': best_loss,
    }, final_path)
    print(f"Final model saved: {final_path} (best loss: {best_loss:.6f})")

    best_path = f"{SAVE_DIR}/speech2action_best_{best_loss:.4f}.pt"
    torch.save(model.state_dict(), best_path)
    print(f"Best model saved: {best_path}")

if __name__ == "__main__":
    train()
