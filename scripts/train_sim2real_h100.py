#!/usr/bin/env python3
"""
Sim2Real Domain Adaptation Training for H100
Learns to bridge simulation-to-real visual domain gap for robot manipulation.
Uses adversarial domain adaptation (DANN-style) with gradient reversal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, signal, subprocess, gc, math
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/organica-ai/logs/sim2real.log')
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_stopped = False
def signal_handler(sig, frame):
    global training_stopped
    logger.info("SIGINT received - graceful shutdown...")
    training_stopped = True
signal.signal(signal.SIGINT, signal_handler)

def get_gpu_temp():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=5)
        temps = result.stdout.strip().split("\n")
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        return float(temps[gpu_id]) if gpu_id < len(temps) else 0.0
    except:
        return 0.0


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class FeatureExtractor(nn.Module):
    """Shared visual feature extractor (ResNet-style)"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TaskPredictor(nn.Module):
    """Predicts robot action from features"""
    def __init__(self, feature_dim=512, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    """Discriminates sim vs real domain"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, alpha=1.0):
        x = GradientReversal.apply(x, alpha)
        return self.net(x)


class Sim2RealModel(nn.Module):
    def __init__(self, feature_dim=512, action_dim=7):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.task_predictor = TaskPredictor(feature_dim, action_dim)
        self.domain_classifier = DomainClassifier(feature_dim)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        actions = self.task_predictor(features)
        domain = self.domain_classifier(features, alpha)
        return actions, domain, features


def train(epochs=200000, batch_size=32):
    model = Sim2RealModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    task_criterion = nn.MSELoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Sim2Real Domain Adaptation | Device: {device}")
    logger.info(f"Parameters: {param_count:,} | Epochs: {epochs:,}")

    save_dir = Path("/data/organica-ai/models/sim2real")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(os.path.expanduser("~/organica-ai/checkpoints/sim2real"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        if training_stopped:
            break

        # Schedule adversarial strength (ramp up over training)
        p = epoch / epochs
        alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

        # Simulated data: sim images (bright, clean) and real images (noisy, darker)
        sim_images = torch.randn(batch_size, 3, 128, 128, device=device) * 0.5 + 0.5
        real_images = torch.randn(batch_size, 3, 128, 128, device=device) * 0.8 + 0.3
        sim_actions = torch.randn(batch_size, 7, device=device) * 0.1
        sim_labels = torch.zeros(batch_size, 1, device=device)  # 0 = sim
        real_labels = torch.ones(batch_size, 1, device=device)  # 1 = real

        optimizer.zero_grad()

        # Forward sim
        sim_pred_actions, sim_domain, _ = model(sim_images, alpha)
        task_loss = task_criterion(sim_pred_actions, sim_actions)
        sim_domain_loss = domain_criterion(sim_domain, sim_labels)

        # Forward real (no task labels for real)
        _, real_domain, _ = model(real_images, alpha)
        real_domain_loss = domain_criterion(real_domain, real_labels)

        domain_loss = (sim_domain_loss + real_domain_loss) / 2
        total_loss = task_loss + domain_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            temp = get_gpu_temp()
            lr = optimizer.param_groups[0]["lr"]
            marker = " *BEST*" if total_loss.item() < best_loss else ""
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            logger.info(
                f"Step {epoch}/{epochs} | Task: {task_loss.item():.4f} | Domain: {domain_loss.item():.4f} | "
                f"Total: {total_loss.item():.4f} | Best: {best_loss:.4f} | Alpha: {alpha:.3f} | "
                f"LR: {lr:.2e} | Temp: {temp:.1f}C | Time: {elapsed/60:.1f}min{marker}"
            )
            if temp > 85:
                logger.warning(f"THERMAL: {temp}C")
                torch.cuda.empty_cache()
                gc.collect()

        if epoch % 10000 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss},
                checkpoint_dir / f"sim2real_step{epoch}.pt"
            )
        if epoch % 50000 == 0:
            torch.save(model.state_dict(), save_dir / f"sim2real_step{epoch}.pt")

    # Save final
    torch.save(model.state_dict(), save_dir / "sim2real_final.pt")
    torch.save(model.state_dict(), save_dir / f"sim2real_best_{best_loss:.4f}.pt")
    logger.info(f"Training complete! Best total loss: {best_loss:.4f} | Time: {(time.time()-start_time)/60:.1f}min")


if __name__ == "__main__":
    train(epochs=200000, batch_size=32)
