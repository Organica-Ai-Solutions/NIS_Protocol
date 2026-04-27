#!/usr/bin/env python3
"""
Vision Tracking Training for H100
Trains object detection + tracking models for camera follow control
Deploys to NeuroLinux for robotics/drone applications
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_DIR = Path.home() / "organica-ai" / "models" / "neurolinux" / "vision_tracking"
LOG_DIR = Path.home() / "organica-ai" / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class SyntheticTrackingDataset(Dataset):
    """Generate synthetic tracking data for training"""
    
    def __init__(self, num_samples=10000, frame_size=(640, 480)):
        self.num_samples = num_samples
        self.frame_w, self.frame_h = frame_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate object position (random walk)
        x = np.random.uniform(0, self.frame_w)
        y = np.random.uniform(0, self.frame_h)
        vx = np.random.uniform(-50, 50)
        vy = np.random.uniform(-50, 50)
        
        # Simulate detection bbox (center ± random size)
        size = np.random.uniform(20, 100)
        bbox = [
            max(0, x - size/2),
            max(0, y - size/2),
            min(self.frame_w, x + size/2),
            min(self.frame_h, y + size/2)
        ]
        
        # Target: pan/tilt rates to center object
        center_x = self.frame_w / 2
        center_y = self.frame_h / 2
        err_x = x - center_x
        err_y = y - center_y
        
        # PD control gains
        kp = 0.005
        kd = 0.001
        pan_rate = -(kp * err_x + kd * vx)
        tilt_rate = -(kp * err_y + kd * vy)
        
        # Clip rates
        max_rate = 1.0
        pan_rate = np.clip(pan_rate, -max_rate, max_rate)
        tilt_rate = np.clip(tilt_rate, -max_rate, max_rate)
        
        # Return as tensors
        state = torch.tensor([x, y, vx, vy], dtype=torch.float32)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        target = torch.tensor([pan_rate, tilt_rate], dtype=torch.float32)
        
        return state, bbox_tensor, target


class CameraFollowNet(nn.Module):
    """Neural network for camera follow control"""
    
    def __init__(self, state_dim=4, bbox_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.bbox_encoder = nn.Sequential(
            nn.Linear(bbox_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, state, bbox):
        state_feat = self.state_encoder(state)
        bbox_feat = self.bbox_encoder(bbox)
        combined = torch.cat([state_feat, bbox_feat], dim=1)
        output = self.fusion(combined)
        return output


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for state, bbox, target in dataloader:
        state = state.to(device)
        bbox = bbox.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(state, bbox)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for state, bbox, target in dataloader:
            state = state.to(device)
            bbox = bbox.to(device)
            target = target.to(device)
            
            output = model(state, bbox)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    print("=" * 60)
    print("🎯 Vision Tracking Training for NeuroLinux")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = SyntheticTrackingDataset(num_samples=8000)
    val_dataset = SyntheticTrackingDataset(num_samples=2000)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("🧠 Creating model...")
    model = CameraFollowNet().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("🚀 Starting training...")
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = MODEL_DIR / "camera_follow_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    # Save final model
    final_path = MODEL_DIR / f"camera_follow_final_epoch{EPOCHS}.pt"
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, final_path)
    
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to: {MODEL_DIR}")
    print()
    print("📦 Model files:")
    print(f"  - camera_follow_best.pt (best validation)")
    print(f"  - camera_follow_final_epoch{EPOCHS}.pt (final)")
    print()
    print("🚀 Ready for NeuroLinux deployment!")
    print()


if __name__ == "__main__":
    main()
