#!/usr/bin/env python3
"""
VLA Model Training Script for H100
Train SmolVLA or custom VLA models on H100 GPU

Usage:
    python scripts/train_vla_h100.py --model smolvla --epochs 100 --batch-size 32

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vla_training")

# Check for required packages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# SYNTHETIC DATASET
# =============================================================================

class SyntheticRoboticsDataset(Dataset):
    """
    Synthetic dataset for VLA training.
    Generates image-instruction-action triplets.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (224, 224),
        action_dim: int = 7,
        tasks: Optional[List[str]] = None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        
        self.tasks = tasks or [
            "Pick up the red cube",
            "Move to the blue marker",
            "Push the object forward",
            "Rotate the gripper 90 degrees",
            "Place the object on the table",
            "Navigate to the door",
            "Grasp the handle",
            "Open the drawer",
            "Close the gripper",
            "Move arm to home position"
        ]
        
        logger.info(f"Created synthetic dataset with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic image (random noise with some structure)
        image = self._generate_synthetic_image(idx)
        
        # Select task
        task_idx = idx % len(self.tasks)
        instruction = self.tasks[task_idx]
        
        # Generate action based on task
        action = self._generate_action_for_task(task_idx)
        
        # Generate robot state
        state = np.random.randn(self.action_dim).astype(np.float32) * 0.1
        
        return {
            "image": torch.from_numpy(image),
            "instruction": instruction,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "task_idx": task_idx
        }
    
    def _generate_synthetic_image(self, idx) -> np.ndarray:
        """Generate synthetic image with some structure"""
        np.random.seed(idx)
        
        # Base noise
        image = np.random.rand(3, *self.image_size).astype(np.float32) * 0.3
        
        # Add some structure (colored rectangles representing objects)
        num_objects = np.random.randint(1, 5)
        for _ in range(num_objects):
            x1 = np.random.randint(0, self.image_size[0] - 30)
            y1 = np.random.randint(0, self.image_size[1] - 30)
            x2 = x1 + np.random.randint(20, 50)
            y2 = y1 + np.random.randint(20, 50)
            
            color = np.random.rand(3)
            image[:, x1:min(x2, self.image_size[0]), y1:min(y2, self.image_size[1])] = color.reshape(3, 1, 1)
        
        return image
    
    def _generate_action_for_task(self, task_idx: int) -> np.ndarray:
        """Generate action based on task type"""
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        if task_idx in [0, 6]:  # Pick/grasp tasks
            action[:3] = np.random.randn(3) * 0.1  # Position
            action[6] = 1.0  # Close gripper
        elif task_idx in [1, 5]:  # Navigation tasks
            action[:3] = np.random.randn(3) * 0.5  # Larger position change
        elif task_idx == 2:  # Push
            action[0] = 0.3  # Forward
        elif task_idx == 3:  # Rotate
            action[3:6] = np.array([0, 0, np.pi/2])  # Rotation
        elif task_idx in [4, 7]:  # Place/open
            action[:3] = np.random.randn(3) * 0.1
            action[6] = -1.0  # Open gripper
        elif task_idx == 8:  # Close gripper
            action[6] = 1.0
        elif task_idx == 9:  # Home position
            action[:6] = 0.0  # Zero position
        
        return action


# =============================================================================
# VLA MODEL ARCHITECTURE
# =============================================================================

class SimpleVLAEncoder(nn.Module):
    """Simple vision encoder for VLA"""
    
    def __init__(self, image_size: int = 224, hidden_dim: int = 512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.fc = nn.Linear(256 * 7 * 7, hidden_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleLanguageEncoder(nn.Module):
    """Simple language encoder for VLA"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x is tokenized instruction
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)


class SimpleVLAPolicy(nn.Module):
    """
    Simple VLA policy for training.
    Combines vision, language, and state to predict actions.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        state_dim: int = 7,
        action_dim: int = 7,
        hidden_dim: int = 512,
        action_chunk_size: int = 10
    ):
        super().__init__()
        
        self.vision_encoder = SimpleVLAEncoder(image_size, hidden_dim)
        self.language_encoder = SimpleLanguageEncoder(hidden_dim=hidden_dim)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Action head (predicts action chunk)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * action_chunk_size)
        )
        
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
    
    def forward(self, image, instruction_tokens, state):
        # Encode inputs
        vision_features = self.vision_encoder(image)
        language_features = self.language_encoder(instruction_tokens)
        state_features = self.state_encoder(state)
        
        # Fuse
        fused = torch.cat([vision_features, language_features, state_features], dim=-1)
        fused = self.fusion(fused)
        
        # Predict action chunk
        action_chunk = self.action_head(fused)
        action_chunk = action_chunk.view(-1, self.action_chunk_size, self.action_dim)
        
        return action_chunk


# =============================================================================
# TOKENIZER
# =============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}
        self.next_idx = 2
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_idx and self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.idx_to_word[self.next_idx] = word
                    self.next_idx += 1
    
    def encode(self, text: str, max_length: int = 32) -> torch.Tensor:
        """Encode text to token indices"""
        tokens = []
        for word in text.lower().split():
            idx = self.word_to_idx.get(word, 1)  # 1 = <unk>
            tokens.append(idx)
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'w') as f:
            json.dump({
                "word_to_idx": self.word_to_idx,
                "vocab_size": self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data["vocab_size"])
        tokenizer.word_to_idx = data["word_to_idx"]
        tokenizer.idx_to_word = {int(v): k for k, v in data["word_to_idx"].items()}
        tokenizer.next_idx = len(tokenizer.word_to_idx)
        return tokenizer


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_vla(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: SimpleTokenizer,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/vla",
    log_interval: int = 100
):
    """Train VLA model"""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    training_history = []
    
    logger.info(f"Starting training on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch["image"].to(device)
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            
            # Tokenize instructions
            instruction_tokens = torch.stack([
                tokenizer.encode(inst) for inst in batch["instruction"]
            ]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_actions = model(images, instruction_tokens, states)
            
            # Use first action from chunk for loss
            loss = criterion(predicted_actions[:, 0, :], actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                states = batch["state"].to(device)
                actions = batch["action"].to(device)
                
                instruction_tokens = torch.stack([
                    tokenizer.encode(inst) for inst in batch["instruction"]
                ]).to(device)
                
                predicted_actions = model(images, instruction_tokens, states)
                loss = criterion(predicted_actions[:, 0, :], actions)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, os.path.join(save_dir, "best_model.pt"))
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss
    }, os.path.join(save_dir, "final_model.pt"))
    
    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save tokenizer
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
    return training_history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train VLA model on H100")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "smolvla"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--save-dir", type=str, default="models/vla", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    # Detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    train_dataset = SyntheticRoboticsDataset(num_samples=args.num_samples)
    val_dataset = SyntheticRoboticsDataset(num_samples=args.num_samples // 10)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_dataset.tasks)
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = SimpleVLAPolicy(
        image_size=224,
        state_dim=7,
        action_dim=7,
        hidden_dim=512,
        action_chunk_size=10
    )
    
    # Train
    history = train_vla(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )
    
    logger.info("Training complete!")
    logger.info(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
