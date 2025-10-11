#!/usr/bin/env python3
"""
🚀 BitNet Online Training System - NIS Protocol v3.1

Real-time BitNet model training and fine-tuning while the system is online.
This enables BitNet to continuously learn from conversations and improve for offline use.

Features:
- Continuous online training from real conversations
- Consciousness-guided training data quality assessment
- Physics-informed response validation for training
- Automatic model checkpointing and offline preparation
- Real-time performance monitoring and adaptation
"""

import asyncio
import json
import logging
import time
import os
import threading
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Deque, Union
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import numpy as np

# Training and model imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        get_scheduler
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRAINING_AVAILABLE = True
except (ImportError, OSError) as e:
    TRAINING_AVAILABLE = False
    logging.warning(f"Training libraries not available ({e}) - using training simulation mode")

# NIS Protocol imports
from ...core.agent import NISAgent
from ...services.consciousness_service import ConsciousnessService
from ...utils.confidence_calculator import calculate_confidence
from ...utils.integrity_metrics import calculate_confidence, create_default_confidence_factors


@dataclass
class TrainingExample:
    """Single training example with NIS validation"""
    prompt: str
    response: str
    consciousness_score: float
    physics_compliance: float
    user_feedback: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0
    used_for_training: bool = False


@dataclass
class OnlineTrainingConfig:
    """Configuration for online BitNet training"""
    # Model settings
    model_path: str = "models/bitnet/models/bitnet"
    checkpoint_dir: str = "models/bitnet/checkpoints"
    mobile_bundle_dir: str = "models/bitnet/mobile"
    mobile_variant: str = "bitnet-b1.58-2b4t-mobile"
    mobile_version: str = "v1"
    create_mobile_bundle: bool = True
    
    # Training hyperparameters
    learning_rate: float = 1e-5  # Lower for online learning
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # LoRA settings for efficient training
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Online training settings
    min_examples_before_training: int = 10
    training_interval_seconds: float = 300.0  # 5 minutes
    quality_threshold: float = 0.7
    max_training_examples: int = 1000
    
    # Validation settings
    consciousness_weight: float = 0.3
    physics_weight: float = 0.3
    user_feedback_weight: float = 0.4
    
    # Checkpointing
    checkpoint_interval_minutes: int = 30
    max_checkpoints: int = 10


if TRAINING_AVAILABLE:
    class NISTrainingDataset(Dataset):
        """Dataset for NIS Protocol training examples"""
        
        def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            
            # Format as conversation
            text = f"Human: {example.prompt}\n\nAssistant: {example.response}"
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze().clone()
            }


class BitNetOnlineTrainer(NISAgent):
    """
    🚀 BitNet Online Trainer
    
    Continuously trains BitNet models using real conversation data with:
    - Real-time learning from user interactions
    - Consciousness-guided quality assessment
    - Physics-informed validation
    - Automatic offline model preparation
    - Performance monitoring and adaptation
    """
    
    def __init__(
        self,
        agent_id: str = "bitnet_online_trainer",
        config: Optional[OnlineTrainingConfig] = None,
        consciousness_service: Optional[ConsciousnessService] = None
    ):
        super().__init__(agent_id)
        
        self.config = config or OnlineTrainingConfig()
        self.consciousness_service = consciousness_service
        self.logger = logging.getLogger(f"nis.training.{agent_id}")
        
        # Training state
        self.training_examples: Deque[TrainingExample] = deque(
            maxlen=self.config.max_training_examples
        )
        self.is_training = False
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.training_metrics = {
            'total_examples_collected': 0,
            'total_training_sessions': 0,
            'average_quality_score': 0.0,
            'model_improvement_score': 0.0,
            'last_training_time': None,
            'next_training_time': None,
            'offline_readiness_score': 0.0
        }

        # Mobile bundle metadata
        self.mobile_bundle_metadata: Dict[str, Any] = {
            "path": None,
            "checksum": None,
            "size_mb": None,
            "version": self.config.mobile_version,
            "variant": self.config.mobile_variant,
            "lora_available": False,
            "download_url": None,
        }
        
        # Background training thread
        self.training_thread = None
        self.should_stop_training = False
        
        # Initialize training system
        if TRAINING_AVAILABLE:
            self._initialize_training_system()
        else:
            self.logger.warning("🔄 Training simulation mode - libraries not available")
        
        # Ensure mobile bundle directory exists
        Path(self.config.mobile_bundle_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"🚀 BitNet Online Trainer initialized: {agent_id}")
    
    def _initialize_training_system(self):
        """Initialize the BitNet training system"""
        try:
            self.logger.info("🔄 Initializing BitNet training system...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Apply LoRA for efficient training
            if self.config.use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                self.model = get_peft_model(self.model, lora_config)
                self.logger.info("✅ Applied LoRA configuration for efficient training")
            
            # Setup optimizer and scheduler
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            # Create checkpoint directory
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("✅ BitNet training system initialized successfully")
            
            # Start background training
            self._start_background_training()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training system: {e}")
            raise
    
    async def add_training_example(
        self,
        prompt: str,
        response: str,
        user_feedback: Optional[float] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new training example from real conversation data
        
        Args:
            prompt: User prompt/input
            response: System response
            user_feedback: Optional user feedback score (0.0-1.0)
            additional_context: Additional context for validation
            
        Returns:
            bool: True if example was added successfully
        """
        try:
            self.logger.info(f"📝 Adding training example: prompt length {len(prompt)}")
            
            # 1. 🧠 Consciousness validation
            consciousness_score = 0.7  # Default
            if self.consciousness_service:
                consciousness_result = await self.consciousness_service.process_through_consciousness({
                    "prompt": prompt,
                    "response": response,
                    "context": additional_context or {}
                })
                consciousness_score = consciousness_result.get(
                    "consciousness_validation", {}
                ).get("consciousness_confidence", 0.7)
            
            # 2. ⚗️ Physics compliance check (placeholder for real implementation)
            physics_compliance = self._assess_physics_compliance(response, additional_context)
            
            # 3. 📊 Calculate overall quality score
            quality_score = self._calculate_quality_score(
                consciousness_score, physics_compliance, user_feedback
            )
            
            # 4. ✅ Add example if quality meets threshold
            if quality_score >= self.config.quality_threshold:
                example = TrainingExample(
                    prompt=prompt,
                    response=response,
                    consciousness_score=consciousness_score,
                    physics_compliance=physics_compliance,
                    user_feedback=user_feedback,
                    quality_score=quality_score
                )
                
                self.training_examples.append(example)
                self.training_metrics['total_examples_collected'] += 1
                
                # Update average quality score
                self._update_average_quality_score(quality_score)
                
                self.logger.info(f"✅ Training example added: quality={quality_score:.3f}")
                return True
            else:
                self.logger.info(f"⚠️ Training example rejected: quality={quality_score:.3f} < {self.config.quality_threshold}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error adding training example: {e}")
            return False
    
    def _assess_physics_compliance(self, response: str, context: Optional[Dict[str, Any]]) -> float:
        """Assess physics compliance of response"""
        # Simple physics compliance assessment
        # In real implementation, this would use the PINN physics validation
        physics_keywords = ["energy", "conservation", "momentum", "force", "physics", "law", "equation"]
        
        if any(keyword in response.lower() for keyword in physics_keywords):
            return 0.8  # Higher compliance for physics-related content
        return 0.6  # Default compliance
    
    def _calculate_quality_score(
        self, 
        consciousness_score: float, 
        physics_compliance: float, 
        user_feedback: Optional[float]
    ) -> float:
        """Calculate overall quality score for training example"""
        
        scores = [consciousness_score, physics_compliance]
        weights = [self.config.consciousness_weight, self.config.physics_weight]
        
        if user_feedback is not None:
            scores.append(user_feedback)
            weights.append(self.config.user_feedback_weight)
        else:
            # Redistribute weights if no user feedback
            weights = [w / sum(weights[:2]) for w in weights[:2]]
        
        return sum(score * weight for score, weight in zip(scores, weights))
    
    def _update_average_quality_score(self, new_score: float):
        """Update running average of quality scores"""
        current_avg = self.training_metrics['average_quality_score']
        total_examples = self.training_metrics['total_examples_collected']
        
        if total_examples == 1:
            self.training_metrics['average_quality_score'] = new_score
        else:
            # Running average update
            self.training_metrics['average_quality_score'] = (
                (current_avg * (total_examples - 1) + new_score) / total_examples
            )
    
    def _start_background_training(self):
        """Start background training thread"""
        if not TRAINING_AVAILABLE:
            self.logger.warning("Training libraries not available - skipping background training")
            return
        
        self.training_thread = threading.Thread(
            target=self._background_training_loop,
            daemon=True
        )
        self.training_thread.start()
        self.logger.info("🔄 Background training thread started")
    
    def _background_training_loop(self):
        """Background loop for periodic training"""
        last_training_time = 0
        last_checkpoint_time = 0
        
        while not self.should_stop_training:
            try:
                current_time = time.time()
                
                # Check if it's time for training
                if (current_time - last_training_time) >= self.config.training_interval_seconds:
                    if len(self.training_examples) >= self.config.min_examples_before_training:
                        self.logger.info(f"🚀 Starting training session with {len(self.training_examples)} examples")
                        
                        success = self._execute_training_session()
                        if success:
                            last_training_time = current_time
                            self.training_metrics['last_training_time'] = datetime.now().isoformat()
                            self.training_metrics['next_training_time'] = (
                                datetime.now() + timedelta(seconds=self.config.training_interval_seconds)
                            ).isoformat()
                
                # Check if it's time for checkpointing
                if (current_time - last_checkpoint_time) >= (self.config.checkpoint_interval_minutes * 60):
                    self._save_checkpoint()
                    last_checkpoint_time = current_time
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in background training loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def _execute_training_session(self) -> bool:
        """Execute a training session with current examples"""
        if self.is_training or not TRAINING_AVAILABLE:
            return False
        
        try:
            self.is_training = True
            start_time = time.time()
            
            # Get unused training examples
            unused_examples = [ex for ex in self.training_examples if not ex.used_for_training]
            if len(unused_examples) < self.config.min_examples_before_training:
                self.logger.info("🔄 Not enough unused examples for training")
                return False
            
            self.logger.info(f"🎯 Training with {len(unused_examples)} new examples")
            
            # Create dataset and dataloader
            dataset = NISTrainingDataset(unused_examples, self.tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            
            # Mark examples as used
            for example in unused_examples:
                example.used_for_training = True
            
            # Calculate training metrics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            training_time = time.time() - start_time
            
            # Update training metrics
            self.training_metrics['total_training_sessions'] += 1
            improvement_score = max(0, 1.0 - avg_loss)  # Simple improvement metric
            self.training_metrics['model_improvement_score'] = improvement_score
            self.training_metrics['average_quality_score'] = np.mean(
                [ex.quality_score for ex in self.training_examples]
            ) if self.training_examples else 0.0
            self.training_metrics['last_training_time'] = datetime.now().isoformat()
            
            # Update offline readiness score
            self._update_offline_readiness_score()
            
            self.logger.info(f"✅ Training session completed: loss={avg_loss:.4f}, time={training_time:.2f}s")
            self.logger.info(f"📊 Offline readiness: {self.training_metrics['offline_readiness_score']:.2f}")
            
            # Prepare mobile bundle if enabled
            if self.config.create_mobile_bundle:
                self._prepare_mobile_bundle(checkpoint_hint="training")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training session failed: {e}")
            return False
        finally:
            self.is_training = False
    
    def _update_offline_readiness_score(self):
        """Update the offline readiness score"""
        # Calculate readiness based on multiple factors
        factors = []
        
        # Number of training examples
        example_score = min(1.0, len(self.training_examples) / 500)  # Target 500 examples
        factors.append(example_score)
        
        # Average quality of examples
        quality_score = self.training_metrics['average_quality_score']
        factors.append(quality_score)
        
        # Model improvement score
        improvement_score = self.training_metrics['model_improvement_score']
        factors.append(improvement_score)
        
        # Number of training sessions
        sessions_score = min(1.0, self.training_metrics['total_training_sessions'] / 10)  # Target 10 sessions
        factors.append(sessions_score)
        
        # Calculate overall readiness
        self.training_metrics['offline_readiness_score'] = np.mean(factors)
    
    def _save_checkpoint(self):
        """Save model checkpoint for offline use"""
        if not TRAINING_AVAILABLE or self.model is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = Path(self.config.checkpoint_dir) / f"bitnet_checkpoint_{timestamp}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save training metrics
            metrics_path = checkpoint_path / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            # Save training configuration
            config_path = checkpoint_path / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_path': self.config.model_path,
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'use_lora': self.config.use_lora,
                    'lora_r': self.config.lora_r,
                    'lora_alpha': self.config.lora_alpha,
                    'total_examples': len(self.training_examples),
                    'offline_readiness_score': self.training_metrics['offline_readiness_score']
                }, f, indent=2)
            
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Prepare mobile bundle if enabled
            if self.config.create_mobile_bundle:
                self._prepare_mobile_bundle(checkpoint_hint=checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoints = sorted(
                [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("bitnet_checkpoint_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the latest N checkpoints
            for checkpoint in checkpoints[self.config.max_checkpoints:]:
                import shutil
                shutil.rmtree(checkpoint)
                self.logger.info(f"🗑️ Removed old checkpoint: {checkpoint.name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old checkpoints: {e}")
    
    def _prepare_mobile_bundle(self, checkpoint_hint: Optional[Union[Path, str]] = None) -> None:
        """Prepare mobile-friendly BitNet bundle for edge devices."""
        try:
            if not self.model or not self.tokenizer:
                self.logger.warning("⚠️ Cannot prepare mobile bundle - model not loaded")
                return

            bundle_dir = Path(self.config.mobile_bundle_dir)
            bundle_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_name = f"{self.config.mobile_variant}_{timestamp}"
            bundle_temp_dir = bundle_dir / bundle_name
            bundle_temp_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path: Optional[Path] = None
            if checkpoint_hint is not None:
                checkpoint_path = Path(str(checkpoint_hint))
                if not checkpoint_path.exists():
                    checkpoint_path = None

            self.logger.info("📦 Preparing mobile BitNet bundle for edge deployment")
            if checkpoint_path is not None and checkpoint_path.is_dir():
                shutil.copytree(checkpoint_path, bundle_temp_dir, dirs_exist_ok=True)
            else:
                self.model.save_pretrained(bundle_temp_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(bundle_temp_dir)

            manifest = {
                "variant": self.config.mobile_variant,
                "version": self.config.mobile_version,
                "created_at": timestamp,
                "offline_readiness_score": self.training_metrics['offline_readiness_score'],
                "total_training_sessions": self.training_metrics['total_training_sessions'],
                "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            }

            manifest_path = bundle_temp_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            archive_basename = bundle_dir / bundle_name
            shutil.make_archive(str(archive_basename), 'zip', root_dir=bundle_temp_dir)
            archive_path = archive_basename.with_suffix('.zip')

            checksum = self._calculate_checksum(archive_path)
            size_mb = round(archive_path.stat().st_size / (1024 * 1024), 2)

            shutil.rmtree(bundle_temp_dir, ignore_errors=True)

            self.mobile_bundle_metadata.update({
                "path": str(archive_path),
                "checksum": checksum,
                "size_mb": size_mb,
                "version": self.config.mobile_version,
                "variant": self.config.mobile_variant,
                "lora_available": self.config.use_lora,
                "download_url": None,
            })

            self.logger.info(
                "✅ Mobile BitNet bundle ready | "
                f"size={size_mb}MB | checksum={checksum[:8]}..."
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to prepare mobile BitNet bundle: {e}")

    @staticmethod
    def _calculate_checksum(path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA-256 checksum for a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics"""
        return {
            "is_training": self.is_training,
            "training_available": TRAINING_AVAILABLE,
            "total_examples": len(self.training_examples),
            "unused_examples": len([ex for ex in self.training_examples if not ex.used_for_training]),
            "metrics": self.training_metrics.copy(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "training_interval_seconds": self.config.training_interval_seconds,
                "quality_threshold": self.config.quality_threshold,
                "model_path": self.config.model_path
            },
            "mobile_bundle": self.mobile_bundle_metadata.copy()
        }
    
    async def force_training_session(self) -> Dict[str, Any]:
        """Force an immediate training session"""
        if self.is_training:
            return {"success": False, "message": "Training already in progress"}
        
        if len(self.training_examples) < self.config.min_examples_before_training:
            return {
                "success": False, 
                "message": f"Not enough examples: {len(self.training_examples)} < {self.config.min_examples_before_training}"
            }
        
        # Execute training in background
        success = self._execute_training_session()
        
        return {
            "success": success,
            "message": "Training session completed" if success else "Training session failed",
            "metrics": self.training_metrics.copy()
        }
    
    def stop_training(self):
        """Stop background training"""
        self.should_stop_training = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        self.logger.info("🛑 BitNet online training stopped")


# Factory function for easy integration
def create_bitnet_online_trainer(
    agent_id: str = "bitnet_online_trainer",
    config: Optional[OnlineTrainingConfig] = None,
    consciousness_service: Optional[ConsciousnessService] = None
) -> BitNetOnlineTrainer:
    """Create a BitNet online trainer instance"""
    return BitNetOnlineTrainer(
        agent_id=agent_id,
        config=config,
        consciousness_service=consciousness_service
    )


# Example usage
async def main():
    """Example usage of BitNet Online Trainer"""
    trainer = create_bitnet_online_trainer()
    
    # Add training examples
    await trainer.add_training_example(
        prompt="Explain quantum entanglement",
        response="Quantum entanglement is a phenomenon where particles become correlated...",
        user_feedback=0.9
    )
    
    # Get training status
    status = await trainer.get_training_status()
    print(f"Training status: {status}")


if __name__ == "__main__":
    asyncio.run(main())