"""
🚀 Training Module - NIS Protocol v3.1

This module provides real-time training capabilities for NIS Protocol models:
- BitNet online training and fine-tuning
- Consciousness-guided training data quality assessment
- Physics-informed response validation for training
- Continuous model improvement while online
"""

from .bitnet_online_trainer import (
    BitNetOnlineTrainer,
    OnlineTrainingConfig,
    TrainingExample,
    create_bitnet_online_trainer
)

__all__ = [
    'BitNetOnlineTrainer',
    'OnlineTrainingConfig', 
    'TrainingExample',
    'create_bitnet_online_trainer'
]