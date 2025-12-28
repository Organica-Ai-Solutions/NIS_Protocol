"""
NVIDIA Isaac Lab 2.2 Integration for NIS Protocol

Unified framework for robot learning built on Isaac Sim:
- 16+ robot models (manipulators, quadrupeds, humanoids)
- 30+ ready-to-train environments
- GPU-accelerated RL/IL/Motion planning
- Integration with popular RL frameworks (PPO, SAC, etc.)

Usage:
    from src.agents.isaac_lab import get_isaac_lab_trainer
"""

from .isaac_lab_trainer import IsaacLabTrainer, get_isaac_lab_trainer

__all__ = ['IsaacLabTrainer', 'get_isaac_lab_trainer']
