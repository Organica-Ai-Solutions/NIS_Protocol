"""
Physics Agents Module - V3.0 Physics-Informed Intelligence

This module implements physics-aware AI agents that enforce physical laws
and constraints through Physics-Informed Neural Networks (PINNs) and 
NVIDIA Nemo integration.

Key Components:
- PhysicsInformedAgent: PINN-based physics constraint validation
- NemoPhysicsProcessor: NVIDIA Nemo integration for physics modeling
- ConservationLaws: Energy, momentum, and mass conservation enforcement
- PhysicsConstraints: Real-time physics law validation
"""

from .physics_agent import PhysicsInformedAgent
from .nemo_physics_processor import NemoPhysicsProcessor
from .conservation_laws import ConservationLaws

__all__ = [
    'PhysicsInformedAgent',
    'NemoPhysicsProcessor', 
    'ConservationLaws'
] 