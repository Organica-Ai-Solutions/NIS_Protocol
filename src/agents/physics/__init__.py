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

from .conservation_laws import ConservationLaws, ConservationLawValidator
from .electromagnetism import MaxwellEquationsValidator
from .thermodynamics import ThermodynamicsValidator
from .quantum_mechanics import QuantumMechanicsValidator
from .unified_physics_agent import PhysicsAgent as PhysicsInformedAgent, PhysicsDomain, PhysicsState, ViolationType as PhysicsViolation

__all__ = [
    "ConservationLaws",
    "ConservationLawValidator",
    "MaxwellEquationsValidator",
    "ThermodynamicsValidator",
    "QuantumMechanicsValidator",
    "PhysicsInformedAgent",
    "PhysicsDomain",
    "PhysicsState",
    "PhysicsViolation",
] 