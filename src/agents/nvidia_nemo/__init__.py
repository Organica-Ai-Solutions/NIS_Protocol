"""
NVIDIA NeMo Enterprise Integration for NIS Protocol

This module provides enterprise-grade AI capabilities using:
- NVIDIA NeMo Framework for model training and physics simulation
- NVIDIA NeMo Agent Toolkit for production agent orchestration
- NVIDIA Cosmos World Foundation Models for physical AI
- Model Context Protocol (MCP) for enterprise tool sharing

Key Components:
- NeMoPhysicsAgent: Real physics simulation with Cosmos models
- NeMoAgentOrchestrator: Enterprise multi-agent coordination
- Production deployment and observability tools
"""

from .nemo_physics_agent import (
    NeMoPhysicsAgent,
    NeMoPhysicsConfig,
    PhysicsSimulationType,
    create_nemo_physics_agent
)

from .nemo_agent_orchestrator import (
    NeMoAgentOrchestrator,
    NeMoAgentConfig,
    AgentFramework,
    ObservabilityProvider,
    create_nemo_agent_orchestrator
)

__all__ = [
    'NeMoPhysicsAgent',
    'NeMoPhysicsConfig', 
    'PhysicsSimulationType',
    'create_nemo_physics_agent',
    'NeMoAgentOrchestrator',
    'NeMoAgentConfig',
    'AgentFramework',
    'ObservabilityProvider',
    'create_nemo_agent_orchestrator'
]
