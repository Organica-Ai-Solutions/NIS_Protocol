"""
NIS Protocol - A biologically inspired framework for multi-agent systems.

This package provides the core components for building intelligent systems
based on the neuro-inspired system protocol.
"""

__version__ = "0.1.0"

# Core components
from src.core.agent import NISAgent, NISLayer
from src.core.registry import NISRegistry

# Emotional state system
from src.emotion.emotional_state import EmotionalStateSystem

# Memory system
from src.memory.memory_manager import MemoryManager

# Coordinator agent
from src.agents.coordination.coordinator_agent import CoordinatorAgent

# Protocol adapters
from src.adapters.bootstrap import configure_coordinator_agent 