"""
NIS Protocol Agents Package

This package contains implementations of specialized agents for each
cognitive layer of the NIS Protocol.
"""

from .perception import VisionAgent, InputAgent
from .interpretation import ParserAgent, IntentAgent
from .reasoning import CortexAgent, PlanningAgent
from .memory import MemoryAgent, LogAgent
from .action import BuilderAgent, DeployerAgent
from .learning import LearningAgent, OptimizerAgent
from .coordination import CoordinatorAgent

__all__ = [
    "VisionAgent",
    "InputAgent",
    "ParserAgent",
    "IntentAgent",
    "CortexAgent",
    "PlanningAgent",
    "MemoryAgent",
    "LogAgent",
    "BuilderAgent",
    "DeployerAgent",
    "LearningAgent",
    "OptimizerAgent",
    "CoordinatorAgent"
]
