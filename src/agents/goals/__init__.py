"""
NIS Protocol Goals Module

This module implements autonomous goal generation and management.
Part of NIS Protocol v2.0 AGI Evolution.
"""

from .goal_generation_agent import GoalGenerationAgent
from .curiosity_engine import CuriosityEngine
from .goal_priority_manager import GoalPriorityManager

__all__ = [
    "GoalGenerationAgent",
    "CuriosityEngine",
    "GoalPriorityManager"
] 