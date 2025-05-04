"""
Perception Layer Agents

This module contains agents for the Perception layer, which is responsible for
processing raw sensory inputs into structured data for higher-level processing.
"""

from .vision_agent import VisionAgent
from .input_agent import InputAgent

__all__ = ["VisionAgent", "InputAgent"] 