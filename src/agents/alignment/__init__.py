"""
NIS Protocol Alignment Agent Module

This module contains the alignment agent components for ethical reasoning,
value alignment, and safety monitoring in the AGI system.
"""

from .ethical_reasoner import EthicalReasoner
from .value_alignment import ValueAlignment
from .safety_monitor import SafetyMonitor

__all__ = [
    "EthicalReasoner",
    "ValueAlignment",
    "SafetyMonitor"
] 