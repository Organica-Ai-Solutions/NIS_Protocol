"""
NIS Protocol Core Components

This package contains the core components of the NIS Protocol.
"""

from .agent import NISAgent, NISLayer
from .registry import NISRegistry

__all__ = ["NISAgent", "NISLayer", "NISRegistry"] 