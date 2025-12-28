"""
NIS Protocol Core Components

This package contains the core components of the NIS Protocol.
"""

from .agent import NISAgent
from .nvidia_integration import NVIDIAStackIntegration, get_nvidia_integration, initialize_nvidia_stack

# StateManager import with fallback
try:
    from .state_manager import StateManager
    _state_manager_available = True
except ImportError:
    StateManager = None
    _state_manager_available = False

__all__ = [
    'NISAgent',
    'NVIDIAStackIntegration',
    'get_nvidia_integration',
    'initialize_nvidia_stack'
]

if _state_manager_available:
    __all__.append('StateManager')