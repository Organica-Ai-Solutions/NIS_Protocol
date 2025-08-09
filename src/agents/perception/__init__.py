"""
Perception Layer Agents

Contains agents responsible for processing sensory inputs including:
- VisionAgent: Processes visual inputs (images, video, UI)
- InputAgent: Handles user inputs and interface interactions
"""

try:
    from .vision_agent import VisionAgent
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    VisionAgent = None

try:
    from .input_agent import InputAgent
    INPUT_AVAILABLE = True
except ImportError:
    INPUT_AVAILABLE = False
    InputAgent = None

__all__ = []

if VISION_AVAILABLE:
    __all__.append('VisionAgent')
if INPUT_AVAILABLE:
    __all__.append('InputAgent') 