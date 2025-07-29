"""
NIS Layer Enum - NIS Protocol v3
Defines the architectural layers within the NIS Protocol.
"""
from enum import Enum

class NISLayer(Enum):
    """Defines the architectural layers within the NIS Protocol."""
    CORE = "core"  # Core agent functionalities and messaging
    INFRASTRUCTURE = "infrastructure"  # Caching, logging, and resource management
    SIGNAL_PROCESSING = "signal_processing"  # Data transformation and analysis
    REASONING = "reasoning"  # Logical inference and decision-making
    PHYSICS = "physics"  # Physics-informed validation and simulation
    CONSCIOUSNESS = "consciousness"  # Meta-cognition and self-awareness
    LEARNING = "learning"  # Adaptation and model training
    INTERPRETATION = "interpretation"  # Semantic understanding of data
    META = "meta"  # System-level coordination and orchestration 