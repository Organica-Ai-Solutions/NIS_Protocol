"""
NIS Agents Package

This package contains all NIS agent implementations organized by their functional layers.
"""

# Core agents - conditional imports
try:
    from .action import ActionAgent, OutputAgent
    ACTION_AVAILABLE = True
except ImportError:
    ACTION_AVAILABLE = False
    ActionAgent = None
    OutputAgent = None

try:
    from .communication import CommunicationAgent, MessageAgent
    COMMUNICATION_AVAILABLE = True
except ImportError:
    COMMUNICATION_AVAILABLE = False
    CommunicationAgent = None
    MessageAgent = None

# Perception layer - conditional import
try:
    from .perception import VisionAgent, InputAgent
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    VisionAgent = None
    InputAgent = None

# Reasoning and coordination - conditional imports
try:
    from .reasoning import ReasoningAgent, InferenceAgent
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    ReasoningAgent = None
    InferenceAgent = None

try:
    from .coordination import CoordinatorAgent
    COORDINATION_AVAILABLE = True
except ImportError:
    COORDINATION_AVAILABLE = False
    CoordinatorAgent = None

# Memory and learning - conditional imports
try:
    from .memory import MemoryAgent
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    MemoryAgent = None

try:
    from .learning import LearningAgent
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    LearningAgent = None

# New AGI v2.0 agents - conditional imports
try:
    from .consciousness import ConsciousAgent
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    ConsciousAgent = None

try:
    from .goals import GoalGenerationAgent
    GOALS_AVAILABLE = True
except ImportError:
    GOALS_AVAILABLE = False
    GoalGenerationAgent = None

# Simulation agents - conditional imports
try:
    from .simulation import ScenarioSimulator, OutcomePredictor, RiskAssessor
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    ScenarioSimulator = None
    OutcomePredictor = None
    RiskAssessor = None

# Build __all__ list dynamically based on available components
__all__ = []

if ACTION_AVAILABLE:
    __all__.extend(["ActionAgent", "OutputAgent"])

if COMMUNICATION_AVAILABLE:
    __all__.extend(["CommunicationAgent", "MessageAgent"])

if PERCEPTION_AVAILABLE:
    __all__.extend(["VisionAgent", "InputAgent"])

if REASONING_AVAILABLE:
    __all__.extend(["ReasoningAgent", "InferenceAgent"])

if COORDINATION_AVAILABLE:
    __all__.extend(["CoordinatorAgent"])

if MEMORY_AVAILABLE:
    __all__.extend(["MemoryAgent"])

if LEARNING_AVAILABLE:
    __all__.extend(["LearningAgent"])

if CONSCIOUSNESS_AVAILABLE:
    __all__.extend(["ConsciousAgent"])

if GOALS_AVAILABLE:
    __all__.extend(["GoalGenerationAgent"])

if SIMULATION_AVAILABLE:
    __all__.extend(["ScenarioSimulator", "OutcomePredictor", "RiskAssessor"])

# Availability flags for external checking
AVAILABLE_MODULES = {
    "action": ACTION_AVAILABLE,
    "communication": COMMUNICATION_AVAILABLE,
    "perception": PERCEPTION_AVAILABLE,
    "reasoning": REASONING_AVAILABLE,
    "coordination": COORDINATION_AVAILABLE,
    "memory": MEMORY_AVAILABLE,
    "learning": LEARNING_AVAILABLE,
    "consciousness": CONSCIOUSNESS_AVAILABLE,
    "goals": GOALS_AVAILABLE,
    "simulation": SIMULATION_AVAILABLE
}
