"""
NIS Protocol - AI Development Platform & SDK
=============================================

The foundational AI operating system for edge devices, autonomous systems, and smart infrastructure.

Quick Start
-----------

.. code-block:: python

    from nis_protocol import NISAgent, NISPlatform
    from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent
    
    # Create a platform instance
    platform = NISPlatform()
    
    # Add agents to your system
    consciousness = ConsciousnessAgent("consciousness_001")
    physics = PhysicsAgent("physics_validator")
    
    platform.add_agent(consciousness)
    platform.add_agent(physics)
    
    # Deploy to edge device
    platform.deploy("edge", device_type="raspberry_pi")

Core Components
--------------

- **NISAgent**: Base class for all intelligent agents
- **NISPlatform**: Central coordination and deployment platform  
- **Protocols**: Integration with third-party AI agent ecosystems
- **Deployment**: Edge, cloud, and hybrid deployment tools
- **CLI Tools**: Command-line interface for platform management

Use Cases
---------

- **Edge AI**: Deploy to Raspberry Pi, embedded systems
- **Robotics**: Autonomous robots with physics validation
- **Drones**: Intelligent UAV control systems
- **Smart Cities**: Distributed infrastructure AI
- **Industrial**: Factory automation and quality control
- **IoT**: Sensor networks and intelligent devices

"""

# Version Information
__version__ = "3.2.0"
__author__ = "Organica AI Solutions"
__email__ = "developers@organicaai.com"
__license__ = "Business Source License 1.1"
__description__ = "AI Development Platform & SDK for Edge Devices, Autonomous Systems, and Smart Infrastructure"

# Core Imports
from .core.agent import NISAgent, NISLayer
from .core.platform import NISPlatform
from .core.registry import AgentRegistry
from .core.messaging import MessageBus

# Agent Imports
try:
    from .agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent as ConsciousnessAgent
except ImportError:
    ConsciousnessAgent = None

try:
    from .agents.physics.unified_physics_agent import UnifiedPhysicsAgent as PhysicsAgent
except ImportError:
    PhysicsAgent = None

try:
    from .agents.reasoning.unified_reasoning_agent import UnifiedReasoningAgent as ReasoningAgent
except ImportError:
    ReasoningAgent = None

try:
    from .agents.memory.enhanced_memory_agent import EnhancedMemoryAgent as MemoryAgent
except ImportError:
    MemoryAgent = None

try:
    from .agents.vision.vision_agent import VisionAgent
except ImportError:
    VisionAgent = None

# Protocol Adapters
try:
    from .protocols.mcp_adapter import MCPAdapter
    from .protocols.acp_adapter import ACPAdapter
    from .protocols.a2a_adapter import A2AAdapter
except ImportError:
    MCPAdapter = None
    ACPAdapter = None
    A2AAdapter = None

# Deployment Tools
try:
    from .deployment.edge import EdgeDeployment
    from .deployment.docker import DockerDeployment
    from .deployment.k8s import KubernetesDeployment
except ImportError:
    EdgeDeployment = None
    DockerDeployment = None 
    KubernetesDeployment = None

# Utilities
from .utils.confidence_calculator import calculate_confidence
from .utils.integrity_metrics import IntegrityMetrics

# Public API
__all__ = [
    # Version Info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__description__",
    
    # Core Platform
    "NISAgent",
    "NISLayer", 
    "NISPlatform",
    "AgentRegistry",
    "MessageBus",
    
    # Agents (if available)
    "ConsciousnessAgent",
    "PhysicsAgent", 
    "ReasoningAgent",
    "MemoryAgent",
    "VisionAgent",
    
    # Protocols (if available)
    "MCPAdapter",
    "ACPAdapter", 
    "A2AAdapter",
    
    # Deployment (if available)
    "EdgeDeployment",
    "DockerDeployment",
    "KubernetesDeployment",
    
    # Utilities
    "calculate_confidence",
    "IntegrityMetrics",
]

# Platform Health Check
def health_check():
    """
    Perform a basic health check of the NIS Protocol installation.
    
    Returns:
        dict: Health status information
    """
    health = {
        "version": __version__,
        "platform_available": True,
        "core_agents": {},
        "protocols": {},
        "deployment": {},
    }
    
    # Check core agents
    health["core_agents"]["consciousness"] = ConsciousnessAgent is not None
    health["core_agents"]["physics"] = PhysicsAgent is not None
    health["core_agents"]["reasoning"] = ReasoningAgent is not None
    health["core_agents"]["memory"] = MemoryAgent is not None
    health["core_agents"]["vision"] = VisionAgent is not None
    
    # Check protocols
    health["protocols"]["mcp"] = MCPAdapter is not None
    health["protocols"]["acp"] = ACPAdapter is not None
    health["protocols"]["a2a"] = A2AAdapter is not None
    
    # Check deployment
    health["deployment"]["edge"] = EdgeDeployment is not None
    health["deployment"]["docker"] = DockerDeployment is not None
    health["deployment"]["kubernetes"] = KubernetesDeployment is not None
    
    return health

# Platform Information
def platform_info():
    """
    Get comprehensive platform information.
    
    Returns:
        dict: Platform capabilities and configuration
    """
    return {
        "name": "NIS Protocol",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "capabilities": {
            "edge_deployment": True,
            "physics_validation": True,
            "consciousness_modeling": True,
            "multi_agent_coordination": True,
            "protocol_integration": True,
            "hybrid_intelligence": True,
        },
        "supported_devices": [
            "Raspberry Pi",
            "NVIDIA Jetson",
            "Intel NUC",
            "Generic Linux",
            "Docker Containers",
            "Kubernetes Clusters",
        ],
        "use_cases": [
            "Autonomous Robotics",
            "Smart Cities",
            "Industrial Automation", 
            "IoT Networks",
            "Drone Systems",
            "Edge AI Applications",
        ],
    }

# Welcome Message
def welcome():
    """Print welcome message for new users."""
    print(f"""
🧠 Welcome to NIS Protocol v{__version__}!
==========================================

The AI Development Platform for Edge Devices, Autonomous Systems, and Smart Infrastructure.

Quick Start:
  1. Initialize a new project:  nis-init my-ai-project
  2. Create an agent:          nis-agent create MyAgent
  3. Deploy to edge:           nis-deploy edge --device raspberry-pi
  4. Start platform:           nis-serve

Documentation: https://docs.nis-protocol.org
Examples:      https://github.com/Organica-Ai-Solutions/NIS_Protocol/examples
Community:     https://community.nis-protocol.org

Ready to build the future of AI! 🚀
""")

# Development Status Warning
import warnings

if "dev" in __version__ or "alpha" in __version__ or "beta" in __version__:
    warnings.warn(
        f"You are using a development version of NIS Protocol ({__version__}). "
        "This version may contain bugs and is not recommended for production use. "
        "Please use a stable release for production deployments.",
        UserWarning,
        stacklevel=2
    )
