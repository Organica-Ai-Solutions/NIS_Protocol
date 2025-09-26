"""
NIS Protocol Agents
"""

# Import agents for easier access
try:
    from .consciousness import ConsciousnessAgent
except ImportError:
    ConsciousnessAgent = None
    
try:
    from .physics import PhysicsAgent
except ImportError:
    PhysicsAgent = None
