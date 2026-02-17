"""
NVIDIA Isaac GR00T N1 Integration for NIS Protocol

World's first open humanoid robot foundation model:
- Generalized humanoid reasoning and skills
- Multi-modal inputs (vision, language, proprioception)
- Pre-trained on diverse humanoid tasks
- Real-world deployment ready

Usage:
    from src.agents.groot import get_groot_agent
"""

from .groot_agent import GR00TAgent, get_groot_agent

__all__ = ['GR00TAgent', 'get_groot_agent']
