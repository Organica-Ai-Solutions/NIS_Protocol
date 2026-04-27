"""
NIS Protocol — Base Agent
Minimal base class used by reasoning and other agents.
"""
import logging
from typing import Any, Dict, Optional


class BaseAgent:
    """Minimal base class for all NIS Protocol agents."""

    def __init__(self, name: str = "base_agent"):
        self.name = name
        self.logger = logging.getLogger(f"nis.agents.{name}")
        self.initialized: bool = False

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def process(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def get_status(self) -> Dict[str, Any]:
        return {"name": self.name, "initialized": self.initialized}
