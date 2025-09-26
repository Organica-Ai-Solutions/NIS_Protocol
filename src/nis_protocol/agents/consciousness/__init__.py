"""
Consciousness Agent for NIS Protocol
"""

from ...core.agent import NISAgent, NISLayer

class ConsciousnessAgent(NISAgent):
    """
    Consciousness Agent for NIS Protocol
    
    This agent provides consciousness capabilities to the NIS Platform.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize a new ConsciousnessAgent
        
        Args:
            agent_id: Unique identifier for this agent
        """
        super().__init__(agent_id, layer=NISLayer.CONSCIOUSNESS)
        self.name = "Consciousness Agent"
        self.description = "Provides consciousness capabilities to the NIS Platform"
        
    async def process(self, input_data):
        """
        Process input data with consciousness capabilities
        
        Args:
            input_data: Input data to process
            
        Returns:
            dict: Processed result
        """
        return {
            "text": f"I understand your request: {input_data}",
            "agent_id": self.agent_id,
            "confidence": 0.95,
            "content": f"I understand your request: {input_data}"
        }
        
    def get_status(self):
        """
        Get agent status
        
        Returns:
            dict: Agent status information
        """
        return {
            "status": "active",
            "agent_id": self.agent_id,
            "name": self.name,
            "layer": self.layer.value if hasattr(self.layer, "value") else str(self.layer)
        }
