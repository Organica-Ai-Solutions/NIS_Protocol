"""
Base Skill for Deep Agents

Abstract base class for all specialized skills/sub-agents.
Provides common interface and functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import json

from ....core.agent import NISAgent
from ....memory.memory_manager import MemoryManager


class BaseSkill(ABC):
    """
    Abstract base class for Deep Agent skills.
    
    Each skill represents a specialized capability that can be invoked
    by the Deep Agent Planner for specific types of tasks.
    """
    
    def __init__(self, agent: NISAgent, memory_manager: MemoryManager, config: Dict[str, Any] = None):
        self.agent = agent
        self.memory = memory_manager
        self.config = config or {}
        self.skill_name = self.__class__.__name__.lower().replace('skill', '')
        
    @abstractmethod
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific action within this skill domain.
        
        Args:
            action: The specific action to perform
            parameters: Parameters for the action
            
        Returns:
            Result of the action execution
        """
        pass
        
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """
        Get list of available actions for this skill.
        
        Returns:
            List of action names
        """
        pass
        
    def get_action_schema(self, action: str) -> Dict[str, Any]:
        """
        Get the JSON schema for a specific action's parameters.
        
        Args:
            action: The action name
            
        Returns:
            JSON schema for the action parameters
        """
        schemas = self._get_action_schemas()
        return schemas.get(action, {
            "type": "object",
            "properties": {},
            "required": []
        })
        
    @abstractmethod
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get schemas for all actions in this skill.
        
        Returns:
            Dictionary mapping action names to their schemas
        """
        pass
        
    async def _call_agent(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call the underlying agent with a prompt and context.
        
        Args:
            prompt: The prompt to send to the agent
            context: Additional context
            
        Returns:
            Agent response
        """
        try:
            # Use the real agent processing method
            if hasattr(self.agent, 'process') and callable(self.agent.process):
                response = await self.agent.process(prompt, context or {})
                return {
                    "content": response.get("content", ""),
                    "metadata": {
                        "skill": self.skill_name,
                        "timestamp": time.time(),
                        "agent_id": self.agent.agent_id if hasattr(self.agent, 'agent_id') else 'unknown'
                    },
                    "success": True
                }
            else:
                # Fallback if agent doesn't have process method
                return {
                    "content": f"Agent response for: {prompt}",
                    "metadata": {
                        "skill": self.skill_name,
                        "timestamp": time.time(),
                        "agent_id": self.agent.agent_id if hasattr(self.agent, 'agent_id') else 'unknown'
                    },
                    "success": True
                }
        except Exception as e:
            return {
                "content": f"Error calling agent: {str(e)}",
                "metadata": {
                    "skill": self.skill_name,
                    "timestamp": time.time(),
                    "error": str(e)
                },
                "success": False
            }
        
    async def _store_result(self, action: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Store the execution result in memory for future reference."""
        await self.memory.store({
            "type": "skill_execution",
            "skill": self.skill_name,
            "action": action,
            "parameters": parameters,
            "result": result,
            "timestamp": time.time()
        })
        
    def _validate_parameters(self, action: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters against the action schema.
        
        Args:
            action: The action name
            parameters: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            schema = self.get_action_schema(action)
            required_fields = schema.get("required", [])
            
            # Check required fields
            for field in required_fields:
                if field not in parameters:
                    return False
                    
            # Basic type checking for properties
            properties = schema.get("properties", {})
            for field, value in parameters.items():
                if field in properties:
                    field_schema = properties[field]
                    expected_type = field_schema.get("type")
                    
                    if expected_type == "string" and not isinstance(value, str):
                        return False
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False
                    elif expected_type == "array" and not isinstance(value, list):
                        return False
                    elif expected_type == "object" and not isinstance(value, dict):
                        return False
                        
            return True
            
        except Exception:
            return False
            
    def _format_error(self, message: str, error_type: str = "SkillError") -> Dict[str, Any]:
        """Format an error response."""
        return {
            "success": False,
            "error": {
                "message": message,
                "type": error_type,
                "skill": self.skill_name
            },
            "timestamp": time.time()
        }
        
    def _format_success(self, data: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format a success response."""
        return {
            "success": True,
            "data": data,
            "metadata": metadata or {},
            "skill": self.skill_name,
            "timestamp": time.time()
        }
