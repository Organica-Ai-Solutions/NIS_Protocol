from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
import os
import requests
from enum import Enum

from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal

class MotorActionType(Enum):
    """Types of motor actions supported by the agent"""
    TEXT_OUTPUT = "text_output"
    COMMAND = "command"
    API_CALL = "api_call"
    EMOTIONAL_RESPONSE = "emotional_response"
    MEMORY_OPERATION = "memory_operation"
    LEARNING_UPDATE = "learning_update"
    NETWORK_ACTION = "network_action"
    SYSTEM_CONTROL = "system_control"
    MCP_TOOL_CALL = "mcp_tool_call"
    MCP_FUNCTION_CALL = "mcp_function_call"

@dataclass
class MotorAction:
    """Represents a motor action to be executed"""
    action_type: str
    parameters: Dict[str, Any]
    priority: float
    timestamp: datetime = datetime.now()
    status: str = "pending"  # pending, executing, completed, failed
    mcp_context: Dict[str, Any] = None  # Context for MCP-related actions

class MCPToolCall:
    """Handles MCP tool calls and responses"""
    
    def __init__(self, api_key: str = None):
        """Initialize MCP tool caller.
        
        Args:
            api_key: Optional API key for MCP. If not provided, will try to get from env.
        """
        self.api_key = api_key or os.getenv("MCP_API_KEY")
        if not self.api_key:
            raise ValueError("MCP API key not found")
            
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make an MCP tool call.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            
        Returns:
            Tool response
        """
        endpoint = f"{self.base_url}/tools/{tool_name}"
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=parameters
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"MCP tool call failed: {str(e)}")

class MotorAgent(NeuralAgent):
    """Agent responsible for output generation and action execution"""
    
    def __init__(
        self,
        agent_id: str = "motor_agent",
        action_queue_size: int = 10,
        execution_threshold: float = 0.6,
        mcp_api_key: str = None
    ):
        """Initialize the motor agent.
        
        Args:
            agent_id: Unique identifier for this agent
            action_queue_size: Maximum number of actions to queue
            execution_threshold: Minimum activation level to execute actions
            mcp_api_key: Optional API key for MCP
        """
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.MOTOR,
            description="Handles output generation and action execution"
        )
        
        self.action_queue_size = action_queue_size
        self.execution_threshold = execution_threshold
        self.action_queue: List[MotorAction] = []
        self.action_history: List[MotorAction] = []
        
        # Initialize MCP tool caller
        try:
            self.mcp_caller = MCPToolCall(mcp_api_key)
        except ValueError as e:
            print(f"Warning: MCP initialization failed: {str(e)}")
            self.mcp_caller = None
        
        # Track execution statistics
        self.total_actions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        
        # Tool registry
        self.registered_tools = {}
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """Register a new tool that can be called by the motor agent.
        
        Args:
            tool_name: Name of the tool
            tool_function: Function to call when tool is invoked
        """
        self.registered_tools[tool_name] = tool_function
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal to generate and execute actions"""
        if not isinstance(signal.content, dict):
            return None
            
        # Extract action information
        action = signal.content.get("action", {})
        action_type = action.get("type")
        parameters = action.get("parameters", {})
        context = action.get("context", {})
        
        if not action_type:
            return None
            
        # Create motor action
        motor_action = MotorAction(
            action_type=action_type,
            parameters=parameters,
            priority=signal.priority,
            mcp_context=context.get("mcp_context")
        )
        
        # Add to queue if space available
        if len(self.action_queue) < self.action_queue_size:
            self.action_queue.append(motor_action)
            
            # Try to execute actions if activation is high enough
            if self.is_active():
                self._execute_queued_actions()
                
            return NeuralSignal(
                source_layer=self.layer,
                target_layer=NeuralLayer.EXECUTIVE,
                content={
                    "status": "action_queued",
                    "action_type": action_type,
                    "queue_size": len(self.action_queue)
                },
                priority=signal.priority
            )
        else:
            # Queue full - notify executive layer
            return NeuralSignal(
                source_layer=self.layer,
                target_layer=NeuralLayer.EXECUTIVE,
                content={
                    "status": "queue_full",
                    "action_type": action_type,
                    "queue_size": len(self.action_queue)
                },
                priority=signal.priority
            )
    
    def _execute_queued_actions(self) -> None:
        """Execute all queued actions if activation level is sufficient"""
        if not self.is_active() or self.activation_level < self.execution_threshold:
            return
            
        while self.action_queue:
            action = self.action_queue.pop(0)
            success = self._execute_action(action)
            
            # Update statistics
            self.total_actions += 1
            if success:
                self.successful_actions += 1
                action.status = "completed"
            else:
                self.failed_actions += 1
                action.status = "failed"
                
            # Add to history
            self.action_history.append(action)
            
            # Limit history size
            if len(self.action_history) > 100:
                self.action_history = self.action_history[-100:]
    
    def _execute_action(self, action: MotorAction) -> bool:
        """Execute a single motor action.
        
        This method handles various action types including MCP tool calls
        and registered tool functions.
        
        Args:
            action: The motor action to execute
            
        Returns:
            True if execution was successful, False otherwise
        """
        try:
            action.status = "executing"
            
            # Handle MCP-specific actions
            if action.action_type == MotorActionType.MCP_TOOL_CALL.value:
                if not self.mcp_caller:
                    print("[Motor Error] MCP not initialized")
                    return False
                    
                tool_name = action.parameters.get("tool_name")
                tool_params = action.parameters.get("parameters", {})
                
                try:
                    result = self.mcp_caller.call_tool(tool_name, tool_params)
                    print(f"[Motor MCP] Tool call successful: {tool_name}")
                    return True
                except Exception as e:
                    print(f"[Motor MCP] Tool call failed: {str(e)}")
                    return False
            
            # Handle registered tool calls
            elif action.action_type == MotorActionType.MCP_FUNCTION_CALL.value:
                function_name = action.parameters.get("function_name")
                function_params = action.parameters.get("parameters", {})
                
                if function_name in self.registered_tools:
                    try:
                        self.registered_tools[function_name](**function_params)
                        print(f"[Motor Function] Called: {function_name}")
                        return True
                    except Exception as e:
                        print(f"[Motor Function] Call failed: {str(e)}")
                        return False
                else:
                    print(f"[Motor Error] Unknown function: {function_name}")
                    return False
            
            # Handle standard action types
            elif action.action_type == MotorActionType.TEXT_OUTPUT.value:
                text = action.parameters.get("text", "")
                print(f"[Motor Output] {text}")
                return True
                
            elif action.action_type == MotorActionType.COMMAND.value:
                command = action.parameters.get("command", "")
                # Implement command execution logic here
                print(f"[Motor Command] Executing: {command}")
                return True
                
            elif action.action_type == MotorActionType.API_CALL.value:
                endpoint = action.parameters.get("endpoint", "")
                payload = action.parameters.get("payload", {})
                # Implement API call logic here
                print(f"[Motor API] Calling {endpoint} with {payload}")
                return True
                
            elif action.action_type == MotorActionType.EMOTIONAL_RESPONSE.value:
                emotion = action.parameters.get("emotion", "neutral")
                intensity = action.parameters.get("intensity", 0.5)
                message = action.parameters.get("message", "")
                print(f"[Motor Emotional] {emotion.upper()} ({intensity:.2f}): {message}")
                return True
                
            elif action.action_type == MotorActionType.MEMORY_OPERATION.value:
                operation = action.parameters.get("operation", "")
                target = action.parameters.get("target", "")
                data = action.parameters.get("data", {})
                print(f"[Motor Memory] {operation} on {target}: {data}")
                return True
                
            elif action.action_type == MotorActionType.LEARNING_UPDATE.value:
                component = action.parameters.get("component", "")
                updates = action.parameters.get("updates", {})
                print(f"[Motor Learning] Updating {component}: {updates}")
                return True
                
            elif action.action_type == MotorActionType.NETWORK_ACTION.value:
                protocol = action.parameters.get("protocol", "")
                action_type = action.parameters.get("action_type", "")
                data = action.parameters.get("data", {})
                print(f"[Motor Network] {protocol} {action_type}: {data}")
                return True
                
            elif action.action_type == MotorActionType.SYSTEM_CONTROL.value:
                control_type = action.parameters.get("control_type", "")
                parameters = action.parameters.get("parameters", {})
                print(f"[Motor System] {control_type}: {parameters}")
                return True
                
            else:
                print(f"[Motor Error] Unknown action type: {action.action_type}")
                return False
                
        except Exception as e:
            print(f"[Motor Error] Action execution failed: {str(e)}")
            return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about action execution"""
        success_rate = (
            self.successful_actions / self.total_actions
            if self.total_actions > 0
            else 0.0
        )
        
        return {
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": success_rate,
            "queued_actions": len(self.action_queue),
            "activation_level": self.activation_level,
            "mcp_available": self.mcp_caller is not None,
            "registered_tools": list(self.registered_tools.keys())
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.action_queue.clear()
        self.action_history.clear()
        self.total_actions = 0
        self.successful_actions = 0
        self.failed_actions = 0 