"""
Motor Layer Agents

This module contains agents for the Motor layer, which is responsible for
output generation and action execution in the neural hierarchy.

The Motor layer supports the following action types:

1. Basic Actions:
   - text_output: Generate text-based output
   - command: Execute system commands
   - api_call: Make external API calls

2. Cognitive Actions:
   - emotional_response: Generate emotionally-aware responses
   - memory_operation: Perform memory-related actions
   - learning_update: Update learning parameters

3. System Actions:
   - network_action: Execute network-related operations
   - system_control: Perform system control actions

4. MCP Integration:
   - mcp_tool_call: Execute MCP tool calls via Anthropic's API
   - mcp_function_call: Execute registered tool functions

The Motor layer now includes full integration with Anthropic's MCP (Managed Compute Protocol),
allowing it to:
- Make direct MCP tool calls using API authentication
- Register and execute local tool functions
- Handle tool-specific contexts and parameters
- Maintain execution statistics for both local and MCP tools

Each action type requires specific parameters and follows the MotorAction
data structure for execution management.
"""

from .motor_agent import MotorAgent, MotorAction, MotorActionType, MCPToolCall

__all__ = ["MotorAgent", "MotorAction", "MotorActionType", "MCPToolCall"] 