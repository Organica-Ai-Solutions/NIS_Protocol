"""
MCP Integration Module for NIS Protocol

This module provides Model Context Protocol (MCP) integration with Deep Agents
and mcp-ui for interactive agent experiences.
"""

from .server import MCPServer
from .schemas import ToolSchemas
from .ui_resources import UIResourceGenerator
from .intent_validator import IntentValidator

__all__ = [
    'MCPServer',
    'ToolSchemas',
    'UIResourceGenerator', 
    'IntentValidator'
]
