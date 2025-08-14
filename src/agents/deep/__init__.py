"""
Deep Agents Module for NIS Protocol

This module implements LangChain-style Deep Agents for complex multi-step reasoning,
planning, and sub-agent orchestration. It provides the intelligence layer that works
with MCP tools and mcp-ui interfaces.
"""

from .planner import DeepAgentPlanner
from .skills import (
    DatasetSkill,
    PipelineSkill,
    ResearchSkill,
    AuditSkill,
    CodeSkill
)

__all__ = [
    'DeepAgentPlanner',
    'DatasetSkill',
    'PipelineSkill', 
    'ResearchSkill',
    'AuditSkill',
    'CodeSkill'
]
