"""
🚀 Autonomous Execution Module - NIS Protocol v3.1

This module provides Anthropic-level autonomous execution capabilities:
- Multi-step reasoning with tool orchestration
- Self-reflection and course correction  
- Goal-driven autonomous behavior
- Human-in-the-loop decision making
"""

from .executor import (
    AnthropicStyleExecutor,
    ExecutionStrategy,
    ExecutionMode,
    ToolCategory,
    ExecutionContext,
    ExecutionStep,
    ExecutionPlan,
    ReflectionInsight,
    create_anthropic_style_executor
)

__all__ = [
    'AnthropicStyleExecutor',
    'ExecutionStrategy', 
    'ExecutionMode',
    'ToolCategory',
    'ExecutionContext',
    'ExecutionStep', 
    'ExecutionPlan',
    'ReflectionInsight',
    'create_anthropic_style_executor'
]