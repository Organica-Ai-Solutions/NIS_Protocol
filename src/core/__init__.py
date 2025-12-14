"""
NIS Protocol Core Components

This package contains the core components of the NIS Protocol.

For NeuroLinux integration, import from neurolinux_interface:
    from nis_protocol.core.neurolinux_interface import (
        SafetyLevel, ActionPlan, ExecutionResult, classify_action
    )
"""

from .agent import NISAgent, NISLayer
from .registry import NISRegistry

# NeuroLinux Interface exports
from .neurolinux_interface import (
    SafetyLevel,
    ActionType,
    ExecutionStatus,
    Action,
    ActionPlan,
    ActionResult,
    ExecutionResult,
    AgentCapability,
    AgentRegistration,
    AuditEntry,
    HealthCheck,
    SystemHealth,
    classify_action,
    requires_confirmation,
)

__all__ = [
    # Core NIS types
    "NISAgent", 
    "NISLayer", 
    "NISRegistry",
    # NeuroLinux Interface types
    "SafetyLevel",
    "ActionType",
    "ExecutionStatus",
    "Action",
    "ActionPlan",
    "ActionResult",
    "ExecutionResult",
    "AgentCapability",
    "AgentRegistration",
    "AuditEntry",
    "HealthCheck",
    "SystemHealth",
    "classify_action",
    "requires_confirmation",
] 