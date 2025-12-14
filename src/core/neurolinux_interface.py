"""
NeuroLinux Interface - NIS Protocol v4.0

This module defines the formal API contract between NIS Protocol and NeuroLinux.
These types are the ONLY interface that NeuroLinux should depend on.

See docs/API_CONTRACT.md for full specification.

Principle: NIS Protocol produces action plans. NeuroLinux executes them.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


class SafetyLevel(Enum):
    """
    Command safety classification.
    
    Defined by NIS Protocol, enforced by NeuroLinux.
    """
    SAFE = "safe"           # Read-only, no side effects
    PRIVILEGED = "privileged"  # Requires elevated permissions
    HARDWARE = "hardware"    # Interacts with physical hardware
    CRITICAL = "critical"    # Requires explicit operator confirmation


class ActionType(Enum):
    """Types of actions NIS Protocol can request."""
    QUERY = "query"          # Information retrieval
    COMMAND = "command"      # System command execution
    AGENT_INVOKE = "agent"   # Invoke an agent
    HARDWARE_CONTROL = "hw"  # Hardware interaction
    STATE_CHANGE = "state"   # System state modification


class ExecutionStatus(Enum):
    """Execution outcome status."""
    SUCCESS = "success"
    PARTIAL = "partial"      # Some actions succeeded
    FAILED = "failed"
    DENIED = "denied"        # Permission denied
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"  # User cancelled confirmation


@dataclass
class Action:
    """
    Single action in an action plan.
    
    NIS Protocol creates these; NeuroLinux executes them.
    """
    action_id: str
    action_type: ActionType
    target: str              # What to act on
    parameters: Dict[str, Any]
    safety_level: SafetyLevel
    timeout_ms: int = 30000
    retry_count: int = 0


@dataclass
class ActionPlan:
    """
    Complete action plan from NIS Protocol.
    
    This is the primary output of cognitive processing.
    NeuroLinux receives this and executes it.
    """
    plan_id: str
    intent: str              # Original user intent
    safety_level: SafetyLevel  # Overall plan safety level (max of all actions)
    agents: List[str]        # Agents involved in creating this plan
    actions: List[Action]    # Ordered list of actions to execute
    rationale: str           # Why this plan was chosen (for explainability)
    requires_confirmation: bool
    estimated_duration_ms: int
    rollback_actions: Optional[List[Action]] = None


@dataclass
class ActionResult:
    """Result of a single action execution."""
    action_id: str
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    audit_id: Optional[str] = None


@dataclass
class ExecutionResult:
    """
    Complete execution result from NeuroLinux.
    
    Returned after executing an ActionPlan.
    """
    plan_id: str
    status: ExecutionStatus
    action_results: List[ActionResult]
    total_duration_ms: int
    audit_id: str            # Master audit entry ID
    confirmation_id: Optional[str] = None
    rollback_executed: bool = False


@dataclass
class AgentCapability:
    """What an agent can do."""
    name: str
    description: str
    action_types: List[ActionType]
    safety_levels: List[SafetyLevel]  # Max safety levels it can handle
    hardware_required: List[str] = field(default_factory=list)


@dataclass
class AgentRegistration:
    """Agent registration with NeuroLinux."""
    agent_id: str
    agent_name: str
    agent_type: str          # embodiment, cognitive, hardware
    capabilities: List[AgentCapability]
    version: str
    source: str = "nis_protocol"  # or "neuroforge"


@dataclass
class AuditEntry:
    """
    Audit log entry.
    
    Every action is logged with causality tracking.
    """
    audit_id: str
    timestamp: datetime
    command: str             # Original command/intent
    action: str              # What was executed
    safety_level: SafetyLevel
    success: bool
    duration_ms: int
    result_summary: str
    rationale: str
    triggered_by: Optional[str] = None  # Parent audit_id for causality
    user_id: Optional[str] = None
    device_id: Optional[str] = None


@dataclass
class HealthCheck:
    """Individual health check."""
    name: str
    value: float
    threshold: Optional[float] = None
    passed: bool = True


@dataclass
class SystemHealth:
    """System health status for OTA gating."""
    healthy: bool
    checks: List[HealthCheck]
    blocking_issues: List[str] = field(default_factory=list)


# Classification helpers

CRITICAL_ACTIONS = {
    "reboot", "shutdown", "ota_update", "factory_reset",
    "can_emergency_stop", "robotics_command"
}

HARDWARE_ACTIONS = {
    "can_send", "camera_control", "gpio_write", "i2c_write",
    "spi_write", "uart_write"
}

PRIVILEGED_ACTIONS = {
    "run_command", "restart_service", "stop_service",
    "install_package", "modify_config"
}


def classify_action(action: str) -> SafetyLevel:
    """
    Classify an action's safety level.
    
    This is the canonical classification used by both
    NIS Protocol (to plan) and NeuroLinux (to enforce).
    """
    action_lower = action.lower()
    
    for critical in CRITICAL_ACTIONS:
        if critical in action_lower:
            return SafetyLevel.CRITICAL
    
    for hw in HARDWARE_ACTIONS:
        if hw in action_lower:
            return SafetyLevel.HARDWARE
    
    for priv in PRIVILEGED_ACTIONS:
        if priv in action_lower:
            return SafetyLevel.PRIVILEGED
    
    return SafetyLevel.SAFE


def requires_confirmation(safety_level: SafetyLevel) -> bool:
    """Check if a safety level requires operator confirmation."""
    return safety_level == SafetyLevel.CRITICAL


# Export all public types
__all__ = [
    # Enums
    "SafetyLevel",
    "ActionType", 
    "ExecutionStatus",
    # Action types
    "Action",
    "ActionPlan",
    "ActionResult",
    "ExecutionResult",
    # Agent types
    "AgentCapability",
    "AgentRegistration",
    # Audit types
    "AuditEntry",
    # Health types
    "HealthCheck",
    "SystemHealth",
    # Classification
    "CRITICAL_ACTIONS",
    "HARDWARE_ACTIONS",
    "PRIVILEGED_ACTIONS",
    "classify_action",
    "requires_confirmation",
]
