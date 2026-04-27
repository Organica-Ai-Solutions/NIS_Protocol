"""
NIS Protocol Core Components

This package contains the core components of the NIS Protocol.

For NeuroLinux integration, import from neurolinux_interface:
    from nis_protocol.core.neurolinux_interface import (
        SafetyLevel, ActionPlan, ExecutionResult, classify_action
    )`r`nNeuroKernel v2 components (all lazily imported, zero-crash on missing deps):
  NeuroKernel            — Central processing pipeline (scan→skills→loop→execute→audit)
  SkillLoader            — SKILL.md hot-reloading knowledge injection
  AuditChain             — Merkle hash-chain tamper-proof audit log
  LoopGuard              — SHA256 circuit breaker for autonomous loops
  DriveScheduler         — Autonomous scheduled drives (Hands equivalent)
  PromptInjectionScanner — 23-pattern threat detection
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

# NeuroKernel v2 — lazy imports so nothing crashes at import time
try:
    from .neurokernel import NeuroKernel, get_neurokernel
    from .skill_loader import SkillLoader, get_skill_loader
    from .audit_chain import AuditChain, get_audit_chain
    from .loop_guard import LoopGuard, get_loop_guard, CircuitBreakerTripped
    from .drive_scheduler import DriveScheduler, Drive, DriveScheduler, get_drive_scheduler
    from .prompt_injection_scanner import PromptInjectionScanner, get_scanner
    _neurokernel_available = True
except ImportError as _nke:
    _neurokernel_available = False
    NeuroKernel = None
    SkillLoader = None
    AuditChain = None
    LoopGuard = None
    DriveScheduler = None
    PromptInjectionScanner = None

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
    "requires_confirmation",`r`n# Base
    'NISAgent',
    'NVIDIAStackIntegration',
    'get_nvidia_integration',
    'initialize_nvidia_stack',
    # NeuroKernel v2
    'NeuroKernel', 'get_neurokernel',
    'SkillLoader', 'get_skill_loader',
    'AuditChain', 'get_audit_chain',
    'LoopGuard', 'get_loop_guard', 'CircuitBreakerTripped',
    'DriveScheduler', 'get_drive_scheduler',
    'PromptInjectionScanner', 'get_scanner',
]

