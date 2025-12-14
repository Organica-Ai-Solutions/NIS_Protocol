# NIS Protocol ↔ NeuroLinux API Contract

**Version:** 1.0  
**Date:** December 13, 2025  
**Status:** Specification

---

## Overview

This document defines the formal API contract between NIS Protocol (open source cognitive layer) and NeuroLinux (closed source execution layer).

**Principle:** NIS Protocol produces *action plans*. NeuroLinux *executes* them.

---

## 1. Action Plan Interface

NIS Protocol produces an `ActionPlan` that NeuroLinux executes.

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class SafetyLevel(Enum):
    """Command safety classification - defined by NIS Protocol"""
    SAFE = "safe"           # Read-only, no side effects
    PRIVILEGED = "privileged"  # Requires elevated permissions
    HARDWARE = "hardware"    # Interacts with physical hardware
    CRITICAL = "critical"    # Requires explicit confirmation

class ActionType(Enum):
    """Types of actions NIS Protocol can request"""
    QUERY = "query"          # Information retrieval
    COMMAND = "command"      # System command execution
    AGENT_INVOKE = "agent"   # Invoke an agent
    HARDWARE_CONTROL = "hw"  # Hardware interaction
    STATE_CHANGE = "state"   # System state modification

@dataclass
class Action:
    """Single action in an action plan"""
    action_id: str
    action_type: ActionType
    target: str              # What to act on
    parameters: Dict[str, Any]
    safety_level: SafetyLevel
    timeout_ms: int = 30000
    retry_count: int = 0

@dataclass
class ActionPlan:
    """Complete action plan from NIS Protocol"""
    plan_id: str
    intent: str              # Original user intent
    safety_level: SafetyLevel  # Overall plan safety level
    agents: List[str]        # Agents involved
    actions: List[Action]    # Ordered list of actions
    rationale: str           # Why this plan was chosen
    requires_confirmation: bool
    estimated_duration_ms: int
    rollback_actions: Optional[List[Action]] = None
```

---

## 2. Execution Result Interface

NeuroLinux returns an `ExecutionResult` after executing an action plan.

```python
from datetime import datetime

class ExecutionStatus(Enum):
    """Execution outcome status"""
    SUCCESS = "success"
    PARTIAL = "partial"      # Some actions succeeded
    FAILED = "failed"
    DENIED = "denied"        # Permission denied
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"  # User cancelled confirmation

@dataclass
class ActionResult:
    """Result of a single action"""
    action_id: str
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    audit_id: Optional[str] = None

@dataclass
class ExecutionResult:
    """Complete execution result from NeuroLinux"""
    plan_id: str
    status: ExecutionStatus
    action_results: List[ActionResult]
    total_duration_ms: int
    audit_id: str            # Master audit entry ID
    confirmation_id: Optional[str] = None  # If confirmation was required
    rollback_executed: bool = False
```

---

## 3. Agent Registry Interface

NIS Protocol registers agents with NeuroLinux.

```python
@dataclass
class AgentCapability:
    """What an agent can do"""
    name: str
    description: str
    action_types: List[ActionType]
    safety_levels: List[SafetyLevel]  # Max safety levels it can handle
    hardware_required: List[str] = []  # e.g., ["camera", "can_bus"]

@dataclass
class AgentRegistration:
    """Agent registration with NeuroLinux"""
    agent_id: str
    agent_name: str
    agent_type: str          # embodiment, cognitive, hardware
    capabilities: List[AgentCapability]
    version: str
    source: str              # "nis_protocol" or "neuroforge"
```

---

## 4. REST API Endpoints

### NeuroLinux Core API (Port 8080)

#### Execute Action Plan
```
POST /v4/nis/execute
Content-Type: application/json

Request:
{
  "plan": ActionPlan
}

Response:
{
  "result": ExecutionResult
}
```

#### Register Agent
```
POST /v4/nis/agents/register
Content-Type: application/json

Request:
{
  "registration": AgentRegistration
}

Response:
{
  "success": true,
  "agent_id": "string"
}
```

#### Get Safety Level for Intent
```
POST /v4/nis/classify
Content-Type: application/json

Request:
{
  "intent": "string"
}

Response:
{
  "safety_level": "safe|privileged|hardware|critical",
  "requires_confirmation": boolean,
  "rationale": "string"
}
```

---

## 5. Confirmation Flow

For `CRITICAL` safety level actions:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ NIS Protocol│────▶│ NeuroLinux  │────▶│   Operator  │
│             │     │   Core      │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │  ActionPlan       │                   │
      │  (critical)       │                   │
      │──────────────────▶│                   │
      │                   │  Confirmation     │
      │                   │  Request          │
      │                   │──────────────────▶│
      │                   │                   │
      │                   │  Confirm/Deny     │
      │                   │◀──────────────────│
      │                   │                   │
      │  ExecutionResult  │                   │
      │◀──────────────────│                   │
```

### Confirmation API

```
GET /v4/permissions/pending
Response:
{
  "pending": [
    {
      "confirmation_id": "string",
      "action": "string",
      "safety_level": "critical",
      "expires_at": "ISO8601",
      "rationale": "string"
    }
  ]
}

POST /v4/permissions/confirm/{confirmation_id}
Response:
{
  "confirmed": true,
  "execution_started": true
}

POST /v4/permissions/deny/{confirmation_id}
Response:
{
  "denied": true
}
```

---

## 6. Audit Trail Interface

Every action is logged with causality tracking.

```python
@dataclass
class AuditEntry:
    """Audit log entry"""
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
```

### Audit API

```
GET /v4/audit/recent?limit=50
Response:
{
  "entries": [AuditEntry],
  "total": int
}

GET /v4/audit/entry/{audit_id}
Response: AuditEntry

GET /v4/audit/causality/{audit_id}
Response:
{
  "chain": [AuditEntry],  # Ordered from root to leaf
  "depth": int
}
```

---

## 7. Health & Telemetry Interface

NeuroLinux reports health to NIS Hub.

```python
@dataclass
class SystemHealth:
    """System health status"""
    healthy: bool
    checks: List[HealthCheck]
    blocking_issues: List[str]

@dataclass
class HealthCheck:
    """Individual health check"""
    name: str
    value: float | bool
    threshold: Optional[float]
    passed: bool

@dataclass
class DeviceTelemetry:
    """Telemetry sent to NIS Hub"""
    device_id: str
    timestamp: datetime
    health: SystemHealth
    metrics: Dict[str, float]  # cpu, memory, disk, temp
    agents: List[AgentStatus]
    audit_count: int
    last_ota_check: Optional[datetime]
```

---

## 8. OTA Update Interface

Health-gated OTA updates.

```python
@dataclass
class OTARequest:
    """OTA update request"""
    component: str           # neurolinux-core, neurohub-ui, etc.
    version: str
    checksum: str
    size_bytes: int
    requires_reboot: bool

@dataclass
class OTAResponse:
    """OTA update response"""
    success: bool
    stage: str               # health_check, awaiting_confirmation, downloading, installing, complete
    confirmation_id: Optional[str]
    error: Optional[str]
    blocking_issues: List[str]
```

### OTA API

```
GET /v4/ota/health
Response: SystemHealth

POST /v4/ota/update?component={component}
Response: OTAResponse

POST /v4/ota/rollback/{component}
Response:
{
  "success": bool,
  "rolled_back_to": "version"
}
```

---

## 9. Error Codes

| Code | Meaning |
|------|---------|
| `NIS_001` | Invalid action plan |
| `NIS_002` | Agent not registered |
| `NIS_003` | Safety level violation |
| `NIS_004` | Confirmation required |
| `NIS_005` | Confirmation expired |
| `NIS_006` | Confirmation denied |
| `NL_001` | Hardware not available |
| `NL_002` | Permission denied |
| `NL_003` | Execution timeout |
| `NL_004` | OTA health check failed |
| `NL_005` | Rollback failed |

---

## 10. Versioning

- API version is in the URL path (`/v4/`)
- Breaking changes require new major version
- Backward compatibility maintained within major version
- Deprecation warnings provided 2 versions in advance

---

## 11. Security

- All endpoints require authentication (JWT or API key)
- `CRITICAL` actions require additional confirmation
- Audit log is append-only
- Hardware actions are rate-limited
- All communication over HTTPS in production

---

**This contract is binding for all implementations.**
