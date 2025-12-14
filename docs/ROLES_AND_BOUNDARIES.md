# NIS / NeuroLinux / NIS Hub – Roles & Boundaries

**Version:** 1.0  
**Date:** December 13, 2025  
**Status:** Internal Architecture Document

---

## Executive Summary

This document defines the clear separation of responsibilities between the three core components of the Organica AI ecosystem:

| Layer | License | Purpose |
|-------|---------|---------|
| **NIS Protocol** | Open Source (Apache 2.0) | Cognitive orchestration substrate |
| **NeuroLinux** | Closed Source | Edge execution environment |
| **NIS Hub** | Closed Source | Cloud coordination & control plane |

**Golden Rule:**
> Open code defines "what is allowed."  
> Closed code decides "when and how it happens."

---

## 1. NIS Protocol (Open Source)

### Role
The **scientific and architectural substrate** for AI cognition. Think of it as the "TCP/IP of cognition" — boring, inspectable, and stable.

### Owns
- Cognitive orchestration logic
- Agent interaction contracts
- Intent → action routing semantics
- Safety level specifications
- Explainability model (causality graphs, audit schemas)
- Reference agents (non-hardware-specific)
- Protocol interfaces (MCP-like boundaries)
- 10-phase consciousness pipeline specification

### Key Components
```
src/core/
├── unified_pipeline.py      # Pipeline modes and context
├── agent_orchestrator.py    # Brain-like agent coordination
├── state_manager.py         # State event management
├── provider_router.py       # LLM provider abstraction
└── symbolic_bridge.py       # Symbolic reasoning

src/agents/
├── robotics/                # Reference robotics agent
├── research/                # Reference research agent
├── consciousness/           # Consciousness pipeline
└── physics/                 # PINN validation
```

### Does NOT Own
- ❌ Privileged system execution
- ❌ Hardware drivers
- ❌ OTA implementation
- ❌ Fleet management
- ❌ Billing/quotas

### Why Open
- Credibility with academic/engineering community
- External validation and trust
- Long-term survivability
- Avoids "black-box AGI" accusations
- Enables third-party agent development

---

## 2. NeuroLinux (Closed Source)

### Role
The **execution environment** for NIS Protocol on edge devices. This is where cognition meets hardware.

### Owns
- Privileged system control (root operations)
- Hardware drivers and glue code
- OTA update implementation details
- Security hardening
- Operator UX (terminal + dashboard)
- Safety enforcement (not just spec)
- Platform-specific optimizations
- Edge-specific reliability hacks
- Audit logging persistence
- Permission confirmation flow

### Key Components
```
neurolinux-os/
├── buildroot/board/neurolinux/overlay/opt/neurolinux/
│   └── neurolinux_core.py   # Core API (FastAPI)
├── buildroot/package/
│   ├── neurokernel/         # Rust agent manager
│   ├── nis-protocol/        # NIS Protocol integration
│   └── neurohub-ui/         # Web dashboard

phase4-distributed/
├── neurohub-ui/             # React dashboard
└── neurogrid/               # P2P mesh networking
```

### Does NOT Own
- ❌ Cognitive orchestration logic (uses NIS Protocol)
- ❌ Agent decision-making algorithms
- ❌ LLM provider selection
- ❌ Fleet-wide policies

### Why Closed
- Runs with root privileges
- Touches hardware directly
- Encodes operational risk
- Where liability lives
- Competitive moat

---

## 3. NIS Hub (Closed Source)

### Role
The **control plane and business lever** for multi-node coordination. Think of it as "Kubernetes for cognition."

### Owns
- Multi-node orchestration
- Federation logic
- Fleet management
- Policy distribution
- Identity, auth, tenancy
- Analytics and telemetry aggregation
- Enterprise/institutional features
- Billing, quotas, governance
- Long-term audit storage
- OTA distribution

### Key Components
```
core/
├── main.py                  # FastAPI backend
├── services/
│   ├── neurolinux_service.py    # NeuroLinux integration
│   ├── protocol_bridge_service.py # Protocol bridging
│   ├── consciousness_service.py  # Consciousness coordination
│   └── security_service.py       # Auth & security

ui/                          # React dashboard
sdk/                         # Client SDKs
```

### Does NOT Own
- ❌ Direct hardware control
- ❌ Local safety enforcement
- ❌ Edge-specific execution

### Why Closed
- Where value concentrates
- Where customers live
- Where misuse must be controlled
- Differentiation at scale

---

## 4. Interaction Rules

### Flow of Control (Golden Path)

```
┌─────────────────────────────────────────────────────────────────┐
│                        OPERATOR INTENT                          │
│                    (human or upstream system)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NEUROLINUX CORE                            │
│                   (receives intent, logs it)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       NIS PROTOCOL                              │
│  • Classifies intent (SafetyLevel)                              │
│  • Determines agent(s)                                          │
│  • Produces action plan (NOT execution)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NEUROLINUX CORE                            │
│  • Enforces safety + permissions                                │
│  • Executes actions on hardware/system                          │
│  • Collects results + telemetry                                 │
│  • Logs to audit trail                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         NIS HUB                                 │
│              (optional: receives telemetry,                     │
│               aggregates, distributes policies)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Non-Negotiable Rules

| Rule | Rationale |
|------|-----------|
| NIS Protocol **never** executes privileged actions | Separation of cognition from execution |
| NeuroLinux **never** invents cognition | Uses NIS Protocol for all decisions |
| NIS Hub **never** touches hardware directly | Cloud is advisory, edge is sovereign |
| Edge must remain **sovereign** | Cloud failure cannot brick devices |

---

## 5. API Boundaries

### NIS Protocol → NeuroLinux

```python
# NIS Protocol provides:
class ActionPlan:
    intent: str
    safety_level: SafetyLevel  # safe, privileged, hardware, critical
    agents: List[str]
    actions: List[Action]
    rationale: str
    requires_confirmation: bool

# NeuroLinux implements:
def execute_plan(plan: ActionPlan) -> ExecutionResult:
    # 1. Check permissions
    # 2. Request confirmation if needed
    # 3. Execute actions
    # 4. Log to audit
    # 5. Return results
```

### NeuroLinux → NIS Hub

```python
# NeuroLinux reports:
class DeviceHeartbeat:
    device_id: str
    status: DeviceStatus
    metrics: SystemMetrics
    agents: List[AgentStatus]
    audit_summary: AuditSummary

# NIS Hub provides:
class PolicyUpdate:
    safety_thresholds: Dict[str, float]
    allowed_actions: List[str]
    ota_available: Optional[OTAInfo]
```

---

## 6. Failure Modes

| Component | Failure | Impact | Recovery |
|-----------|---------|--------|----------|
| NIS Protocol | Crash | No new cognition | NeuroLinux continues with cached policies |
| NeuroLinux | Crash | Device offline | Watchdog restarts, rollback if OTA failed |
| NIS Hub | Unreachable | No fleet coordination | Devices operate autonomously |
| Network | Down | No cloud sync | Full local operation continues |

**Key Principle:** Edge devices must be fully functional without cloud connectivity.

---

## 7. Open vs Closed Interface Contract

Everything closed **must have** a documented open interface:

| Closed Component | Open Interface |
|------------------|----------------|
| NeuroLinux Core | `/v4/*` REST API (documented in API.md) |
| NeuroLinux OTA | OTA spec in SAFETY.md |
| NIS Hub | `/api/v1/*` REST API |
| NIS Hub Federation | Federation protocol spec |

Everything open **can be replaced** without breaking products:

| Open Component | Replacement Path |
|----------------|------------------|
| NIS Protocol agents | Custom agents via NeuroForge SDK |
| NIS Protocol pipeline | Custom pipeline implementing same interface |
| NIS Protocol LLM providers | Any provider implementing ProviderInterface |

---

## 8. Governance

### NIS Protocol (Open)
- Versioned specs with semantic versioning
- Changelog with semantic meaning
- Backward compatibility guarantees
- Public roadmap
- Clear "experimental vs stable" markers
- Apache 2.0 license

### NeuroLinux (Closed)
- Internal roadmap
- Security-first changes
- Explicit hardware support matrix
- Release notes focused on operators
- Proprietary license

### NIS Hub (Closed)
- Customer-driven roadmap
- SLA thinking
- Compliance posture
- Kill-switch and incident response plans
- Enterprise license

---

## 9. AWS / Cloudelligent Integration

AWS is **infrastructure substrate**, not "the brain."

### AWS Is For
- ✅ Running NIS Hub
- ✅ Hosting shared services (auth, logs, analytics)
- ✅ Optional cloud cognition augmentation
- ✅ CI/CD, OTA distribution
- ✅ Long-term storage of audit logs
- ✅ Fleet coordination

### AWS Is NOT For
- ❌ Direct control of edge hardware
- ❌ Replacing local safety checks
- ❌ Becoming a single point of failure

---

## 10. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   NIS PROTOCOL (Open)                                           │
│   "What is allowed"                                             │
│   • Cognition specs                                             │
│   • Agent contracts                                             │
│   • Safety levels                                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   NEUROLINUX (Closed)                                           │
│   "When and how it happens"                                     │
│   • Execution                                                   │
│   • Hardware                                                    │
│   • Safety enforcement                                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   NIS HUB (Closed)                                              │
│   "Who and at what scale"                                       │
│   • Fleet management                                            │
│   • Policies                                                    │
│   • Business logic                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**This document is the source of truth for architectural decisions.**

Any feature that violates these boundaries must be explicitly approved and documented.
