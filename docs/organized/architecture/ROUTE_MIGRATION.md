# NIS Protocol v4.0 - Route Migration Documentation

**Version:** 1.0  
**Date:** 2025-11-30  
**Status:** In Progress (50% Complete)

---

## Overview

This document tracks the migration of API endpoints from the monolithic `main.py` (11,516 lines) to modular route files in the `routes/` directory for improved maintainability, testability, and code organization.

### Migration Goals

1. **Modularity**: Split endpoints by domain (robotics, physics, voice, etc.)
2. **Maintainability**: Smaller, focused files are easier to maintain
3. **Testability**: Each route module can be tested independently
4. **Dependency Injection**: Clean separation of concerns with injectable dependencies
5. **Bug Fixes**: Fix issues discovered during migration (missing decorators, etc.)

---

## Migration Status

### Summary

| Metric | Value |
|--------|-------|
| **Total Endpoints** | 222 |
| **Endpoints in Route Modules** | 222 (100%) |
| **Route Modules Created** | 24 |
| **main.py Lines** | 1,196 (was 12,291) |
| **Code Reduction** | 90% |
| **Bugs Found & Fixed** | 2 |

### Route Modules

| Module | Endpoints | Size | Status | Description |
|--------|-----------|------|--------|-------------|
| `robotics.py` | 5 | 17KB | ✅ Ready | FK/IK, trajectory planning, telemetry |
| `physics.py` | 6 | 17KB | ✅ Ready | PINN validation, equations, constants |
| `bitnet.py` | 6 | 17KB | ✅ Ready | BitNet training, status, export |
| `webhooks.py` | 3 | 5KB | ✅ Ready | Webhook registration, management |
| `monitoring.py` | 16 | 14KB | ✅ Ready | Health, metrics, analytics, system |
| `memory.py` | 15 | 16KB | ✅ Ready | Conversations, topics, persistence |
| `chat.py` | 9 | 12KB | ✅ Ready | Simple chat, streaming, reflective |
| `agents.py` | 11 | 13KB | ✅ Ready | Learning, planning, simulation, ethics |
| `research.py` | 4 | 11KB | ✅ Ready | Deep research, claim validation |
| `voice.py` | 7 | 33KB | ✅ Ready | STT, TTS, voice chat, settings |
| `protocols.py` | 22 | 34KB | ✅ Ready | MCP, A2A, ACP integrations |
| `vision.py` | 12 | 32KB | ✅ Ready | Image analysis, generation, visualization |
| `reasoning.py` | 3 | 8KB | ✅ Ready | Collaborative reasoning, debate |
| `consciousness.py` | 28 | 35KB | ✅ Ready | V4.0 evolution, genesis, collective, embodiment |
| `system.py` | 18 | 25KB | ✅ Ready | Configuration, state, edge AI, brain orchestration |
| `nvidia.py` | 8 | 15KB | ✅ Ready | NVIDIA Inception, NeMo, enterprise features |
| `auth.py` | 11 | 10KB | ✅ Ready | Authentication, user management, API keys |
| `utilities.py` | 14 | 15KB | ✅ Ready | Cost tracking, cache, templates, code execution |
| `v4_features.py` | 11 | 8KB | ✅ Ready | V4.0 memory, self-modification, goals |
| `llm.py` | 7 | 12KB | ✅ Ready | LLM optimization, consensus, analytics |
| `unified.py` | 4 | 10KB | ✅ Ready | Unified pipeline, autonomous mode, integration |
| `core.py` | 2 | 6KB | ✅ Ready | Root endpoint, health check |

---

## Migration Complete ✅

The migration is now **100% complete**:
- All 222 endpoints are in modular route files
- `main.py` reduced from 12,291 lines to 1,196 lines (90% reduction)
- Frontend UI endpoints removed (frontend is in separate repository)
- All route modules tested and compiling

---

## Bugs Found in main.py

During migration, the following bugs were discovered:

### Bug #1: Missing Decorator on `conduct_deep_research`

**Location:** `main.py` line 8645  
**Issue:** Function is missing `@app.post("/research/deep", tags=["Research"])` decorator  
**Impact:** Endpoint is unreachable - function exists but cannot be called via API  
**Status:** ✅ Fixed in `routes/research.py`

```python
# BROKEN in main.py (line 8645):
async def conduct_deep_research(request: ResearchRequest):
    # Missing @app.post decorator!

# FIXED in routes/research.py:
@router.post("/deep")
async def deep_research(request: DeepResearchRequest):
```

### Bug #2: Missing Decorator on `get_multimodal_status`

**Location:** `main.py` line 9644  
**Issue:** Function is missing `@app.get("/agents/multimodal/status", tags=["Multimodal"])` decorator  
**Impact:** Endpoint is unreachable  
**Status:** ✅ Fixed in `routes/reasoning.py`

```python
# BROKEN in main.py (line 9644):
async def get_multimodal_status():
    # Missing @app.get decorator!

# FIXED in routes/reasoning.py:
@router.get("/agents/multimodal/status")
async def get_multimodal_status():
```

---

## Architecture

### Dependency Injection Pattern

Each route module uses a dependency injection pattern for accessing shared resources:

```python
# In route module (e.g., routes/chat.py)
def set_dependencies(llm_provider=None, reflective_generator=None):
    """Set dependencies for the chat router"""
    router._llm_provider = llm_provider
    router._reflective_generator = reflective_generator

def get_llm_provider():
    return getattr(router, '_llm_provider', None)
```

### Integration Pattern

When integrating routes into main.py:

```python
# 1. Import routers
from routes import (
    robotics_router, physics_router, bitnet_router, webhooks_router,
    monitoring_router, memory_router, chat_router, agents_router,
    research_router, voice_router, protocols_router, vision_router,
    reasoning_router, set_monitoring_dependencies, set_chat_dependencies,
    # ... other setters
)

# 2. Include routers
app.include_router(robotics_router)
app.include_router(physics_router)
# ... etc

# 3. Inject dependencies (in initialize_system())
set_monitoring_dependencies(
    llm_provider=llm_provider,
    conversation_memory=conversation_memory,
    agent_registry=agent_registry,
    tool_registry=tool_registry
)
```

---

## File Structure

```
routes/
├── __init__.py          # Exports all routers and dependency setters
├── agents.py            # Agent management endpoints
├── bitnet.py            # BitNet training endpoints
├── chat.py              # Chat endpoints
├── memory.py            # Memory management endpoints
├── monitoring.py        # Health/metrics endpoints
├── physics.py           # Physics validation endpoints
├── protocols.py         # MCP/A2A/ACP protocol endpoints
├── reasoning.py         # Reasoning endpoints
├── research.py          # Research endpoints
├── robotics.py          # Robotics endpoints
├── vision.py            # Vision/image endpoints
├── voice.py             # Voice/audio endpoints
└── webhooks.py          # Webhook endpoints
```

---

## Endpoint Reference

### Robotics (`/robotics`)
- `POST /robotics/forward_kinematics` - Compute FK
- `POST /robotics/inverse_kinematics` - Compute IK
- `POST /robotics/plan_trajectory` - Plan trajectory
- `GET /robotics/capabilities` - Get capabilities
- `GET /robotics/telemetry/{robot_id}` - Get telemetry

### Physics (`/physics`)
- `POST /physics/validate/true-pinn` - PINN validation
- `POST /physics/solve/heat-equation` - Solve heat equation
- `POST /physics/solve/wave-equation` - Solve wave equation
- `GET /physics/capabilities` - Get capabilities
- `GET /physics/constants` - Get physics constants
- `POST /physics/validate` - General validation

### Voice
- `POST /communication/consciousness_voice` - Vocalize consciousness
- `GET /communication/status` - Communication status
- `POST /voice/transcribe` - Speech-to-text
- `GET /voice/settings` - Get voice settings
- `POST /voice/settings/update` - Update settings
- `GET /voice/test-speaker/{speaker}` - Test speaker
- `WS /ws/voice-chat` - Real-time voice chat

### Vision
- `POST /vision/analyze` - Analyze image
- `POST /vision/analyze/simple` - Simple analysis
- `POST /vision/generate` - Generate visual
- `POST /image/generate` - Generate image (DALL-E)
- `POST /image/edit` - Edit image
- `POST /visualization/create` - Create visualization
- `POST /visualization/chart` - Generate chart
- `POST /visualization/diagram` - Generate diagram
- `POST /visualization/auto` - Auto-detect visualization
- `POST /visualization/interactive` - Interactive chart
- `POST /visualization/dynamic` - Dynamic chart
- `POST /document/analyze` - Analyze document

### Protocols
- `GET /protocol/mcp/tools` - MCP tools
- `POST /protocol/mcp/initialize` - Initialize MCP
- `POST /protocol/a2a/create-task` - Create A2A task
- `GET /protocol/a2a/task/{task_id}` - Get task status
- `DELETE /protocol/a2a/task/{task_id}` - Cancel task
- `GET /protocol/acp/agent-card` - Get agent card
- `POST /protocol/acp/execute` - Execute ACP agent
- `GET /agents` - List ACP agents
- `POST /runs` - Create ACP run
- `GET /runs/{run_id}` - Get run status
- `GET /sessions/{session_id}` - Get session
- `GET /protocol/health` - Protocol health
- `POST /protocol/translate` - Translate message
- `POST /api/mcp/tools` - Execute MCP tool
- `POST /api/mcp/ui-action` - Handle UI action
- `POST /api/mcp/plans` - Create plan
- `POST /api/mcp/plans/{plan_id}/execute` - Execute plan
- `GET /api/mcp/plans/{plan_id}/status` - Plan status
- `GET /api/mcp/info` - MCP info
- `POST /api/mcp/invoke` - LangGraph invoke
- `POST /mcp/chat` - MCP chat
- `GET /tools/list` - List tools

### Consciousness (`/v4`)
- `POST /v4/consciousness/evolve` - Trigger self-evolution
- `GET /v4/consciousness/evolution/history` - Evolution history
- `GET /v4/consciousness/performance` - Performance trends
- `POST /v4/consciousness/genesis` - Create dynamic agent
- `GET /v4/consciousness/genesis/history` - Genesis history
- `POST /v4/consciousness/collective/register` - Register peer
- `POST /v4/consciousness/collective/decide` - Collective decision
- `POST /v4/consciousness/collective/sync` - Sync state
- `GET /v4/consciousness/collective/status` - Collective status
- `POST /v4/consciousness/plan` - Create autonomous plan
- `GET /v4/consciousness/plan/status` - Plan status
- `POST /v4/consciousness/marketplace/publish` - Publish insight
- `GET /v4/consciousness/marketplace/list` - List insights
- `GET /v4/consciousness/marketplace/insight/{id}` - Get insight
- `POST /v4/consciousness/multipath/start` - Start multipath reasoning
- `POST /v4/consciousness/multipath/collapse` - Collapse to single path
- `GET /v4/consciousness/multipath/state` - Get multipath state
- `POST /v4/consciousness/ethics/evaluate` - Ethical evaluation
- `POST /v4/consciousness/embodiment/state/update` - Update body state
- `POST /v4/consciousness/embodiment/motion/check` - Check motion safety
- `POST /v4/consciousness/embodiment/action/execute` - Execute action
- `GET /v4/consciousness/embodiment/status` - Embodiment status
- `GET /v4/consciousness/embodiment/redundancy/status` - Redundancy status
- `POST /v4/consciousness/embodiment/diagnostics` - Self-diagnostics
- `GET /v4/consciousness/embodiment/redundancy/degradation` - Degradation mode
- `GET /v4/dashboard/complete` - Complete system dashboard

---

## Integration Status

### Router Integration (Completed)

All 23 modular routers have been integrated into `main.py`:

```python
# main.py (lines 443-486)
from routes import (
    robotics_router, physics_router, bitnet_router, webhooks_router,
    monitoring_router, memory_router, chat_router, agents_router,
    research_router, voice_router, protocols_router, vision_router,
    reasoning_router, consciousness_router, system_router, nvidia_router,
    auth_router, utilities_router, v4_features_router, llm_router, unified_router,
    # ... dependency setters
)

app.include_router(robotics_router)
# ... all 23 routers included
```

### Dependency Injection (Completed)

Dependencies are injected in `initialize_system()` (lines 1016-1126):

```python
# Example dependency injection
set_chat_dependencies(
    llm_provider=llm_provider,
    reflective_generator=reflective_generator,
    bitnet_trainer=bitnet_trainer,
    consciousness_service=consciousness_service
)
```

## Known Issues

- Remaining ~16 endpoints still in main.py (core system endpoints like `/`, `/health`, `/console`)
- These endpoints are deeply integrated with main.py's global state
- Original migrated endpoints still exist in main.py (will be removed after testing)

---

## Backup Information

| File | Location | Size | Hash |
|------|----------|------|------|
| Original main.py | `backups/main_backup_20251130_213230.py` | 484KB | See stats.json |
| Refactor backup | `backups/refactor_20251130_213456/` | - | SHA256 verified |

---

## Changelog

### 2025-11-30
- Created 15 route modules
- Migrated 119 endpoints (50%)
- Found and fixed 2 bugs (missing decorators)
- Created comprehensive documentation
