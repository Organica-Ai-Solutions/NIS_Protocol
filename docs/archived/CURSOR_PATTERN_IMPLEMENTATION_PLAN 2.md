# Cursor Pattern Implementation Plan for NIS Protocol

**Branch**: `feature/cursor-pattern-agent-orchestration`

**Goal**: Apply Cursor's agent architecture (context orchestration + action constraints + tight feedback loops) to existing NIS Protocol code.

---

## BRUTAL HONESTY: What We're Actually Doing

**NOT creating new files** - modifying existing architecture to add:
1. **Action DSL** (Tailwind-equivalent for agent actions)
2. **Context Orchestrator** (Cursor-style context builder)
3. **Agent Execution Loop** (propose → apply → validate → revise)
4. **Audit Trail** (causality tracking)

**What's REAL vs HYPE**:
- ✅ Real: Constrained action space (no free-form chaos)
- ✅ Real: Scoped context (no dumping entire state)
- ✅ Real: Validation loops (auto-repair on failure)
- ❌ Not: "AI that thinks" - it's pattern matching with constraints
- ❌ Not: "AGI" - it's good engineering

---

## Phase 1: Action DSL (Modify Existing Files)

### File: `src/core/agent_orchestrator.py`

**Current State**:
- Has `AgentStatus`, `AgentType`, `ActivationTrigger` enums
- Has `AgentDefinition` dataclass
- Has context analysis and dependency resolution

**Modifications Needed**:

1. **Add Action DSL Enum** (after line 50)
```python
class AgentAction(Enum):
    """Constrained action space - Tailwind for agent operations"""
    # Sensing
    READ_SENSOR = "read_sensor"
    QUERY_STATE = "query_state"
    
    # Memory
    QUERY_MEMORY = "query_memory"
    STORE_MEMORY = "store_memory"
    
    # Planning
    CREATE_PLAN = "create_plan"
    EXECUTE_PLAN = "execute_plan"
    
    # Tools
    RUN_TOOL = "run_tool"
    CALL_LLM = "call_llm"
    
    # Deployment
    DEPLOY_EDGE = "deploy_edge"
    UPDATE_CONFIG = "update_config"
    
    # Policy
    SET_POLICY = "set_policy"
    CHECK_POLICY = "check_policy"
```

2. **Add Action Definition Dataclass** (after AgentMetrics)
```python
@dataclass
class ActionDefinition:
    """Defines a constrained action with validation rules"""
    action_type: AgentAction
    agent_id: str
    parameters: Dict[str, Any]
    requires_approval: bool = False
    rollback_enabled: bool = True
    timeout_seconds: float = 30.0
    audit_id: Optional[str] = None
```

3. **Add Action Result Dataclass**
```python
@dataclass
class ActionResult:
    """Result of an action execution"""
    action_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    audit_trail: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.audit_trail is None:
            self.audit_trail = []
```

---

## Phase 2: Context Orchestrator (Modify Existing Files)

### File: `src/core/agent_orchestrator.py`

**Current State**:
- Has `ContextAnalyzer` class (line 118)
- Has `_determine_required_agents` method (line 446)

**Modifications Needed**:

1. **Enhance ContextAnalyzer** (find existing class, add methods)
```python
class ContextAnalyzer:
    """Cursor-style context builder - scoped and targeted"""
    
    def build_context_pack(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Build just-in-time context pack (Cursor's secret sauce)
        
        Returns ONLY what this agent needs:
        - Relevant state
        - Relevant memory
        - Relevant tools
        - NO noise
        """
        context_pack = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "request": request_data,
            "state": self._get_relevant_state(agent_id),
            "memory": self._get_relevant_memory(agent_id, request_data),
            "tools": self._get_allowed_tools(agent_id),
            "policies": self._get_active_policies(agent_id),
            "token_budget": max_tokens
        }
        
        return context_pack
    
    def _get_relevant_state(self, agent_id: str) -> Dict[str, Any]:
        """Get only state relevant to this agent"""
        # Implementation: filter state by agent scope
        pass
    
    def _get_relevant_memory(self, agent_id: str, request: Dict) -> List[Dict]:
        """Get only memories relevant to this request"""
        # Implementation: semantic search with limit
        pass
    
    def _get_allowed_tools(self, agent_id: str) -> List[str]:
        """Get tools this agent is allowed to use"""
        # Implementation: policy-based tool access
        pass
    
    def _get_active_policies(self, agent_id: str) -> List[Dict]:
        """Get policies that apply to this agent"""
        # Implementation: policy engine query
        pass
```

---

## Phase 3: Agent Execution Loop (Modify Existing Files)

### File: `src/core/agent_orchestrator.py`

**Current State**:
- Has `process_request` method (line 414)
- Has `_process_through_pipeline` method (not shown, but exists)

**Modifications Needed**:

1. **Add Agent Execution Loop** (new method in NISAgentOrchestrator)
```python
async def execute_agent_action(
    self,
    action: ActionDefinition,
    context_pack: Dict[str, Any]
) -> ActionResult:
    """
    Cursor-style execution loop: propose → apply → validate → revise
    
    This is the CORE of Cursor's reliability.
    """
    action_id = f"action_{int(time.time())}_{action.agent_id}"
    start_time = time.time()
    audit_trail = []
    
    try:
        # 1. PROPOSE (build action plan)
        audit_trail.append({
            "step": "propose",
            "timestamp": time.time(),
            "action": action.action_type.value,
            "parameters": action.parameters
        })
        
        # 2. VALIDATE (check constraints BEFORE execution)
        validation = await self._validate_action(action, context_pack)
        if not validation["valid"]:
            return ActionResult(
                action_id=action_id,
                success=False,
                output=None,
                error=f"Validation failed: {validation['reason']}",
                execution_time=time.time() - start_time,
                audit_trail=audit_trail
            )
        
        audit_trail.append({
            "step": "validate",
            "timestamp": time.time(),
            "result": "passed"
        })
        
        # 3. APPLY (execute the action)
        result = await self._apply_action(action, context_pack)
        
        audit_trail.append({
            "step": "apply",
            "timestamp": time.time(),
            "result": result
        })
        
        # 4. VERIFY (check post-conditions)
        verification = await self._verify_result(action, result, context_pack)
        
        if not verification["valid"]:
            # 5. REVISE (auto-repair if possible)
            if action.rollback_enabled:
                await self._rollback_action(action, result)
                audit_trail.append({
                    "step": "rollback",
                    "timestamp": time.time(),
                    "reason": verification["reason"]
                })
            
            return ActionResult(
                action_id=action_id,
                success=False,
                output=result,
                error=f"Verification failed: {verification['reason']}",
                execution_time=time.time() - start_time,
                audit_trail=audit_trail
            )
        
        audit_trail.append({
            "step": "verify",
            "timestamp": time.time(),
            "result": "passed"
        })
        
        # SUCCESS
        return ActionResult(
            action_id=action_id,
            success=True,
            output=result,
            execution_time=time.time() - start_time,
            audit_trail=audit_trail
        )
        
    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        
        # Auto-rollback on exception
        if action.rollback_enabled:
            try:
                await self._rollback_action(action, None)
            except:
                pass
        
        return ActionResult(
            action_id=action_id,
            success=False,
            output=None,
            error=str(e),
            execution_time=time.time() - start_time,
            audit_trail=audit_trail
        )

async def _validate_action(
    self,
    action: ActionDefinition,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate action BEFORE execution"""
    # Check policies
    # Check permissions
    # Check resource availability
    # Check dependencies
    return {"valid": True}

async def _apply_action(
    self,
    action: ActionDefinition,
    context: Dict[str, Any]
) -> Any:
    """Execute the actual action"""
    # Route to appropriate handler based on action.action_type
    pass

async def _verify_result(
    self,
    action: ActionDefinition,
    result: Any,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify result meets post-conditions"""
    # Check invariants
    # Check state consistency
    # Check output validity
    return {"valid": True}

async def _rollback_action(
    self,
    action: ActionDefinition,
    result: Any
) -> None:
    """Rollback action if it failed"""
    # Restore previous state
    # Undo side effects
    pass
```

---

## Phase 4: LLM Provider Integration (Modify Existing Files)

### File: `src/llm/llm_manager.py`

**Current State**:
- Has `GeneralLLMProvider` class (line 63)
- Has `generate_response` method (exists but not shown)

**Modifications Needed**:

1. **Add Context-Aware LLM Call** (new method in GeneralLLMProvider)
```python
async def generate_with_context_pack(
    self,
    context_pack: Dict[str, Any],
    user_message: str,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cursor-style LLM call with scoped context
    
    Instead of dumping everything, we send ONLY:
    - Relevant state
    - Relevant memory
    - Allowed tools
    - Current policies
    """
    
    # Build system prompt from context pack
    system_prompt = self._build_system_prompt(context_pack)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Add relevant memory as context
    if context_pack.get("memory"):
        memory_context = self._format_memory_context(context_pack["memory"])
        messages.insert(1, {"role": "system", "content": memory_context})
    
    # Call LLM
    response = await self.generate_response(
        messages=messages,
        provider=provider,
        model=model,
        max_tokens=context_pack.get("token_budget", 4000)
    )
    
    return {
        "response": response,
        "context_used": {
            "state_keys": list(context_pack.get("state", {}).keys()),
            "memory_count": len(context_pack.get("memory", [])),
            "tools_available": context_pack.get("tools", []),
            "policies_active": len(context_pack.get("policies", []))
        },
        "tokens_used": self._estimate_tokens(messages, response)
    }

def _build_system_prompt(self, context_pack: Dict[str, Any]) -> str:
    """Build focused system prompt from context pack"""
    agent_id = context_pack["agent_id"]
    state = context_pack.get("state", {})
    tools = context_pack.get("tools", [])
    policies = context_pack.get("policies", [])
    
    prompt = f"""You are agent: {agent_id}

Current State:
{json.dumps(state, indent=2)}

Available Tools:
{', '.join(tools)}

Active Policies:
{self._format_policies(policies)}

Respond concisely and follow all policies."""
    
    return prompt

def _format_memory_context(self, memories: List[Dict]) -> str:
    """Format relevant memories"""
    if not memories:
        return ""
    
    context = "Relevant Context:\n"
    for mem in memories[:5]:  # Limit to top 5
        context += f"- {mem.get('content', '')}\n"
    
    return context

def _format_policies(self, policies: List[Dict]) -> str:
    """Format active policies"""
    if not policies:
        return "No special policies"
    
    return "\n".join([f"- {p.get('rule', '')}" for p in policies])

def _estimate_tokens(self, messages: List[Dict], response: str) -> int:
    """Rough token estimation"""
    total_text = " ".join([m.get("content", "") for m in messages]) + response
    return len(total_text) // 4  # Rough estimate
```

---

## Phase 5: Audit Trail (New File - Minimal)

### File: `src/core/audit_trail.py` (NEW - but minimal)

```python
#!/usr/bin/env python3
"""
Audit Trail for NIS Protocol Agent Actions
Cursor-style causality tracking
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AuditEntry:
    """Single audit entry"""
    audit_id: str
    timestamp: float
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AuditTrail:
    """
    Audit trail manager - every action is logged
    
    This is Cursor's "git diff" equivalent.
    """
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[AuditEntry] = []
    
    def log_action(
        self,
        agent_id: str,
        action_type: str,
        parameters: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
        execution_time: float = 0.0
    ) -> str:
        """Log an action"""
        audit_id = f"audit_{int(time.time())}_{agent_id}"
        
        entry = AuditEntry(
            audit_id=audit_id,
            timestamp=time.time(),
            agent_id=agent_id,
            action_type=action_type,
            parameters=parameters,
            result=result,
            success=success,
            error=error,
            execution_time=execution_time
        )
        
        self.entries.append(entry)
        self._write_to_disk(entry)
        
        return audit_id
    
    def _write_to_disk(self, entry: AuditEntry):
        """Write audit entry to disk"""
        log_file = self.log_dir / f"{entry.audit_id}.json"
        with open(log_file, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def get_agent_history(self, agent_id: str, limit: int = 100) -> List[AuditEntry]:
        """Get audit history for an agent"""
        return [e for e in self.entries if e.agent_id == agent_id][-limit:]
    
    def get_failed_actions(self, limit: int = 100) -> List[AuditEntry]:
        """Get failed actions"""
        return [e for e in self.entries if not e.success][-limit:]
```

---

## Implementation Order

### Week 1: Core Infrastructure
1. ✅ Create branch
2. Add Action DSL to `agent_orchestrator.py`
3. Add ActionDefinition and ActionResult dataclasses
4. Test: Verify enums and dataclasses work

### Week 2: Context Orchestration
1. Enhance ContextAnalyzer in `agent_orchestrator.py`
2. Implement `build_context_pack` method
3. Implement helper methods (_get_relevant_state, etc.)
4. Test: Build context packs for different agents

### Week 3: Execution Loop
1. Add `execute_agent_action` method to orchestrator
2. Implement validation, apply, verify, rollback methods
3. Test: Execute actions with validation and rollback

### Week 4: LLM Integration
1. Add `generate_with_context_pack` to llm_manager.py
2. Implement context-aware prompt building
3. Test: LLM calls with scoped context

### Week 5: Audit Trail
1. Create `audit_trail.py`
2. Integrate with execution loop
3. Test: Verify all actions are logged

### Week 6: Integration & Testing
1. Wire everything together in main.py
2. Update endpoints to use new execution loop
3. Test end-to-end flows
4. Performance testing

---

## Success Metrics (BRUTAL HONESTY)

**What SUCCESS looks like**:
- ✅ Actions are constrained (can't do random stuff)
- ✅ Context is scoped (no 100k token dumps)
- ✅ Failures auto-rollback (no broken state)
- ✅ Every action is auditable (full causality)
- ✅ LLM calls are focused (better responses)

**What SUCCESS is NOT**:
- ❌ Not "AGI" or "self-aware"
- ❌ Not "thinking" - it's pattern matching
- ❌ Not "learning" - it's execution discipline
- ❌ Not magic - it's good engineering

**Honest assessment**: This is ~60% of Cursor's power. The other 40% is:
- Editor integration (we don't have)
- AST parsing (we don't have)
- Git integration (we have partial)

But 60% is HUGE for agent reliability.

---

## Files to Modify (Summary)

1. **`src/core/agent_orchestrator.py`** - Main changes
   - Add Action DSL enums
   - Add ActionDefinition/ActionResult dataclasses
   - Enhance ContextAnalyzer
   - Add execute_agent_action method
   - Add validation/verify/rollback methods

2. **`src/llm/llm_manager.py`** - LLM integration
   - Add generate_with_context_pack method
   - Add context-aware prompt building

3. **`src/core/audit_trail.py`** - NEW (minimal)
   - AuditEntry dataclass
   - AuditTrail class

4. **`main.py`** - Integration
   - Wire new execution loop
   - Update endpoints to use constrained actions

---

## Testing Plan

### Unit Tests
- Action DSL validation
- Context pack building
- Execution loop (propose/apply/validate/revise)
- Rollback functionality
- Audit logging

### Integration Tests
- Full agent execution flow
- Multi-agent coordination with constraints
- LLM calls with scoped context
- Failure scenarios and rollback

### Performance Tests
- Context pack size (should be <10% of full state)
- Execution time (should be <2x current)
- Memory usage (should be similar)

---

## Next Steps

1. Review this plan
2. Start with Phase 1 (Action DSL)
3. Implement incrementally
4. Test each phase before moving to next
5. Merge to main when stable

**Estimated time**: 4-6 weeks for full implementation
**Risk level**: Medium (modifying core orchestrator)
**Rollback plan**: Branch-based, can revert anytime
