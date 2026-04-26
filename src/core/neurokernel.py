"""
NIS Protocol — NeuroKernel v2
==============================
The central intelligence kernel of the NIS Protocol.

v1 was the NISAgentOrchestrator — brain-like agent coordination.
v2 adds the five new components learned from OpenFang's architecture,
integrated natively (not as a dependency):

  SkillLoader          → Knowledge layer (SKILL.md injection)
  AuditChain           → Memory/Consciousness (Merkle tamper-proof log)
  LoopGuard            → Attention (SHA256 circuit breaker)
  DriveScheduler       → Autonomous Action (scheduled drives/hands)
  PromptInjectionScanner → Security (16-layer protection)

The NeuroKernel is the single entry point for ALL NIS Protocol processing:

  user_input → [Scanner] → [SkillLoader inject] → [LLM / Tool] 
             → [LoopGuard check] → [Execute] → [AuditChain log]
             → response

DIKW flow (Nested Intelligence Stack):
  Data        → raw input + sensor data
  Information → scanned, skill-enriched context
  Knowledge   → LLM reasoning + tool results
  Wisdom      → AuditChain pattern + Drive adaptive scheduling

Architecture:
  - Fully async
  - Zero new pip dependencies (all stdlib + already installed)
  - Drop-in integration with existing NISAgent and NISAgentOrchestrator
  - Exposes /neurokernel/* routes for dashboard visibility
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("nis.neurokernel")

# ── Lazy imports (no circular deps) ───────────────────────────────────────────

def _skill_loader():
    from .skill_loader import get_skill_loader
    return get_skill_loader()

def _audit_chain():
    from .audit_chain import get_audit_chain
    return get_audit_chain()

def _loop_guard():
    from .loop_guard import get_loop_guard
    return get_loop_guard()

def _scanner():
    from .prompt_injection_scanner import get_scanner
    return get_scanner()

def _drive_scheduler():
    from .drive_scheduler import get_drive_scheduler
    return get_drive_scheduler()


# ── Processing result ─────────────────────────────────────────────────────────

@dataclass
class KernelResult:
    """Standard result from the NeuroKernel processing pipeline."""
    request_id: str
    success: bool
    response: str
    audit_entry_id: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    processing_ms: float = 0.0
    layer_trace: List[str] = field(default_factory=list)


# ── NeuroKernel ───────────────────────────────────────────────────────────────

class NeuroKernel:
    """
    NIS Protocol v2 central intelligence kernel.

    Usage:
        kernel = NeuroKernel()
        await kernel.startup()

        result = await kernel.process(
            agent_id="cosmos-reasoner",
            layer="reasoning",
            action_type="cosmos_reason",
            user_input="pick up the red block",
            handler=my_llm_call,
        )
    """

    def __init__(self, auto_start_drives: bool = True):
        self._auto_start_drives = auto_start_drives
        self._started = False
        self._request_count = 0
        self._start_time = time.time()

        # Components (lazy-initialized)
        self._skill_loader = None
        self._audit_chain = None
        self._loop_guard = None
        self._scanner = None
        self._drive_scheduler = None

    # ── Startup ───────────────────────────────────────────────────────────────

    async def startup(self):
        """Initialize all kernel components. Call once at app startup."""
        if self._started:
            return
        logger.info("NeuroKernel v2 starting up...")

        # Initialize components
        self._skill_loader  = _skill_loader()
        self._audit_chain   = _audit_chain()
        self._loop_guard    = _loop_guard()
        self._scanner       = _scanner()
        self._drive_scheduler = _drive_scheduler()

        # Load skills
        n_skills = self._skill_loader.load_all()
        logger.info(f"  Skills loaded: {n_skills}")

        # Register and start built-in drives
        if self._auto_start_drives:
            from .drive_scheduler import create_nis_drives
            for drive in create_nis_drives():
                self._drive_scheduler.register(drive)
            await self._drive_scheduler.start()
            logger.info(f"  Drives started: {len(self._drive_scheduler._drives)}")

        # Log startup event
        self._audit_chain.log(
            agent_id="neurokernel",
            action_type="startup",
            layer="core",
            payload={"skills": n_skills, "drives": len(self._drive_scheduler._drives)},
            tags=["lifecycle"],
        )

        self._started = True
        logger.info("NeuroKernel v2 ready")

    async def shutdown(self):
        """Graceful shutdown."""
        if self._drive_scheduler and self._drive_scheduler._running:
            await self._drive_scheduler.stop()
        try:
            if self._audit_chain:
                self._audit_chain.log(
                    agent_id="neurokernel", action_type="shutdown",
                    layer="core", payload={"uptime_secs": time.time() - self._start_time},
                    tags=["lifecycle"],
                )
        except Exception as e:
            logger.warning(f"NeuroKernel shutdown audit log failed: {e}")
        self._started = False

    # ── Main processing pipeline ───────────────────────────────────────────────

    async def process(
        self,
        agent_id: str,
        layer: str,
        action_type: str,
        user_input: str,
        handler: Callable[..., Coroutine],
        handler_args: Optional[Dict[str, Any]] = None,
        context_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        skip_scan: bool = False,
        skip_loop_guard: bool = False,
    ) -> KernelResult:
        """
        Full NeuroKernel processing pipeline:
          1. Scan input for injection threats
          2. Inject relevant skills into system prompt
          3. Check loop guard
          4. Execute handler (LLM call, tool, etc.)
          5. Log to AuditChain
          6. Return KernelResult

        Args:
            agent_id:    Which agent is processing (for audit)
            layer:       DIKW layer ("perception", "reasoning", "action", etc.)
            action_type: What kind of action ("llm_call", "tool_call", "arm_move", etc.)
            user_input:  The raw user message / query
            handler:     Async callable to execute (LLM, tool, etc.)
            handler_args: Extra kwargs for handler
            context_id:  Session/conversation ID for LoopGuard
            system_prompt: Base system prompt (skills will be appended)
            skip_scan:   Disable injection scanner (for trusted internal calls)
            skip_loop_guard: Disable loop guard (for startup drives)
        """
        request_id = uuid.uuid4().hex[:12]
        start = time.time()
        trace: List[str] = []
        handler_args = handler_args or {}
        context_id = context_id or agent_id

        self._request_count += 1
        self._ensure_started()

        # ── Layer 1: Injection Scanner ─────────────────────────────────────
        trace.append("scan")
        if not skip_scan and user_input:
            scan_result = self._scanner.scan(user_input, context=agent_id)
            if scan_result.action.value == "block":
                entry_id = self._audit_chain.log(
                    agent_id=agent_id, action_type="blocked_input",
                    layer="security",
                    payload={"reason": scan_result.summary(), "score": scan_result.score},
                    success=False, tags=["security", "blocked"],
                )
                return KernelResult(
                    request_id=request_id, success=False,
                    response="Input blocked by security scanner.",
                    audit_entry_id=entry_id,
                    blocked=True, block_reason=scan_result.summary(),
                    processing_ms=(time.time() - start) * 1000,
                    layer_trace=trace,
                )
            if scan_result.sanitized_text:
                user_input = scan_result.sanitized_text
                trace.append("sanitized")

        # ── Layer 2: Skill injection ───────────────────────────────────────
        trace.append("skills")
        skills_used: List[str] = []
        enriched_prompt = system_prompt or ""
        if self._skill_loader and user_input:
            matching_skills = self._skill_loader.skills_for_query(user_input, max_skills=2)
            skills_used = [s.name for s in matching_skills]
            if matching_skills:
                enriched_prompt = self._skill_loader.inject_into_prompt(enriched_prompt, user_input)
                trace.append(f"injected:{','.join(skills_used)}")

        # ── Layer 3: Loop guard ────────────────────────────────────────────
        trace.append("loop_guard")
        tool_args = {"input": user_input[:100], "action": action_type}  # always defined
        if not skip_loop_guard:
            report = self._loop_guard.check(action_type, tool_args, context_id, user_input)
            if report.detected and report.recommendation == "break":
                entry_id = self._audit_chain.log(
                    agent_id=agent_id, action_type="loop_blocked",
                    layer="attention",
                    payload={"loop_type": report.loop_type, "detail": report.details},
                    success=False, tags=["loop", "blocked"],
                )
                return KernelResult(
                    request_id=request_id, success=False,
                    response=f"Loop detected: {report.details}. Resetting.",
                    audit_entry_id=entry_id,
                    blocked=True, block_reason=report.details,
                    processing_ms=(time.time() - start) * 1000,
                    layer_trace=trace,
                )

        # ── Layer 4: Execute ──────────────────────────────────────────────
        trace.append("execute")
        try:
            if enriched_prompt and "system_prompt" in handler_args:
                handler_args["system_prompt"] = enriched_prompt
            elif enriched_prompt and system_prompt is not None:
                handler_args["_enriched_prompt"] = enriched_prompt

            output = await handler(user_input, **handler_args)
            success = True
            response_text = str(output) if not isinstance(output, str) else output
            trace.append("ok")
        except Exception as e:
            success = False
            response_text = f"Handler error: {e}"
            trace.append(f"error:{type(e).__name__}")
            logger.error(f"NeuroKernel handler error [{agent_id}]: {e}")

        # ── Layer 5: Record + Audit ────────────────────────────────────────
        trace.append("audit")
        duration_ms = (time.time() - start) * 1000

        if not skip_loop_guard:
            self._loop_guard.record(action_type, tool_args, context_id, made_progress=success)

        entry_id = self._audit_chain.log(
            agent_id=agent_id,
            action_type=action_type,
            layer=layer,
            payload={
                "input_preview": user_input[:200],
                "response_preview": response_text[:200],
                "context_id": context_id,
            },
            skill_attribution=skills_used,
            success=success,
            duration_ms=duration_ms,
            tags=[layer, action_type],
        )

        return KernelResult(
            request_id=request_id,
            success=success,
            response=response_text,
            audit_entry_id=entry_id,
            skills_used=skills_used,
            processing_ms=duration_ms,
            layer_trace=trace,
        )

    # ── Tool call wrapper ──────────────────────────────────────────────────────

    async def tool_call(
        self,
        agent_id: str,
        tool_name: str,
        tool_fn: Callable[..., Coroutine],
        args: Dict[str, Any],
        context_id: str = "default",
    ) -> KernelResult:
        """
        Execute a tool call through the NeuroKernel pipeline.
        Includes loop guard, audit, and loop guard recording.
        """
        self._ensure_started()
        start = time.time()
        request_id = uuid.uuid4().hex[:12]

        # Loop guard
        report = self._loop_guard.check(tool_name, args, context_id)
        if report.detected and report.recommendation == "break":
            return KernelResult(
                request_id=request_id, success=False,
                response=f"Tool loop blocked: {report.details}",
                blocked=True, block_reason=report.details,
                processing_ms=(time.time() - start) * 1000,
                layer_trace=["loop_guard:break"],
            )

        try:
            result = await tool_fn(**args)
            success = True
            response = str(result)[:500]
        except Exception as e:
            success = False
            response = str(e)

        duration_ms = (time.time() - start) * 1000
        self._loop_guard.record(tool_name, args, context_id, made_progress=success)

        entry_id = self._audit_chain.log(
            agent_id=agent_id, action_type="tool_call",
            layer="action",
            payload={"tool": tool_name, "args": args, "response_preview": response[:200]},
            success=success, duration_ms=duration_ms,
            tags=["tool_call", tool_name],
        )

        return KernelResult(
            request_id=request_id, success=success, response=response,
            audit_entry_id=entry_id, processing_ms=duration_ms,
            layer_trace=["loop_guard:ok", "execute", "audit"],
        )

    # ── Introspection ──────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full kernel status for /neurokernel/status endpoint."""
        self._ensure_started()
        uptime = time.time() - self._start_time
        return {
            "version": "2.0.0",
            "started": self._started,
            "uptime_secs": round(uptime, 1),
            "request_count": self._request_count,
            "components": {
                "skill_loader":  self._skill_loader.stats() if self._skill_loader else None,
                "audit_chain":   self._audit_chain.stats() if self._audit_chain else None,
                "loop_guard":    self._loop_guard.stats() if self._loop_guard else None,
                "scanner":       self._scanner.stats() if self._scanner else None,
                "drive_scheduler": {
                    "drives": len(self._drive_scheduler._drives) if self._drive_scheduler else 0,
                    "running": self._drive_scheduler._running if self._drive_scheduler else False,
                } if self._drive_scheduler else None,
            },
            "dikw_layers": {
                "data":        "SkillLoader (raw SKILL.md on disk)",
                "information": "Scanner + LoopGuard (structured, safe context)",
                "knowledge":   "AuditChain (verified, tamper-proof decisions)",
                "wisdom":      "DriveScheduler (autonomous, adaptive scheduling)",
            }
        }

    def _ensure_started(self):
        """Initialize components synchronously if startup() was never awaited.

        NOTE: drives are NOT started here because starting them requires
        asyncio.create_task() which needs a running event loop. Call
        await kernel.startup() at app startup to get drives running.
        """
        if self._started:
            return
        self._skill_loader     = self._skill_loader    or _skill_loader()
        self._audit_chain      = self._audit_chain     or _audit_chain()
        self._loop_guard       = self._loop_guard      or _loop_guard()
        self._scanner          = self._scanner         or _scanner()
        self._drive_scheduler  = self._drive_scheduler or _drive_scheduler()
        # Load skills synchronously (no event loop needed)
        try:
            self._skill_loader.load_all()
        except Exception as e:
            logger.warning(f"_ensure_started: skill load failed: {e}")
        self._started = True


# ── Module-level singleton ────────────────────────────────────────────────────

_neurokernel: Optional[NeuroKernel] = None


def get_neurokernel(auto_start_drives: bool = True) -> NeuroKernel:
    global _neurokernel
    if _neurokernel is None:
        _neurokernel = NeuroKernel(auto_start_drives=auto_start_drives)
    return _neurokernel
