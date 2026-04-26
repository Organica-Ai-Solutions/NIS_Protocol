"""
NIS Protocol — NeuroKernel v2 API Routes
==========================================
Exposes all NeuroKernel components via REST endpoints.
These power the NeuroKernel tab in the NIS Protocol dashboard.

Endpoints:
  GET  /neurokernel/status            — Full kernel status (DIKW layers, all components)
  GET  /neurokernel/audit             — Recent audit chain entries
  POST /neurokernel/audit/verify      — Verify chain integrity
  GET  /neurokernel/skills            — List loaded skills
  POST /neurokernel/skills/refresh    — Hot-reload SKILL.md files
  GET  /neurokernel/drives            — List all autonomous drives
  POST /neurokernel/drives/{id}/trigger — Manually trigger a drive
  POST /neurokernel/drives/{id}/pause   — Pause a drive
  POST /neurokernel/drives/{id}/resume  — Resume a drive
  GET  /neurokernel/loop_guard        — Loop guard stats
  POST /neurokernel/loop_guard/reset  — Reset loop guard for a context
  POST /neurokernel/scan              — Scan text for injection threats
  POST /neurokernel/process           — Run text through full NeuroKernel pipeline
"""

import time
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("nis.routes.neurokernel")

router = APIRouter(prefix="/neurokernel", tags=["NeuroKernel v2"])


def _kernel():
    from src.core.neurokernel import get_neurokernel
    return get_neurokernel()


# ── Models ─────────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    text: str
    context: str = "api"


class ProcessRequest(BaseModel):
    agent_id: str = "neurokernel-api"
    layer: str = "reasoning"
    action_type: str = "llm_call"
    user_input: str
    context_id: Optional[str] = None
    skip_scan: bool = False


class TriggerRequest(BaseModel):
    approve: bool = False


class ResetLoopRequest(BaseModel):
    context_id: str = "default"


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/status")
async def neurokernel_status():
    """
    Full NeuroKernel v2 status — all 5 components, DIKW layer mapping,
    uptime, request count, and drive states.
    """
    try:
        return _kernel().status()
    except Exception as e:
        raise HTTPException(500, f"NeuroKernel status error: {e}")


@router.get("/health")
async def neurokernel_health():
    """Quick liveness check."""
    k = _kernel()
    return {"ok": True, "started": k._started, "requests": k._request_count}


# ── Audit Chain ────────────────────────────────────────────────────────────────

@router.get("/audit")
async def get_audit_recent(n: int = 50, agent_id: Optional[str] = None,
                            action_type: Optional[str] = None, since_minutes: Optional[float] = None):
    """Get recent audit chain entries, optionally filtered. Always returns a JSON array."""
    from src.core.audit_chain import get_audit_chain
    chain = get_audit_chain()
    if agent_id or action_type or since_minutes:
        result = chain.query(agent_id=agent_id, action_type=action_type,
                             since_seconds=since_minutes * 60 if since_minutes else None,
                             limit=n)
    else:
        result = chain.recent(n)
    # Normalise: dashboard JS expects a plain list
    if isinstance(result, dict):
        result = result.get("entries", result.get("items", []))
    return result if isinstance(result, list) else []


@router.post("/audit/verify")
async def verify_audit_chain():
    """Verify the cryptographic integrity of the entire AuditChain."""
    from src.core.audit_chain import get_audit_chain
    result = get_audit_chain().verify()
    if not result["valid"]:
        logger.error(f"AuditChain integrity failure: {result}")
    return result


@router.get("/audit/stats")
async def audit_stats():
    from src.core.audit_chain import get_audit_chain
    return get_audit_chain().stats()


# ── Skill Loader ───────────────────────────────────────────────────────────────

@router.get("/skills")
async def list_skills():
    """List all loaded SKILL.md files with tags, summaries, and usage counts."""
    from src.core.skill_loader import get_skill_loader
    loader = get_skill_loader()
    return {"skills": loader.list_skills(), "stats": loader.stats()}


@router.post("/skills/refresh")
async def refresh_skills():
    """Hot-reload all SKILL.md files. Returns count of refreshed skills."""
    from src.core.skill_loader import get_skill_loader
    loader = get_skill_loader()
    refreshed = loader.refresh()
    return {"refreshed": refreshed, "total": len(loader.list_skills())}


@router.get("/skills/context")
async def skill_context_for(query: str = "pick up the block"):
    """Show what skill context would be injected for a given query."""
    from src.core.skill_loader import get_skill_loader
    loader = get_skill_loader()
    matching = loader.skills_for_query(query, max_skills=3)
    context = loader.build_context_for(query)
    return {
        "query": query,
        "matching_skills": [s.name for s in matching],
        "injected_context_preview": context[:800],
        "context_length": len(context),
    }


# ── Drive Scheduler ────────────────────────────────────────────────────────────

@router.get("/drives")
async def list_drives():
    """List all autonomous drives with status, schedule, and last result."""
    from src.core.drive_scheduler import get_drive_scheduler
    scheduler = get_drive_scheduler()
    return {
        "drives": scheduler.list_drives(),
        "running": scheduler._running,
        "recent_results": scheduler.recent_results(10),
    }


@router.post("/drives/{drive_id}/trigger")
async def trigger_drive(drive_id: str, req: TriggerRequest = TriggerRequest()):
    """Manually trigger an autonomous drive. Set approve=true for approval-gated drives."""
    from src.core.drive_scheduler import get_drive_scheduler
    scheduler = get_drive_scheduler()
    result = await scheduler.trigger(drive_id, approve=req.approve)
    if result is None:
        raise HTTPException(404, f"Drive '{drive_id}' not found")
    return {
        "drive_id": result.drive_id,
        "run_id": result.run_id,
        "success": result.success,
        "duration_ms": result.duration_ms,
        "output": str(result.output)[:500] if result.output else None,
        "error": result.error,
    }


@router.post("/drives/{drive_id}/pause")
async def pause_drive(drive_id: str):
    from src.core.drive_scheduler import get_drive_scheduler
    get_drive_scheduler().pause(drive_id)
    return {"drive_id": drive_id, "status": "paused"}


@router.post("/drives/{drive_id}/resume")
async def resume_drive(drive_id: str):
    from src.core.drive_scheduler import get_drive_scheduler
    get_drive_scheduler().resume(drive_id)
    return {"drive_id": drive_id, "status": "resumed"}


# ── Loop Guard ─────────────────────────────────────────────────────────────────

@router.get("/loop_guard")
async def loop_guard_stats(context_id: Optional[str] = None):
    """Get loop guard statistics for all contexts or a specific one."""
    from src.core.loop_guard import get_loop_guard
    return get_loop_guard().stats(context_id)


@router.post("/loop_guard/reset")
async def reset_loop_guard(req: ResetLoopRequest):
    """Reset loop guard state for a specific context (e.g., start of new session)."""
    from src.core.loop_guard import get_loop_guard
    get_loop_guard().reset(req.context_id)
    return {"context_id": req.context_id, "reset": True}


# ── Injection Scanner ──────────────────────────────────────────────────────────

@router.post("/scan")
async def scan_text(req: ScanRequest):
    """
    Scan text for prompt injection, jailbreak, hardware override,
    shell injection, data exfiltration, and role confusion attempts.
    """
    from src.core.prompt_injection_scanner import get_scanner
    scanner = get_scanner()
    result = scanner.scan(req.text, context=req.context)
    return {
        "safe": result.safe,
        "action": result.action.value,
        "score": result.score,
        "threats": [
            {
                "pattern_id": t.pattern_id,
                "category": t.category,
                "severity": t.severity.value,
                "description": t.description,
                "matched": t.matched_text[:60],
            }
            for t in result.threats
        ],
        "sanitized_preview": result.sanitized_text[:200] if result.sanitized_text else None,
        "scan_ms": round(result.scan_ms, 2),
        "stats": scanner.stats(),
    }


# ── Process (full pipeline) ────────────────────────────────────────────────────

@router.post("/process")
async def process_through_kernel(req: ProcessRequest):
    """
    Run text through the full NeuroKernel pipeline:
    Scan → Skill inject → Loop guard → Echo handler → Audit log → Return result.

    This is the demo/test endpoint. In production, the kernel is called
    internally by agent handlers.
    """
    kernel = _kernel()

    async def echo_handler(text: str, **kwargs) -> str:
        """Demo handler: just echoes back with skill context info."""
        enriched = kwargs.get("_enriched_prompt", "")
        skills_note = f"\n[Skills injected: {len(enriched)} chars]" if enriched else ""
        return f"NeuroKernel processed: '{text[:100]}'{skills_note}"

    result = await kernel.process(
        agent_id=req.agent_id,
        layer=req.layer,
        action_type=req.action_type,
        user_input=req.user_input,
        handler=echo_handler,
        context_id=req.context_id,
        skip_scan=req.skip_scan,
    )

    return {
        "request_id": result.request_id,
        "success": result.success,
        "response": result.response,
        "blocked": result.blocked,
        "block_reason": result.block_reason,
        "skills_used": result.skills_used,
        "audit_entry_id": result.audit_entry_id,
        "processing_ms": round(result.processing_ms, 2),
        "layer_trace": result.layer_trace,
    }


# ── DIKW Analysis ──────────────────────────────────────────────────────────────

def get_loop_guard_stub():
    """Safely return loop_guard stats; used by /dikw and /neurokernel/status."""
    try:
        from src.core.loop_guard import get_loop_guard
        return get_loop_guard().stats()
    except Exception:
        return {}


def _get_scanner_stub():
    """Safely return scanner stats; used by /dikw."""
    try:
        from src.core.prompt_injection_scanner import get_scanner
        return get_scanner().stats()
    except Exception:
        return {}


@router.get("/dikw")
async def dikw_analysis():
    """
    Show how the current NeuroKernel state maps to the DIKW pyramid.
    Data → Information → Knowledge → Wisdom
    """
    from src.core.skill_loader import get_skill_loader
    from src.core.audit_chain import get_audit_chain
    from src.core.drive_scheduler import get_drive_scheduler

    loader = get_skill_loader()
    chain = get_audit_chain()
    scheduler = get_drive_scheduler()

    return {
        "dikw_pyramid": {
            "data": {
                "layer": "Data",
                "component": "SkillLoader",
                "description": "Raw domain expertise files on disk (SKILL.md, agent.toml)",
                "metrics": loader.stats(),
            },
            "information": {
                "layer": "Information",
                "component": "PromptInjectionScanner + LoopGuard",
                "description": "Scanned, structured, safe context — noise and threats removed",
                "metrics": {
                    "loop_guard": get_loop_guard_stub(),
                    "scanner": _get_scanner_stub(),
                },
            },
            "knowledge": {
                "layer": "Knowledge",
                "component": "AuditChain",
                "description": "Verified, tamper-proof log of every decision and its attribution",
                "metrics": chain.stats(),
            },
            "wisdom": {
                "layer": "Wisdom",
                "component": "DriveScheduler",
                "description": "Autonomous, adaptive drives — the system acts without being asked",
                "metrics": {
                    "total_drives": len(scheduler._drives),
                    "active_drives": sum(1 for d in scheduler._drives.values() if d.enabled),
                    "scheduler_running": scheduler._running,
                    "recent_results": scheduler.recent_results(5),
                },
            },
        }
    }


# get_loop_guard_stub defined above /dikw route — kept for reference only
# (duplicate removed)
