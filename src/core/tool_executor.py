"""
NIS Protocol — Shared Tool Executor
====================================
HTTP-based tool runner that works in both REST endpoints AND WebSocket handlers.
Returns structured results so any caller (CLI, REST, WS) can consume them identically.

Tool registry:
  camera_snapshot  — Pi camera via NeuroLinux Agent
  xarm_control     — xArm named-position control via OpenClaw bridge
  cosmos_plan      — NVIDIA Cosmos Reason2 spatial reasoning
  system_status    — Live health of all NIS services
  list_skills      — Enumerate available OpenClaw skills
  memory_search    — Semantic search across ChromaDB memories
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("nis.tool_executor")

# ── Service endpoints (override with env vars for containerised deploys) ─────
import os
AGENT_URL   = os.getenv("NEUROLINUX_AGENT_URL", "http://localhost:8085")
NIS_URL     = os.getenv("NIS_PROTOCOL_URL",      "http://localhost:8000")
COSMOS_URL  = os.getenv("COSMOS_URL",            "http://localhost:8100")


def _r(d: dict, **extra) -> dict:
    """Merge base result dict with extras."""
    d.update(extra)
    return d


# ── Individual tool implementations ──────────────────────────────────────────

async def camera_snapshot(query: str = "") -> dict:
    """Capture a snapshot from the Pi camera and return base64 image."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(f"{AGENT_URL}/camera/snapshot")
            if r.status_code == 200:
                data = r.json()
                img = data.get("image_base64") or data.get("image")
                return {
                    "tool": "camera_snapshot",
                    "ok": True,
                    "image_base64": img,
                    "summary": "Snapshot captured" if img else "No image data in response",
                    "raw": data,
                }
            return {"tool": "camera_snapshot", "ok": False,
                    "summary": f"Camera HTTP {r.status_code}", "image_base64": None}
    except Exception as e:
        return {"tool": "camera_snapshot", "ok": False, "summary": f"Camera offline: {e}", "image_base64": None}


async def xarm_control(command: str, named_position: str | None = None) -> dict:
    """Send an xArm command via the OpenClaw bridge or directly to the agent."""
    CMD_TO_NAMED = {
        "home":          "home",
        "pick":          "pick_table",
        "place":         "place_bin",
        "inspect":       "inspect",
        "open_gripper":  None,   # uses dedicated endpoint
        "close_gripper": None,
        "wave":          None,
        "stop":          None,
        "ready":         "home",
        "lift":          "lift_grip",
    }

    # Named-position move (preferred)
    target = named_position or CMD_TO_NAMED.get(command)
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            if command == "open_gripper":
                r = await c.post(f"{AGENT_URL}/arm/gripper/open")
                return {"tool": "xarm_control", "ok": r.status_code == 200,
                        "summary": "Gripper opened" if r.status_code == 200 else f"HTTP {r.status_code}",
                        "command": command}

            if command == "close_gripper":
                r = await c.post(f"{AGENT_URL}/arm/gripper/close")
                return {"tool": "xarm_control", "ok": r.status_code == 200,
                        "summary": "Gripper closed" if r.status_code == 200 else f"HTTP {r.status_code}",
                        "command": command}

            if command == "stop":
                r = await c.post(f"{AGENT_URL}/arm/stop")
                return {"tool": "xarm_control", "ok": r.status_code == 200,
                        "summary": "Arm stopped", "command": command}

            if command == "wave":
                r = await c.post(f"{AGENT_URL}/arm/wave")
                return {"tool": "xarm_control", "ok": r.status_code == 200,
                        "summary": "Arm waved" if r.status_code == 200 else f"HTTP {r.status_code}",
                        "command": command}

            if target:
                r = await c.post(f"{AGENT_URL}/arm/named/{target}")
                return {
                    "tool": "xarm_control",
                    "ok": r.status_code == 200,
                    "summary": f"Moved to '{target}'" if r.status_code == 200 else f"HTTP {r.status_code} for '{target}'",
                    "command": command,
                    "position": target,
                }

            # Fallback — OpenClaw bridge
            r = await c.post(f"{NIS_URL}/openclaw/invoke",
                             json={"tool": "nis_xarm", "args": {"command": command}})
            if r.status_code == 200:
                data = r.json()
                return {"tool": "xarm_control", "ok": True,
                        "summary": str(data.get("result", {}).get("result", f"{command} executed")),
                        "command": command}
            return {"tool": "xarm_control", "ok": False,
                    "summary": f"OpenClaw bridge HTTP {r.status_code}", "command": command}
    except Exception as e:
        return {"tool": "xarm_control", "ok": False, "summary": f"xArm error: {e}", "command": command}


async def pick_and_place(image_b64: str | None = None) -> dict:
    """
    Full pick-and-place via Pi agent's /arm/pick_and_place endpoint.
    The agent handles the 9-step sequence with proper gripper control:
      home → inspect → open_gripper → pick_table → close_gripper
           → lift_grip → place_bin → open_gripper → home
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{AGENT_URL}/arm/pick_and_place")
            if r.status_code == 200:
                data = r.json()
                return {
                    "tool": "pick_and_place",
                    "ok": data.get("ok", False),
                    "summary": data.get("summary", "Pipeline complete"),
                    "steps": data.get("steps", []),
                    "total": data.get("total", 0),
                }
            return {"tool": "pick_and_place", "ok": False,
                    "summary": f"Agent returned HTTP {r.status_code}",
                    "steps": []}
    except Exception as e:
        return {"tool": "pick_and_place", "ok": False,
                "summary": f"Pick-and-place error: {e}", "steps": []}


async def cosmos_plan(query: str, image_b64: str | None = None) -> dict:
    """Run Cosmos Reason2 spatial reasoning via OpenClaw bridge."""
    payload: dict[str, Any] = {"tool": "nis_cosmos_plan", "args": {"query": query}}
    if image_b64:
        payload["args"]["image_base64"] = image_b64
    try:
        async with httpx.AsyncClient(timeout=35.0) as c:
            r = await c.post(f"{NIS_URL}/openclaw/invoke", json=payload)
            if r.status_code == 200:
                data = r.json()
                actions = data.get("result", {}).get("action_recommendations", [])
                plan_text = data.get("result", {}).get("spatial_analysis", "")
                return {
                    "tool": "cosmos_plan",
                    "ok": True,
                    "summary": "\n".join(f"• {a}" for a in actions) if actions else plan_text or "Cosmos returned no plan",
                    "actions": actions,
                    "plan_text": plan_text,
                    "raw": data.get("result", {}),
                }
            return {"tool": "cosmos_plan", "ok": False, "summary": f"OpenClaw bridge HTTP {r.status_code}"}
    except Exception as e:
        return {"tool": "cosmos_plan", "ok": False, "summary": f"Cosmos error: {e}"}


async def system_status() -> dict:
    """Ping all NIS services and return health matrix."""
    checks = [
        ("NIS Protocol",    f"{NIS_URL}/health"),
        ("NeuroLinux Agent",f"{AGENT_URL}/health"),
        ("OpenClaw Bridge", f"{NIS_URL}/openclaw/status"),
        ("Cosmos Reason2",  "http://localhost:8100/health"),
        ("H100 Relay",      "http://localhost:8101/health"),
    ]
    lines = []
    services: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=3.0) as c:
        for name, url in checks:
            try:
                r = await c.get(url)
                status = "online"
                lines.append(f"[OK] {name}: {r.status_code}")
            except Exception:
                status = "offline"
                lines.append(f"[--] {name}: offline")
            services[name] = {"status": status, "url": url}
    return {
        "tool": "system_status",
        "ok": True,
        "summary": "\n".join(lines),
        "services": services,
    }


async def list_skills() -> dict:
    """List all registered OpenClaw / NIS tools."""
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.post(f"{NIS_URL}/openclaw/invoke",
                             json={"tool": "nis_skills", "args": {}})
            if r.status_code == 200:
                data = r.json()
                skills = data.get("result", {}).get("skills", [])
                lines = [f"• {s['name']}: {s.get('description','')[:80]}" for s in skills[:15]]
                return {"tool": "list_skills", "ok": True,
                        "summary": "\n".join(lines) if lines else "No skills registered yet",
                        "skills": skills}
    except Exception as e:
        return {"tool": "list_skills", "ok": False, "summary": f"Skills unavailable: {e}"}
    return {"tool": "list_skills", "ok": False, "summary": "Skills unavailable"}


async def memory_search(query: str, top_k: int = 3, memory_system=None) -> dict:
    """Retrieve relevant memories from ChromaDB (injected or via HTTP)."""
    if memory_system is None:
        return {"tool": "memory_search", "ok": False, "summary": "Memory system not initialised", "memories": []}
    try:
        results = await memory_system.retrieve(query, top_k=top_k, min_relevance=0.25)
        snippets = []
        for r in results:
            entry = r.entry if hasattr(r, "entry") else r
            content = getattr(entry, "content", str(entry))
            score   = getattr(r, "combined_score", 0.0)
            snippets.append({"content": content[:300], "relevance": round(score, 3)})
        return {
            "tool": "memory_search",
            "ok": True,
            "summary": f"Found {len(snippets)} relevant memories",
            "memories": snippets,
        }
    except Exception as e:
        logger.warning("Memory search failed: %s", e)
        return {"tool": "memory_search", "ok": False, "summary": f"Memory error: {e}", "memories": []}


# ── Intent → tool mapping ─────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "vision":  ["snapshot", "photo", "picture", "what do you see", "look", "camera", "see", "show me"],
    "xarm":    ["pick", "place", "grab", "stack", "sort", "move arm", "xarm", "gripper",
                "open gripper", "close gripper", "wave", "home", "inspect", "lift arm",
                "put down", "pick and place", "pick-and-place"],
    "cosmos":  ["cosmos", "plan", "robot plan", "cookoff", "execute plan", "spatial", "reason"],
    "status":  ["status", "health", "services", "running", "system", "ports", "uptime", "online"],
    "skills":  ["skill", "skills", "openclaw", "what can you do", "capabilities", "tools"],
}


def detect_intent(message: str) -> str:
    """Classify user message into a routing intent."""
    m = message.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in m for k in keywords):
            return intent
    return "chat"


async def dispatch(intent: str, message: str,
                   image_b64: str | None = None,
                   memory_system=None,
                   context_id: str = "default") -> dict:
    """
    Route intent to the correct tool and return a unified result dict.
    Protected by NeuroKernel LoopGuard + AuditChain.
    """
    import time as _time
    _start = _time.time()

    # ── NeuroKernel: Loop guard ────────────────────────────────────────────
    try:
        from .loop_guard import get_loop_guard
        _guard = get_loop_guard()
        _args = {"intent": intent, "msg": message[:80]}
        _report = _guard.check(intent, _args, context_id=context_id, semantic_text=message)
        if _report.detected and _report.recommendation == "break":
            logger.warning(f"[LoopGuard] Tool loop blocked: {_report.details}")
            return {"tool": intent, "ok": False,
                    "summary": f"Loop guard: {_report.details}", "loop_blocked": True}
    except Exception as _lge:
        logger.debug(f"LoopGuard check skipped: {_lge}")
        _guard = None
        _args = {}

    if intent == "vision":
        result = await camera_snapshot(message)
        if result["ok"] and result.get("image_base64") and \
                any(k in message.lower() for k in ["what", "where", "how", "plan", "cosmos", "reason"]):
            cosmos = await cosmos_plan(message, result["image_base64"])
            result["cosmos"] = cosmos

    elif intent == "xarm":
        m = message.lower()
        if "pick and place" in m or "pick-and-place" in m or "full demo" in m:
            result = await pick_and_place()
        elif "open" in m and "gripper" in m:
            result = await xarm_control("open_gripper")
        elif "close" in m and "gripper" in m:
            result = await xarm_control("close_gripper")
        elif "wave" in m:
            result = await xarm_control("wave")
        elif "home" in m:
            result = await xarm_control("home")
        elif "inspect" in m:
            result = await xarm_control("inspect")
        elif "pick" in m:
            result = await xarm_control("pick")
        elif "place" in m:
            result = await xarm_control("place")
        elif "lift" in m:
            result = await xarm_control("lift")
        elif "stop" in m:
            result = await xarm_control("stop")
        else:
            result = await xarm_control("status")

    elif intent == "cosmos":
        result = await cosmos_plan(message, image_b64)

    elif intent == "status":
        result = await system_status()

    elif intent == "skills":
        result = await list_skills()

    else:
        result = {"tool": "none", "ok": True, "summary": "", "intent": "chat"}

    # ── NeuroKernel: Record + Audit ────────────────────────────────────────
    try:
        if _guard:
            _guard.record(intent, _args, context_id=context_id, made_progress=result.get("ok", False))
        from .audit_chain import get_audit_chain
        get_audit_chain().log(
            agent_id="tool_executor",
            action_type="tool_call",
            layer="action",
            payload={"intent": intent, "tool": result.get("tool", intent),
                     "ok": result.get("ok", False), "summary": result.get("summary", "")[:120]},
            success=result.get("ok", False),
            duration_ms=(_time.time() - _start) * 1000,
            tags=["tool_call", intent],
        )
    except Exception as _ae:
        logger.debug(f"NeuroKernel audit skipped: {_ae}")

    return result


# ── Async sleep helper (avoids importing asyncio at module level) ─────────────

async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
