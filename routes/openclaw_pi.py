"""
NIS Protocol v4.0 - OpenClaw Bridge

Allows OpenClaw gateway to invoke NIS Protocol as an agent/tool.
Maps OpenClaw tool invocations to NIS Protocol endpoints (chat, cosmos, robotics,
trading, and the full Organica AI stack).

Supported tools:
  nis_chat          — Send a message to the NIS Protocol LLM / chat pipeline
  nis_cosmos_plan   — Get a robot action plan (Cosmos Cookoff / cookoff route)
  nis_skills        — List available OpenClaw-compatible skills
  nis_xarm          — Send a command to the Hiwonder xArm via NeuroLinux
  nis_stack         — Full Organica AI stack health snapshot
  nis_neurokernel   — Query/control NeuroKernel v2 (DIKW layers, drives, agents)
  nis_openfang      — Execute NIS tools via OpenFang MCP interface
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.openclaw")

router = APIRouter(prefix="/openclaw", tags=["OpenClaw Bridge"])

# ── API key guard ──────────────────────────────────────────────────────────────
_NIS_API_KEY = os.getenv("NIS_OPENCLAW_KEY", os.getenv("NIS_API_KEY", ""))
_oc_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _require_openclaw_key(key: str = Security(_oc_key_header)):
    """Reject requests that don't carry a valid gateway key."""
    if _NIS_API_KEY and key != _NIS_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key")
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────
#  Dependency Injection
# ──────────────────────────────────────────────────────────

def get_llm_provider():
    """LLM provider injected at startup."""
    return getattr(router, "_llm_provider", None)


def set_dependencies(llm_provider=None) -> None:
    """Call from main.py startup to inject services."""
    router._llm_provider = llm_provider  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────

class InvokeRequest(BaseModel):
    """OpenClaw tool invocation request."""
    tool: str = Field(..., description="Tool name, e.g. nis_chat, nis_cosmos_plan")
    args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Tool arguments")


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def _get_memory():
    """Get enhanced chat memory instance if available."""
    try:
        import routes.memory as _mem_route
        return getattr(_mem_route.router, '_enhanced_chat_memory', None)
    except Exception:
        return None


async def _handle_chat(args: Dict[str, Any]) -> Dict[str, Any]:
    """Route nis_chat to the LLM provider or chat/simple endpoint."""
    msg = (
        args.get("message")
        or args.get("query")
        or args.get("text")
        or ""
    ).strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message or query required")

    conv_id = args.get("conversation_id") or args.get("session_id") or "default"
    memory = _get_memory()

    # Retrieve recent conversation context for continuity
    context_lines = []
    if memory:
        try:
            recent = await memory._get_conversation_messages(conv_id, limit=8)
            for m in recent[-6:]:  # last 6 messages
                role_label = "User" if m.role == "user" else "NIS"
                context_lines.append(f"{role_label}: {m.content[:300]}")
        except Exception:
            pass

    llm = get_llm_provider()
    if llm:
        try:
            system_prompt = (
                "You are NIS Protocol v4.0, the central AI brain of Organica AI Solutions, "
                "built by Diego Torres (founder, 2022). "
                "You run on a Raspberry Pi (192.168.1.163) connected to a physical 6DOF xArm robot, "
                "YOLO object detection, and persistent memory. "
                "Your connected services (Windows dev machine):\n"
                "- AlphaCortex (:5000) — US equity trading, RSI/MACD + Claude AI, Alpaca paper trading\n"
                "- ArbitrageMachine (:8000) — Crypto arb Binance/Coinbase/Kraken, LangGraph AI agent\n"
                "- SmartPortfolio (:8002) — Portfolio optimization (long-running project)\n"
                "- Organica Framework (:8900) — 39 specialized AI agents\n"
                "- Orion (:8080) — TypeScript coding AI agent\n"
                "- NIS Docker (:8007) — local fallback instance of yourself\n\n"
                "You can dispatch to all services via nis_alphacortex, nis_arbitrage, and nis_stack tools. "
                "You have persistent memory and multi-agent reasoning. "
                "Be direct and technical. When asked about services, use the tools to get real data — don't guess."
            )
            messages = [{"role": "system", "content": system_prompt}]
            if context_lines:
                messages.append({
                    "role": "system",
                    "content": "Recent conversation context:\n" + "\n".join(context_lines)
                })
            messages.append({"role": "user", "content": msg})

            # Save user message before LLM call
            if memory:
                try:
                    await memory.add_message(conv_id, "user", msg)
                except Exception:
                    pass

            result = await llm.generate_response(messages=messages)
            response_text = (
                result.get("response") or result.get("content") or result.get("text") or str(result)
            )

            # Save assistant response
            if memory and response_text:
                try:
                    await memory.add_message(conv_id, "assistant", response_text)
                except Exception:
                    pass

            return {
                "response": response_text,
                "provider": result.get("provider", "nis_llm"),
                "model": result.get("model", "unknown"),
                "conversation_id": conv_id,
                "memory_active": memory is not None,
            }
        except Exception as exc:
            logger.warning("LLM invoke failed, using fallback: %s", exc)

    # Fallback – no provider available
    return {
        "response": (
            "NIS Protocol received your message. "
            "(LLM provider not available at this time — start the server with API keys.)"
        ),
        "provider": "fallback",
        "model": "none",
        "conversation_id": conv_id,
    }


async def _handle_cosmos_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    """Route nis_cosmos_plan to the Cosmos Cookoff pipeline.

    Priority: H100 Reason2 /robot-plan → local reasoner fallback.
    """
    import os, httpx
    query = (args.get("query") or args.get("task") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query or task required")

    robot_state = args.get("robot_state") or {}
    image_b64 = args.get("image_base64")

    # H100_REASON_URL must be explicitly set — default is empty so we skip it
    # (H100 NVIDIA grant ended Mar 2026; trying it just adds timeout latency)
    H100_REASON_URL = os.environ.get("H100_REASON_URL", "")

    # ── 1. Try H100 Cosmos Reason2 /robot-plan (only if explicitly configured) ─
    if H100_REASON_URL:
        try:
            body: Dict[str, Any] = {"command": query, "robot_type": "xarm"}
            if image_b64:
                body["image_base64"] = image_b64
            if robot_state:
                body["robot_state"] = robot_state
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(f"{H100_REASON_URL}/robot-plan", json=body)
                if r.status_code == 200:
                    d = r.json()
                    actions = d.get("action_plan", [])
                    if not actions and d.get("action"):
                        actions = [d["action"]]
                    if not actions:
                        raw = d.get("reasoning", d.get("response", ""))
                        actions = [ln.lstrip("0123456789.-*• ").strip()
                                   for ln in raw.split("\n")
                                   if ln.strip() and len(ln.strip()) > 3][:6]
                    logger.info("openclaw cosmos_plan: H100 OK")
                    return {
                        "cosmos_reasoning": {
                            "reasoning_chain": d.get("reasoning", "")[:600],
                            "spatial_understanding": {"source": "h100_robot_plan"},
                        },
                        "action_recommendations": actions or ["inspect", "reach", "grasp"],
                        "combined_confidence": d.get("confidence", 0.85),
                        "nis_physics_validation": {"safe": d.get("safe_to_execute", True)},
                        "robot_state": robot_state,
                        "source": "h100_cosmos_reason2",
                        "timestamp": time.time(),
                    }
        except Exception as exc:
            logger.warning("openclaw cosmos_plan H100 failed: %s", exc)

    # ── 2. Local reasoner fallback ────────────────────────────────────────────
    try:
        import numpy as np
        from src.agents.cosmos import get_cosmos_reasoner

        reasoner = get_cosmos_reasoner()
        if not reasoner.initialized:
            await reasoner.initialize()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        if image_b64:
            try:
                import base64, io
                raw = base64.b64decode(image_b64)
                try:
                    from PIL import Image
                    image = np.array(Image.open(io.BytesIO(raw)))
                except ImportError:
                    pass
            except Exception as e:
                logger.debug("Image decode: %s", e)

        result = await reasoner.reason(image=image, task=query, constraints=[])
        plan = result.get("plan", [])
        actions = [s.get("action", str(s)) for s in plan] if isinstance(plan, list) else [plan]

        return {
            "cosmos_reasoning": {
                "reasoning_chain": result.get("reasoning_trace", ""),
                "spatial_understanding": result.get("physics_understanding", {}),
            },
            "action_recommendations": actions,
            "combined_confidence": result.get("confidence", 0.75),
            "nis_physics_validation": result.get("safety_check", {}),
            "robot_state": robot_state,
            "source": "local_fallback",
            "timestamp": time.time(),
        }

    except (ImportError, AttributeError, Exception) as exc:
        logger.warning("openclaw cosmos_plan local fallback failed: %s", exc)
        return {
            "cosmos_reasoning": {
                "reasoning_chain": f"Simulated plan for: {query}",
                "spatial_understanding": {},
            },
            "action_recommendations": [
                f"Analyze scene for '{query}'",
                "Plan gripper trajectory",
                "Execute motion",
                "Return to home",
            ],
            "combined_confidence": 0.5,
            "nis_physics_validation": {},
            "robot_state": robot_state,
            "source": "simulation",
            "timestamp": time.time(),
            "simulation": True,
        }


def _handle_skills(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return available skills list."""
    try:
        from src.skills.openclaw_skill_loader import list_skills
        entries = list_skills()
        return {
            "skills": [
                {
                    "name": e.skill.name,
                    "description": e.skill.description[:200],
                    "emoji": e.skill.raw.get("metadata", {}).get("openclaw", {}).get("emoji")
                    if isinstance(e.skill.raw.get("metadata"), dict)
                    else None,
                }
                for e in entries
            ],
            "count": len(entries),
        }
    except ImportError:
        return {"skills": [], "count": 0}


async def _handle_xarm(args: Dict[str, Any]) -> Dict[str, Any]:
    """Forward xArm commands to NeuroLinux Agent on port 8085.

    Uses direct REST endpoints (/arm/home, /arm/wave, etc.) for reliability.
    Falls back to /agent/chat for unknown commands.
    """
    import httpx

    command = args.get("command", "status")

    # Map command names to direct REST endpoints on the agent (port 8085).
    # NOTE: /arm/named/* stubs do NOT exist for pick_table — use group_move or
    # cookoff/pick instead. Kept here for simple non-pick gestures only.
    DIRECT_ENDPOINTS: Dict[str, str] = {
        "home":          "/arm/home",
        "wave":          "/arm/wave",
        "inspect":       "/arm/inspect",
        "reach":         "/arm/reach",
        "reach_forward": "/arm/reach",
        "ready":         "/arm/ready",
        "stop":          "/arm/stop",
        "open_gripper":  "/arm/gripper/open",
        "close_gripper": "/arm/gripper/close",
        "open":          "/arm/gripper/open",
        "close":         "/arm/gripper/close",
        # "pick" / "place" are NOT simple named poses — they must go through
        # the cookoff IK pipeline via /cookoff/pick (NIS port 8000).
        "pick":          "__ik_pick__",
        "place":         "__ik_pick__",
    }

    agent_base = "http://localhost:8085"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if command == "status":
                r = await client.get(f"{agent_base}/health")
                if r.status_code == 200:
                    d = r.json()
                    return {
                        "command": "status",
                        "result": {
                            "ok": True,
                            "connected": not d.get("xarm_simulation", True),
                            "simulation": d.get("xarm_simulation", True),
                            "port": d.get("xarm_port", "?"),
                            "result": f"xArm {'PHYSICAL' if not d.get('xarm_simulation') else 'SIM'} on {d.get('xarm_port','?')}",
                        },
                    }

            endpoint = DIRECT_ENDPOINTS.get(command)
            if endpoint == "__ik_pick__":
                # Route pick/place through NIS cookoff IK pipeline (port 8000)
                try:
                    import httpx as _hx
                    async with _hx.AsyncClient(timeout=60.0) as nis:
                        nr = await nis.post(
                            "http://localhost:8000/cookoff/pick",
                            json={"zone": args.get("zone", "left90"), "dry_run": False},
                        )
                        nd = nr.json() if nr.status_code == 200 else {"error": f"NIS HTTP {nr.status_code}"}
                        return {"command": command, "result": {**nd, "ok": nd.get("ok", nr.status_code == 200)}}
                except Exception as pe:
                    return {"command": command, "result": {"error": str(pe), "ok": False}}

            if endpoint:
                r = await client.post(f"{agent_base}{endpoint}", json={})
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "command": command,
                        "result": {
                            "ok":         data.get("ok", True),
                            "result":     f"{command} executed {'[PHYSICAL]' if not data.get('simulation') else '[SIM]'}",
                            "simulation": data.get("simulation", True),
                        },
                    }
                return {"command": command, "result": {"error": f"Agent HTTP {r.status_code}", "ok": False}}

            # Unknown command — fall back to /agent/chat
            r = await client.post(f"{agent_base}/agent/chat", json={"message": command})
            if r.status_code == 200:
                data = r.json()
                return {
                    "command": command,
                    "result": {
                        "ok":     True,
                        "result": data.get("response", f"{command} sent"),
                    },
                }
            return {"command": command, "result": {"error": f"Agent HTTP {r.status_code}", "ok": False}}

    except Exception as exc:
        logger.warning("Agent unreachable for xarm command '%s': %s", command, exc)
        return {"command": command, "result": {"error": str(exc), "ok": False}}


# ──────────────────────────────────────────────────────────
#  Docker host resolution helper
# ──────────────────────────────────────────────────────────

def _resolve_host(localhost_url: str) -> str:
    """Replace 'localhost'/'127.0.0.1' with 'host.docker.internal' when running
    inside a Docker container so the container can reach Windows host services."""
    if os.path.exists("/.dockerenv"):
        return localhost_url.replace("localhost", "host.docker.internal") \
                            .replace("127.0.0.1", "host.docker.internal")
    return localhost_url


async def _http_get(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(_resolve_host(url))
        r.raise_for_status()
        return r.json()


async def _http_post(url: str, body: Dict[str, Any] = None, timeout: float = 10.0) -> Dict[str, Any]:
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(_resolve_host(url), json=body or {})
        r.raise_for_status()
        return r.json()


# ──────────────────────────────────────────────────────────
#  Trading & stack tool handlers
# ──────────────────────────────────────────────────────────

# Pi NIS Protocol — used for NeuroKernel + OpenFang (Pi-only services).
# When running in Docker on Windows, default to Pi IP.  On Pi set PI_NIS_URL=http://localhost:8000.
_PI_NIS_BASE  = os.getenv("PI_NIS_URL", "http://192.168.1.163:8000")


async def _handle_stack(args):
    """Pi-native service health snapshot."""
    import httpx, asyncio as _aio
    async def ping(url):
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(url)
                return r.status_code < 500
        except Exception:
            return False
    xarm_ok, yolo_ok, nk_ok, of_ok = await _aio.gather(
        ping("http://localhost:8085/health"),
        ping("http://localhost:8000/yolo/status"),
        ping("http://localhost:8000/neurokernel/health"),
        ping("http://localhost:8000/openfang/status"),
    )
    return {
        "stack": {
            "xarm_agent :8085": bool(xarm_ok),
            "yolo :8000": bool(yolo_ok),
            "neurokernel :8000": bool(nk_ok),
            "openfang :8000": bool(of_ok),
        },
        "node": "pi-edge",
        "timestamp": time.time(),
    }

async def _dispatch_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Internal tool router used by the agent loop."""
    t = tool.lower().replace("nis_", "")
    if t == "chat":
        return await _handle_chat(args)
    if t in ("cosmos_plan", "cosmos", "robot_plan"):
        return await _handle_cosmos_plan(args)
    if t == "skills":
        return _handle_skills(args)
    if t in ("xarm", "arm"):
        return await _handle_xarm(args)
    if t in ("stack", "health", "services"):
        return await _handle_stack(args)
    if t in ("neurokernel", "kernel"):
        return await _handle_neurokernel(args)
    if t in ("openfang", "mcp"):
        return await _handle_openfang(args)
    raise ValueError(f"Unknown tool: {tool!r}")


async def _handle_agent(args: Dict[str, Any]) -> Dict[str, Any]:
    """ReAct-style agentic loop.

    The LLM iteratively decides which NIS tools to call until it can give
    a final answer.  Trace is returned so the caller can show step-by-step
    reasoning.

    Args:
      goal / message / query  — natural language task description
      max_steps               — maximum tool calls before giving up (default 5, max 8)
    """
    goal = (args.get("goal") or args.get("message") or args.get("query") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal, message, or query required")

    max_steps = min(int(args.get("max_steps", 5)), 8)

    TOOL_DESCRIPTIONS = (
        "nis_yolo: Scan the robot table with YOLO object detection. Args: {}\n"
        "nis_xarm: Control xArm robot. Args: {command: 'status'|'home'|'wave'|'open'|'close'|'pick'|'place'|'inspect'}\n"
        "nis_neurokernel: NeuroKernel v2. Args: {action: 'status'|'process'|'drives'|'skills', text?: str}\n"
        "nis_stack: Full Organica stack health. Args: {}\n"
        "nis_cosmos_plan: Robot motion plan via Cosmos pipeline. Args: {query: str}\n"
        "nis_openfang: OpenFang MCP tools (arm/camera/cosmos). Args: {action: 'tools'|'call'|'status', tool_name?: str, tool_args?: object}\n"
    )

    system_prompt = (
        "You are an autonomous AI agent for Organica AI Solutions controlling a Raspberry Pi robot stack. "
        "Complete the user's goal using these tools:\n\n"
        f"{TOOL_DESCRIPTIONS}\n"
        "Rules:\n"
        "- To call a tool output ONLY: TOOL_CALL: {\"tool\": \"<name>\", \"args\": {...}}\n"
        "- To give the final answer output ONLY: FINAL: <your concise answer>\n"
        f"- Use the minimum tools needed. Max {max_steps} tool calls.\n"
        "- After each tool result, either call another tool or give FINAL: <answer>."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]

    llm = get_llm_provider()
    if not llm:
        return {
            "goal": goal,
            "answer": "LLM provider not available — cannot run agent loop.",
            "steps": 0,
            "trace": [],
            "error": "no_llm",
        }

    trace: List[Dict[str, Any]] = []

    for step in range(max_steps):
        try:
            llm_result = await llm.generate_response(messages=messages)
        except Exception as exc:
            return {
                "goal": goal,
                "answer": f"LLM error at step {step}: {exc}",
                "steps": step,
                "trace": trace,
                "error": str(exc),
            }

        response_text = (
            llm_result.get("response") or llm_result.get("content") or llm_result.get("text") or ""
        ).strip()

        # ── Final answer ───────────────────────────────────────────────────────
        if response_text.upper().startswith("FINAL:"):
            return {
                "goal": goal,
                "answer": response_text[6:].strip(),
                "steps": step,
                "trace": trace,
                "model": llm_result.get("model", "unknown"),
            }

        # ── Tool call ──────────────────────────────────────────────────────────
        if "TOOL_CALL:" in response_text:
            try:
                json_part = response_text.split("TOOL_CALL:", 1)[1].strip()
                call = json.loads(json_part)
                tool_name: str = call["tool"]
                tool_args: Dict[str, Any] = call.get("args", {})
            except Exception as parse_err:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"JSON parse error: {parse_err}. "
                        "Please output exactly: TOOL_CALL: {\"tool\": \"...\", \"args\": {...}}"
                    ),
                })
                continue

            try:
                tool_result = await _dispatch_tool(tool_name, tool_args)
                result_str = json.dumps(tool_result)[:800]
            except Exception as tool_err:
                tool_result = {"error": str(tool_err)}
                result_str = str(tool_err)[:300]

            trace.append({
                "step": step + 1,
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result_str[:300],
            })

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}:\n{result_str}\n\n"
                    "Continue: call another tool or give FINAL: <answer>"
                ),
            })
        else:
            # LLM answered directly without proper prefix — treat as final answer
            return {
                "goal": goal,
                "answer": response_text,
                "steps": step,
                "trace": trace,
            }

    # Max steps exhausted
    last_preview = trace[-1]["result_preview"] if trace else ""
    return {
        "goal": goal,
        "answer": f"Reached max steps ({max_steps}). Last result: {last_preview[:200]}",
        "steps": max_steps,
        "trace": trace,
        "truncated": True,
    }


async def _handle_neurokernel(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NeuroKernel v2 via Pi NIS Protocol routes.

    Actions:
      status   — Full kernel status (DIKW layers, drives, components)
      process  — Run text through NeuroKernel pipeline
      drives   — List autonomous drives + states
      skills   — List loaded SKILL.md entries
      scan     — Injection threat scan on text
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _PI_NIS_BASE

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:

            if action == "status":
                r = await c.get(f"{base}/neurokernel/status")
                r.raise_for_status()
                return r.json()

            if action == "drives":
                r = await c.get(f"{base}/neurokernel/drives")
                r.raise_for_status()
                return r.json()

            if action == "skills":
                r = await c.get(f"{base}/neurokernel/skills")
                r.raise_for_status()
                return r.json()

            if action == "process":
                text = args.get("text") or args.get("input") or args.get("message", "")
                if not text:
                    raise HTTPException(status_code=400, detail="text required for process action")
                body = {
                    "agent_id": args.get("agent_id", "openclaw"),
                    "layer": args.get("layer", "reasoning"),
                    "action_type": args.get("action_type", "llm_call"),
                    "user_input": text,
                    "context_id": args.get("context_id"),
                    "skip_scan": args.get("skip_scan", False),
                }
                r = await c.post(f"{base}/neurokernel/process", json={k: v for k, v in body.items() if v is not None})
                r.raise_for_status()
                return r.json()

            if action == "scan":
                text = args.get("text", "")
                r = await c.post(f"{base}/neurokernel/scan", json={"text": text, "context": "openclaw"})
                r.raise_for_status()
                return r.json()

            # Default: health
            r = await c.get(f"{base}/neurokernel/health")
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "NeuroKernel", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("neurokernel handler failed: %s", exc)
        return {"service": "NeuroKernel", "error": str(exc), "online": False}


async def _handle_openfang(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to OpenFang MCP bridge via Pi NIS Protocol routes.

    Actions:
      tools    — List all NIS tools exposed through OpenFang
      call     — Execute a named NIS tool (requires tool_name + tool_args)
      agents   — List active NIS agents
      status   — OpenFang system status
    """
    import httpx

    action = args.get("action", "tools").lower()
    base = _PI_NIS_BASE

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:

            if action == "tools":
                r = await c.get(f"{base}/openfang/tools")
                r.raise_for_status()
                return r.json()

            if action == "call":
                tool_name = args.get("tool_name") or args.get("tool")
                if not tool_name:
                    raise HTTPException(status_code=400, detail="tool_name required for call action")
                tool_args = args.get("tool_args") or args.get("args") or {}
                r = await c.post(f"{base}/openfang/tools/call",
                                 json={"name": tool_name, "arguments": tool_args})
                r.raise_for_status()
                return r.json()

            if action == "agents":
                r = await c.get(f"{base}/openfang/agents")
                r.raise_for_status()
                return r.json()

            # Default: status
            r = await c.get(f"{base}/openfang/status")
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "OpenFang", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("openfang handler failed: %s", exc)
        return {"service": "OpenFang", "error": str(exc), "online": False}


@router.post("/invoke", summary="Invoke NIS Protocol from OpenClaw")
async def openclaw_invoke(request: InvokeRequest, _: str = Security(_require_openclaw_key)) -> Dict[str, Any]:
    """
    Bridge endpoint for OpenClaw gateway.
    Maps OpenClaw tool calls to real NIS Protocol capabilities.
    """
    tool = request.tool.lower().strip()
    args = request.args or {}

    if tool in ("nis_chat", "chat"):
        result = await _handle_chat(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_cosmos_plan", "cosmos_plan", "robot_plan"):
        result = await _handle_cosmos_plan(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_skills", "skills_list"):
        result = _handle_skills(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_xarm", "xarm"):
        result = await _handle_xarm(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_alphacortex", "alphacortex", "alpha", "trading"):
        result = await _handle_alphacortex(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_arbitrage", "arbitrage", "arb", "crypto"):
        result = await _handle_arbitrage(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_stack", "stack", "health", "services"):
        result = await _handle_stack(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_agent", "agent", "agentic", "run"):
        result = await _handle_agent(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_neurokernel", "neurokernel", "kernel"):
        result = await _handle_neurokernel(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_openfang", "openfang", "mcp"):
        result = await _handle_openfang(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_organica", "organica", "agent_framework"):
        result = await _handle_organica(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_orion", "orion", "coding"):
        result = await _handle_orion(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_portfolio", "portfolio", "smartportfolio"):
        result = await _handle_portfolio(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_mexpay", "mexpay", "spei", "fintech"):
        result = await _handle_mexpay(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_hub", "hub", "orchestration", "fleet"):
        result = await _handle_hub(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_moe", "moe", "embeddings", "semantic"):
        result = await _handle_moe(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_auto", "auto", "automotive", "vehicle"):
        result = await _handle_auto(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_organica_web", "organica_web", "orgweb"):
        result = await _handle_organica_web(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_cryptobot", "cryptobot", "crypto_bot", "trading_bot"):
        result = await _handle_cryptobot(args)
        return {"status": "ok", "tool": tool, "result": result}

    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown tool: '{tool}'. "
            "Supported: nis_chat, nis_cosmos_plan, nis_skills, nis_xarm, "
            "nis_alphacortex, nis_arbitrage, nis_stack, nis_neurokernel, nis_openfang, "
            "nis_organica, nis_orion, nis_portfolio, nis_mexpay, nis_hub, nis_moe, nis_auto, nis_agent"
        ),
    )


@router.get("/tools", summary="List NIS Protocol tools for OpenClaw")
async def openclaw_tools() -> Dict[str, Any]:
    """
    Return tool definitions for OpenClaw agent config.
    Add this URL as a 'nis-protocol' agent in your .openclaw/config.yaml.
    """
    return {
        "tools": [
            {
                "name": "nis_chat",
                "description": "Send a message to NIS Protocol LLM agent (GPT-4, Claude, Gemini, etc.)",
                "args": {
                    "message": "string — the user's message",
                },
            },
            {
                "name": "nis_cosmos_plan",
                "description": "Get a robot action plan using NVIDIA Cosmos Cookoff pipeline",
                "args": {
                    "query": "string — task description, e.g. 'Pick up the red cube'",
                    "robot_state": "object? — current robot joint/gripper state",
                    "image_base64": "string? — base64 encoded scene image",
                },
            },
            {
                "name": "nis_skills",
                "description": "List available OpenClaw-compatible skills registered in NIS Protocol",
                "args": {},
            },
            {
                "name": "nis_xarm",
                "description": "Control Hiwonder xArm 1.6 connected to NeuroLinux",
                "args": {
                    "command": "string — status | home | open | close | move",
                    "port": "string? — serial port, default /dev/ttyUSB0",
                    "model": "string? — 1s or 1.6 (default)",
                },
            },
            {
                "name": "nis_alphacortex",
                "description": "Query AlphaCortex US equity trading service: positions, account, analysis, orders, scheduler status",
                "args": {
                    "action": "string — status | positions | account | orders | analyze | scheduler (default: status)",
                    "symbol": "string? — required for analyze action (e.g. NVDA)",
                },
            },
            {
                "name": "nis_arbitrage",
                "description": "Query ArbitrageMachine crypto arbitrage service: AI agent status, opportunities, metrics, recent decisions",
                "args": {
                    "action": "string — status | opportunities | metrics | decisions (default: status)",
                },
            },
            {
                "name": "nis_stack",
                "description": "Get full Organica AI stack health — all services online/offline at a glance",
                "args": {},
            },
            {
                "name": "nis_neurokernel",
                "description": "Query/control NeuroKernel v2 on the Pi: DIKW layers, drives, skills, injection scan, full pipeline processing",
                "args": {
                    "action": "string — status | process | drives | skills | scan (default: status)",
                    "text": "string? — required for process and scan actions",
                    "layer": "string? — DIKW layer for process: data|information|knowledge|wisdom|reasoning (default: reasoning)",
                    "agent_id": "string? — agent ID for process action (default: openclaw)",
                },
            },
            {
                "name": "nis_openfang",
                "description": "Execute NIS tools via OpenFang MCP interface on the Pi: arm moves, camera, Cosmos reasoning, system status",
                "args": {
                    "action": "string — tools | call | agents | status (default: tools)",
                    "tool_name": "string? — required for call action (e.g. arm_move_home, camera_snapshot, cosmos_reason)",
                    "tool_args": "object? — arguments passed to the tool",
                },
            },
            {
                "name": "nis_agent",
                "description": (
                    "Autonomous agentic loop: describe a goal in natural language and NIS Protocol "
                    "will iteratively call the right tools (YOLO, xArm, NeuroKernel, trading, etc.) "
                    "until the task is complete. Returns answer + step-by-step trace."
                ),
                "args": {
                    "goal": "string — natural language goal (e.g. 'pick up the lighter', 'show me my trading positions')",
                    "max_steps": "integer? — max tool calls before giving up (default 5, max 8)",
                },
            },
            {
                "name": "nis_organica",
                "description": (
                    "Route to any of the 39 specialized Organica Framework agents: engineering, design, "
                    "marketing, ops, research, legal, product, and more. List agents or call a specific one by name."
                ),
                "args": {
                    "action": "string — route | list | call | health (default: route)",
                    "message": "string? — message to send (required for route and call)",
                    "agent_id": "string? — specific agent name/ID to call (required for call action)",
                },
            },
            {
                "name": "nis_orion",
                "description": (
                    "Orion TypeScript/JavaScript coding AI agent (:8080). Ask code, build, debug, "
                    "architecture, or refactoring questions. Best for anything code-related."
                ),
                "args": {
                    "message": "string — your coding question or task",
                    "action": "string? — chat | health (default: chat)",
                },
            },
            {
                "name": "nis_portfolio",
                "description": (
                    "Query SmartPortfolio service (:8002): portfolio optimization engine, "
                    "account summary, equity/cash balance."
                ),
                "args": {
                    "action": "string — status | account | health (default: status)",
                },
            },
            {
                "name": "nis_mexpay",
                "description": (
                    "MEXPAY20022 Mexican fintech service (:3000): SPEI real-time payments, "
                    "ISO 20022 messaging, CLABE validation, transfer history, participating banks."
                ),
                "args": {
                    "action": "string — status | transfer | history | banks | validate | info | health (default: status)",
                    "from_clabe": "string? — source CLABE (18 digits) — required for transfer",
                    "to_clabe": "string? — destination CLABE (18 digits) — required for transfer",
                    "amount": "number? — transfer amount in MXN — required for transfer",
                    "concept": "string? — payment concept/description — required for transfer",
                    "clabe": "string? — CLABE to validate — required for validate",
                },
            },
            {
                "name": "nis_hub",
                "description": (
                    "NIS-HUB central orchestration service (:8003): node registration and status, "
                    "fleet device management, mission coordination, inter-agent routing."
                ),
                "args": {
                    "action": "string — status | nodes | fleet | fleet_devices | missions | health (default: status)",
                },
            },
            {
                "name": "nis_moe",
                "description": (
                    "NIS Mixture-of-Experts semantic embedding model (:8004): "
                    "embed text into high-dimensional vectors, compute cosine similarity, batch embedding."
                ),
                "args": {
                    "action": "string — embed | similarity | batch | info | health (default: health)",
                    "text": "string? — text to embed (required for embed)",
                    "text1": "string? — first text for similarity (required for similarity)",
                    "text2": "string? — second text for similarity (required for similarity)",
                    "texts": "list? — list of texts for batch embedding (required for batch)",
                },
            },
            {
                "name": "nis_auto",
                "description": (
                    "NIS-AUTO automotive AGI agent (:8005): cognitive reasoning for vehicle systems, "
                    "OBD-II integration, chat interface, consciousness status."
                ),
                "args": {
                    "action": "string — status | chat | consciousness | agents | process | health (default: status)",
                    "message": "string? — message for chat action",
                    "text": "string? — text for process action",
                },
            },
            {
                "name": "nis_organica_web",
                "description": (
                    "OrganicaAI website backend (:5001): Gemini Pro chat API with user auth. "
                    "Login to get a JWT token, then chat with Gemini."
                ),
                "args": {
                    "action": "string — health | login | chat (default: health)",
                    "email": "string? — user email (required for login)",
                    "password": "string? — user password (required for login)",
                    "token": "string? — JWT token (required for chat)",
                    "message": "string? — chat message (required for chat)",
                },
            },
            {
                "name": "nis_cryptobot",
                "description": (
                    "CryptoBot Alpaca crypto trading service (:5002): strategy management, "
                    "positions, trades, market data, backtesting. Supertrend + MACD strategies."
                ),
                "args": {
                    "action": "string — status | health | account | positions | trades | market | strategies | start | stop | backtest (default: status)",
                    "symbol": "string? — trading pair e.g. BTC/USD, ETH/USD (for market and backtest)",
                    "strategy": "string? — strategy type for backtest: supertrend | macd",
                    "start_date": "string? — backtest start date YYYY-MM-DD",
                    "end_date": "string? — backtest end date YYYY-MM-DD",
                },
            },
        ],
    }


@router.get("/status", summary="OpenClaw bridge health check")
async def openclaw_status() -> Dict[str, Any]:
    """Check which NIS Protocol capabilities are available for OpenClaw."""
    # LLM
    llm_ok = get_llm_provider() is not None
    # Cosmos stack
    try:
        from src.agents.cosmos import get_cosmos_reasoner  # noqa
        cosmos_ok = True
    except ImportError:
        cosmos_ok = False
    # xArm — check agent reachability via sync httpx (safe inside async context)
    try:
        import httpx
        r = httpx.get("http://localhost:8085/health", timeout=2.0)
        xarm_ok = r.status_code == 200
    except Exception:
        xarm_ok = False
    # Skills
    try:
        from src.skills.openclaw_skill_loader import list_skills
        skills_count = len(list_skills())
    except ImportError:
        skills_count = 0

    # Trading services — async-safe quick pings
    import asyncio
    async def _ping(url: str) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=1.5) as c:
                r = await c.get(_resolve_host(url))
                return r.status_code < 500
        except Exception:
            return False

    (
        nk_ok, of_ok,
    ) = await asyncio.gather(
        _ping("http://localhost:8000/neurokernel/health"),
        _ping("http://localhost:8000/openfang/status"),
        return_exceptions=False,
    )

    return {
        "status": "ok",
        "node": "pi-edge",
        "capabilities": {
            "nis_chat": llm_ok,
            "nis_cosmos_plan": cosmos_ok,
            "nis_xarm": xarm_ok,
            "nis_skills": skills_count > 0,
            "nis_neurokernel": bool(nk_ok),
            "nis_openfang": bool(of_ok),
            "nis_stack": True,
            "nis_agent": llm_ok,
        },
        "services": {
            "xarm_agent": xarm_ok,
            "neurokernel": bool(nk_ok),
            "openfang": bool(of_ok),
        },
        "skills_loaded": skills_count,
        "bridge_version": "1.7-pi",
    }
