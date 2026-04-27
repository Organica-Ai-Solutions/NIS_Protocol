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
  nis_alphacortex   — Query AlphaCortex: positions, account, analysis, orders
  nis_arbitrage     — Query ArbitrageMachine: status, opportunities, metrics
  nis_stack         — Full Organica AI stack health snapshot
  nis_neurokernel   — Query/control NeuroKernel v2 (DIKW layers, drives, agents)
  nis_openfang      — Execute NIS tools via OpenFang MCP interface
  nis_organica      — Route to any of the 39 Organica Framework specialized agents
  nis_orion         — Send a code/build/debug question to Orion coding AI (:8080)
  nis_portfolio     — Query SmartPortfolio: account, optimization status, summary
  nis_mexpay        — Mexican fintech: SPEI real-time payments, ISO 20022 messaging (:3010)
  nis_hub           — NIS-HUB: central orchestration, node management, fleet, missions (:8003)
  nis_moe           — NIS MoE: semantic embeddings and similarity search (:8004)
  nis_auto          — NIS-AUTO: automotive/OBD-II AGI agent (:8005)
  nis_organica_web  — OrganicaAI website backend: Gemini chat, user auth (:5001)
  nis_cryptobot     — CryptoBot: Alpaca crypto trading, strategies, backtesting (:5002)
  nis_s3            — AWS S3 browser: list, download, info for s3://nis-finetuning-bucket-penti-1753760384/
  nis_memory        — NIS persistent memory: store, retrieve, query conversation history and key-value pairs
  nis_reasoning     — Multi-agent collaborative reasoning or structured debate (/reasoning/*)
  nis_autonomy      — Autonomy Engine on the Pi: task submission, start/stop, goals, history (/autonomy/*)
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

_ALPHA_BASE   = os.getenv("ALPHACORTEX_URL", "http://localhost:5000")
_ARB_BASE     = os.getenv("ARBITRAGE_URL", "http://localhost:8000")
_SP_BASE      = os.getenv("SMARTPORTFOLIO_URL", "http://localhost:8002")
_ORG_BASE     = os.getenv("ORGANICA_URL", "http://localhost:8900")
_ORION_BASE   = os.getenv("ORION_URL", "http://localhost:8080")
_MEXPAY_BASE  = os.getenv("MEXPAY_URL", "http://localhost:3010")
_HUB_BASE     = os.getenv("NIS_HUB_URL", "http://localhost:8003")
_MOE_BASE     = os.getenv("NIS_MOE_URL", "http://localhost:8004")
_AUTO_BASE    = os.getenv("NIS_AUTO_URL", "http://localhost:8005")
_ORGWEB_BASE  = os.getenv("ORGANICA_WEB_URL", "http://localhost:5001")
_CRYPTOBOT_BASE = os.getenv("CRYPTOBOT_URL", "http://localhost:5002")
# Pi NIS Protocol — used for NeuroKernel + OpenFang (Pi-only services).
# When running in Docker on Windows, default to Pi IP.  On Pi set PI_NIS_URL=http://localhost:8000.
_PI_NIS_BASE  = os.getenv("PI_NIS_URL", "http://192.168.1.163:8000")


async def _handle_alphacortex(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to AlphaCortex trading service."""
    action = args.get("action", "status").lower()

    if action == "analyze":
        symbol = args.get("symbol", "")
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol required for analyze")
        result = await _http_post(f"{_ALPHA_BASE}/analyze", {"symbol": symbol.upper()}, timeout=15.0)
        return result

    if action == "positions":
        return await _http_get(f"{_ALPHA_BASE}/api/positions")

    if action == "account":
        return await _http_get(f"{_ALPHA_BASE}/api/account")

    if action == "orders":
        status = args.get("status", "open")
        return await _http_get(f"{_ALPHA_BASE}/api/orders?status={status}")

    if action == "scheduler":
        return await _http_get(f"{_ALPHA_BASE}/scheduler/status")

    if action == "start":
        symbols  = args.get("symbols")
        interval = args.get("interval_sec")
        body = {}
        if symbols:  body["symbols"]      = symbols
        if interval: body["interval_sec"] = interval
        return await _http_post(f"{_ALPHA_BASE}/scheduler/start", body, timeout=10.0)

    if action == "stop":
        return await _http_post(f"{_ALPHA_BASE}/scheduler/stop", {}, timeout=10.0)

    if action == "trade":
        symbol = args.get("symbol", "")
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol required for trade")
        body = {k: args[k] for k in ("symbol", "price", "rsi", "macd", "volatility") if k in args}
        body["symbol"] = symbol.upper()
        return await _http_post(f"{_ALPHA_BASE}/trade", body, timeout=20.0)

    if action == "close":
        symbol = args.get("symbol", "")
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol required for close")
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.delete(f"{_ALPHA_BASE}/api/positions/{symbol.upper()}")
            return r.json() if r.status_code < 300 else {"error": r.text, "status": r.status_code}

    # Default: health + account summary
    try:
        health = await _http_get(f"{_ALPHA_BASE}/health")
        acct = await _http_get(f"{_ALPHA_BASE}/api/account")
        positions = await _http_get(f"{_ALPHA_BASE}/api/positions")
        sched = await _http_get(f"{_ALPHA_BASE}/scheduler/status")
        return {
            "service": "AlphaCortex",
            "health": health,
            "account": acct,
            "positions": positions,
            "scheduler": sched.get("data", sched),
        }
    except Exception as exc:
        return {"service": "AlphaCortex", "error": str(exc), "online": False}


async def _handle_arbitrage(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to ArbitrageMachine crypto arbitrage service."""
    action = args.get("action", "status").lower()
    arb_key = os.getenv("INTERNAL_API_KEY", "")
    headers = {"X-API-Key": arb_key} if arb_key else {}

    import httpx
    async with httpx.AsyncClient(timeout=8.0) as c:
        base = _resolve_host(_ARB_BASE)

        if action == "opportunities":
            r = await c.get(f"{base}/api/opportunities", headers=headers)
            r.raise_for_status()
            return r.json()

        if action == "metrics":
            r = await c.get(f"{base}/api/metrics", headers=headers)
            r.raise_for_status()
            return r.json()

        if action == "decisions":
            r = await c.get(f"{base}/api/ai/decisions", headers=headers)
            r.raise_for_status()
            return r.json()

        # Default: status
        try:
            health_r = await c.get(f"{base}/api/health")
            ai_r = await c.get(f"{base}/api/ai/status", headers=headers)
            metrics_r = await c.get(f"{base}/api/metrics", headers=headers)
            return {
                "service": "ArbitrageMachine",
                "health": health_r.json() if health_r.status_code == 200 else {"error": health_r.status_code},
                "ai": ai_r.json() if ai_r.status_code == 200 else {"error": ai_r.status_code},
                "metrics": metrics_r.json() if metrics_r.status_code == 200 else {"error": metrics_r.status_code},
            }
        except Exception as exc:
            return {"service": "ArbitrageMachine", "error": str(exc), "online": False}


async def _handle_stack(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return full Organica AI stack health snapshot."""
    import httpx

    async def ping(url: str, path: str = "/health") -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(_resolve_host(url) + path)
                return r.status_code < 500
        except Exception:
            return False

    results = await __import__("asyncio").gather(
        ping(_ALPHA_BASE),
        ping(_ARB_BASE, "/api/health"),
        ping(_SP_BASE),
        ping(_ORG_BASE),
        ping(_ORION_BASE),
        ping(_HUB_BASE),
        ping(_MOE_BASE),
        ping(_AUTO_BASE),
        ping(_ORGWEB_BASE),
        ping(_CRYPTOBOT_BASE, "/api/health"),
        return_exceptions=True,
    )
    names = [
        "AlphaCortex :5000", "ArbitrageMachine :8000", "SmartPortfolio :8002",
        "Organica :8900", "Orion :8080",
        "NIS-HUB :8003", "NIS-MoE :8004", "NIS-AUTO :8005",
        "OrganicaWeb :5001", "CryptoBot :5002",
    ]
    return {
        "stack": {
            name: (bool(r) if not isinstance(r, Exception) else False)
            for name, r in zip(names, results)
        },
        "timestamp": time.time(),
    }


async def _handle_yolo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run YOLO object detection via Pi NIS :8000/yolo/detect."""
    import httpx
    base = _PI_NIS_BASE
    action = args.get("action", "detect").lower()
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            if action == "status":
                r = await c.get(f"{base}/yolo/status")
            else:
                r = await c.get(f"{base}/yolo/detect")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        return {"service": "YOLO", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("yolo handler failed: %s", exc)
        return {"service": "YOLO", "error": str(exc), "online": False}


async def _handle_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS Protocol memory system.

    Actions:
      query    — Search conversations by keyword/topic (default)
      store    — Store a key/value in persistent memory (requires key, value)
      retrieve — Retrieve a value by namespace + key
      stats    — Memory system stats (conversation count, topics, etc.)
      list     — List keys in a namespace
    """
    import httpx

    action = args.get("action", "query").lower()
    base = _resolve_host("http://localhost:8000")

    try:
        async with httpx.AsyncClient(timeout=8.0) as c:

            if action == "stats":
                r = await c.get(f"{base}/memory/stats")
                r.raise_for_status()
                return r.json()

            if action == "store":
                key = args.get("key")
                value = args.get("value")
                if not key or value is None:
                    raise HTTPException(status_code=400, detail="key and value required for store")
                namespace = args.get("namespace", "openclaw")
                r = await c.post(f"{base}/memory/store",
                                 json={"namespace": namespace, "key": key, "value": value, "ttl": args.get("ttl")})
                r.raise_for_status()
                return r.json()

            if action == "retrieve":
                key = args.get("key")
                if not key:
                    raise HTTPException(status_code=400, detail="key required for retrieve")
                namespace = args.get("namespace", "openclaw")
                r = await c.post(f"{base}/memory/retrieve", json={"namespace": namespace, "key": key})
                r.raise_for_status()
                return r.json()

            if action == "list":
                namespace = args.get("namespace", "openclaw")
                r = await c.get(f"{base}/memory/list/{namespace}")
                r.raise_for_status()
                return r.json()

            # Default: query conversations
            query = args.get("query") or args.get("message") or args.get("text", "")
            limit = int(args.get("limit", 5))
            conv_id = args.get("conversation_id")
            if conv_id:
                r = await c.get(f"{base}/memory/conversation/{conv_id}")
            else:
                r = await c.get(f"{base}/memory/conversations", params={"limit": limit})
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "NIS-Memory", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("memory handler failed: %s", exc)
        return {"service": "NIS-Memory", "error": str(exc), "online": False}


async def _handle_reasoning(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS Protocol multi-agent reasoning.

    Actions:
      collaborative — Multiple AI agents reason together (default)
      debate        — Two agents debate opposite sides of a topic
    """
    import httpx

    action = args.get("action", "collaborative").lower()
    base = _resolve_host("http://localhost:8000")
    topic = args.get("topic") or args.get("question") or args.get("message") or args.get("text", "")
    if not topic:
        raise HTTPException(status_code=400, detail="topic, question, or message required")

    rounds = int(args.get("rounds", 2))
    mode = args.get("mode", "collaborative")

    try:
        async with httpx.AsyncClient(timeout=45.0) as c:
            if action == "debate":
                r = await c.post(f"{base}/reasoning/debate",
                                 json={"topic": topic, "rounds": rounds})
            else:
                r = await c.post(f"{base}/reasoning/collaborative",
                                 json={"topic": topic, "mode": mode, "rounds": rounds})
            r.raise_for_status()
            data = r.json()
            return {
                "service": "NIS-Reasoning",
                "synthesis": data.get("synthesis") or data.get("conclusion") or data.get("response", ""),
                "reasoning_trace": data.get("reasoning_trace") or data.get("agents_reasoning", []),
                "model": data.get("model", "multi-agent"),
                "rounds": data.get("rounds", rounds),
            }

    except httpx.HTTPStatusError as exc:
        return {"service": "NIS-Reasoning", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("reasoning handler failed: %s", exc)
        return {"service": "NIS-Reasoning", "error": str(exc), "online": False}


async def _handle_autonomy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS Protocol Autonomy Engine on the Pi.

    Actions:
      status   — Full engine state: queue, recent actions, watchdog (default)
      start    — Start the autonomy engine
      stop     — Pause the autonomy engine
      task     — Inject a task into the queue (requires text)
      history  — Last N completed tasks
      watchdog — Watchdog status for all Pi services
      goal     — Set persistent goal (requires text); GET with action=goal to read
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _PI_NIS_BASE

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:

            if action == "start":
                r = await c.post(f"{base}/autonomy/start")
                r.raise_for_status()
                return r.json()

            if action == "stop":
                r = await c.post(f"{base}/autonomy/stop")
                r.raise_for_status()
                return r.json()

            if action == "task":
                text = args.get("text") or args.get("message") or args.get("task", "")
                if not text:
                    raise HTTPException(status_code=400, detail="text required for task action")
                priority = args.get("priority", "normal")
                r = await c.post(f"{base}/autonomy/task",
                                 json={"text": text, "priority": priority})
                r.raise_for_status()
                return r.json()

            if action == "history":
                limit = int(args.get("limit", 10))
                r = await c.get(f"{base}/autonomy/history", params={"limit": limit})
                r.raise_for_status()
                return r.json()

            if action == "watchdog":
                r = await c.get(f"{base}/autonomy/watchdog")
                r.raise_for_status()
                return r.json()

            if action == "goal":
                text = args.get("text") or args.get("goal")
                if text:
                    r = await c.post(f"{base}/autonomy/goal", json={"text": text})
                else:
                    r = await c.get(f"{base}/autonomy/status")
                r.raise_for_status()
                return r.json()

            # Default: status
            r = await c.get(f"{base}/autonomy/status")
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "AutonomyEngine", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("autonomy handler failed: %s", exc)
        return {"service": "AutonomyEngine", "error": str(exc), "online": False}


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
    if t in ("yolo", "vision", "detect", "scan"):
        return await _handle_yolo(args)
    if t in ("memory", "mem", "recall", "remember"):
        return await _handle_memory(args)
    if t in ("reasoning", "reason", "think", "debate"):
        return await _handle_reasoning(args)
    if t in ("autonomy", "auto_engine", "engine"):
        return await _handle_autonomy(args)
    if t in ("alphacortex", "alpha", "trading"):
        return await _handle_alphacortex(args)
    if t in ("arbitrage", "arb", "crypto"):
        return await _handle_arbitrage(args)
    if t in ("stack", "health", "services", "system"):
        return await _handle_stack(args)
    if t in ("neurokernel", "kernel"):
        return await _handle_neurokernel(args)
    if t in ("openfang", "mcp"):
        return await _handle_openfang(args)
    if t in ("organica", "agents", "agent_framework"):
        return await _handle_organica(args)
    if t in ("orion", "code", "coding"):
        return await _handle_orion(args)
    if t in ("portfolio", "smartportfolio", "sp"):
        return await _handle_portfolio(args)
    if t in ("mexpay", "spei", "iso20022", "payments", "fintech"):
        return await _handle_mexpay(args)
    if t in ("hub", "nis_hub", "orchestration", "fleet"):
        return await _handle_hub(args)
    if t in ("moe", "embed", "embeddings", "semantic"):
        return await _handle_moe(args)
    if t in ("auto", "automotive", "obd", "vehicle"):
        return await _handle_auto(args)
    if t in ("organica_web", "orgweb", "organicaweb", "gemini_chat"):
        return await _handle_organica_web(args)
    if t in ("cryptobot", "crypto_bot", "cryptotrading", "trading_bot"):
        return await _handle_cryptobot(args)
    if t in ("s3", "bucket", "aws_s3", "aws"):
        return await _handle_s3(args)
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
        "nis_alphacortex: US equity trading. Args: {action: 'status'|'positions'|'account'|'orders'|'analyze'|'scheduler'|'start'|'stop'|'trade'|'close', symbol?: str}\n"
        "nis_arbitrage: Crypto arbitrage. Args: {action: 'status'|'opportunities'|'metrics'}\n"
        "nis_stack: Full Organica stack health. Args: {}\n"
        "nis_cosmos_plan: Robot motion plan via Cosmos pipeline. Args: {query: str}\n"
        "nis_openfang: OpenFang MCP tools (arm/camera/cosmos). Args: {action: 'tools'|'call'|'status', tool_name?: str, tool_args?: object}\n"
        "nis_organica: Route to one of 39 specialized Organica agents. Args: {action: 'route'|'list'|'call', message?: str, agent_id?: str}\n"
        "nis_orion: Orion TypeScript coding AI. Args: {message: str} — use for code, build, debug questions\n"
        "nis_portfolio: SmartPortfolio service. Args: {action: 'status'|'account'|'health'|'analyze'|'rebalance', tickers: [...], allocations: {...}, risk_tolerance: 'medium', threshold: 0.05, paper: false}\n"
        "nis_mexpay: Mexican fintech — SPEI payments, ISO 20022, banks. Args: {action: 'status'|'transfer'|'history'|'banks'|'validate', from_clabe?: str, to_clabe?: str, amount?: number, concept?: str, clabe?: str}\n"
        "nis_hub: NIS-HUB orchestration — nodes, fleet, missions. Args: {action: 'status'|'nodes'|'fleet'|'missions'|'health'}\n"
        "nis_moe: NIS MoE semantic embeddings. Args: {action: 'embed'|'similarity'|'batch'|'info'|'health', text?: str, text1?: str, text2?: str, texts?: list}\n"
        "nis_auto: NIS-AUTO automotive AGI. Args: {action: 'status'|'chat'|'consciousness'|'agents'|'process', message?: str, text?: str}\n"
        "nis_organica_web: OrganicaAI Gemini chat. Args: {action: 'health'|'login'|'chat', email?: str, password?: str, token?: str, message?: str}\n"
        "nis_cryptobot: CryptoBot Alpaca trading. Args: {action: 'status'|'health'|'account'|'positions'|'trades'|'market'|'strategies'|'start'|'stop'|'backtest', symbol?: str}\n"
        "nis_s3: Browse/download from AWS S3 training bucket (models, checkpoints, LoRA weights). Args: {action: 'list'|'dirs'|'info'|'download', prefix?: str, s3_key?: str, local_path?: str}\n"
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


async def _handle_organica(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to Organica Agent Framework (:8900).

    Actions:
      list    — List all available agents with names/descriptions
      route   — Route a message to the best-fit agent (auto-select)
      call    — Send a message to a specific named agent (requires agent_id)
      health  — Framework health + agent count
    """
    import httpx

    action = args.get("action", "route").lower()
    base = _resolve_host(_ORG_BASE)

    try:
        async with httpx.AsyncClient(timeout=20.0) as c:

            if action in ("list", "agents"):
                r = await c.get(f"{base}/agents")
                r.raise_for_status()
                data = r.json()
                agents = data.get("agents", data) if isinstance(data, dict) else data
                return {"service": "OrganicaFramework", "agents": agents, "count": len(agents) if isinstance(agents, list) else "?"}

            if action in ("call", "agent"):
                agent_id = args.get("agent_id") or args.get("agent")
                message = args.get("message") or args.get("query") or args.get("text", "")
                if not agent_id:
                    raise HTTPException(status_code=400, detail="agent_id required for call action")
                if not message:
                    raise HTTPException(status_code=400, detail="message required for call action")
                r = await c.post(f"{base}/route",
                                 json={"message": message, "agent_id": agent_id})
                r.raise_for_status()
                return r.json()

            if action == "health":
                r = await c.get(f"{base}/health")
                r.raise_for_status()
                return r.json()

            # Default: route to best-fit agent
            message = args.get("message") or args.get("query") or args.get("text", "")
            if not message:
                raise HTTPException(status_code=400, detail="message required for route action")
            r = await c.post(f"{base}/route", json={"message": message})
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "OrganicaFramework", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("organica handler failed: %s", exc)
        return {"service": "OrganicaFramework", "error": str(exc), "online": False}


async def _handle_orion(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to Orion coding AI agent (:8080).

    Actions:
      chat   — Send a message and get a response (default)
      health — Check Orion is online
    """
    import httpx

    action = args.get("action", "chat").lower()
    base = _resolve_host(_ORION_BASE)

    try:
        async with httpx.AsyncClient(timeout=30.0) as c:

            if action == "health":
                r = await c.get(f"{base}/")
                return {"service": "Orion", "online": r.status_code < 500, "status": r.status_code}

            # Default: chat
            message = args.get("message") or args.get("query") or args.get("text") or args.get("question", "")
            if not message:
                raise HTTPException(status_code=400, detail="message or query required")
            r = await c.post(f"{base}/api/chat",
                             json={"message": message},
                             timeout=30.0)
            r.raise_for_status()
            data = r.json()
            response_text = (
                data.get("response") or data.get("message") or data.get("content") or str(data)
            )
            return {
                "service": "Orion",
                "response": response_text,
                "model": data.get("model", "unknown"),
            }

    except httpx.HTTPStatusError as exc:
        return {"service": "Orion", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("orion handler failed: %s", exc)
        return {"service": "Orion", "error": str(exc), "online": False}


async def _handle_portfolio(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to SmartPortfolio service (:8002).

    Actions:
      account    — Live account snapshot (equity, cash, daily P&L, mode)
      health     — Service health check
      status     — Full status (health + account)  [default]
      analyze    — Run Markowitz optimization on tickers, return allocations
                   args: tickers (list or comma-str), start_date, risk_tolerance
      rebalance  — Execute rebalance via Alpaca (uses env credentials + circuit breaker)
                   args: allocations {ticker: weight} OR tickers (auto-optimizes first)
                         threshold (default 0.05)
      optimize   — Alias for analyze
    """
    import httpx, asyncio

    action = args.get("action", "status").lower()
    base = _resolve_host(_SP_BASE)

    try:
        async with httpx.AsyncClient(timeout=30.0) as c:

            # ── account ────────────────────────────────────────────────────
            if action == "account":
                r = await c.get(f"{base}/internal/account")
                r.raise_for_status()
                return r.json()

            # ── health ─────────────────────────────────────────────────────
            if action == "health":
                r = await c.get(f"{base}/health")
                r.raise_for_status()
                return r.json()

            # ── analyze / optimize ─────────────────────────────────────────
            if action in ("analyze", "optimize"):
                tickers = args.get("tickers", [])
                if isinstance(tickers, str):
                    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
                if not tickers:
                    return {"error": "tickers required for analyze action"}
                payload = {
                    "tickers":        tickers,
                    "start_date":     args.get("start_date", "2023-01-01"),
                    "risk_tolerance": args.get("risk_tolerance", "medium"),
                }
                r = await c.post(f"{base}/internal/analyze", json=payload)
                r.raise_for_status()
                data = r.json()
                # Return a concise summary with the key allocations
                return {
                    "service":     "SmartPortfolio",
                    "action":      "analyze",
                    "tickers":     tickers,
                    "allocations": data.get("allocations", {}),
                    "sharpe":      data.get("performance", {}).get("sharpe_ratio"),
                    "volatility":  data.get("performance", {}).get("annual_volatility"),
                    "expected_return": data.get("performance", {}).get("expected_annual_return"),
                    "full":        data,
                }

            # ── rebalance ──────────────────────────────────────────────────
            if action == "rebalance":
                allocations = args.get("allocations", {})

                # If no allocations provided, auto-optimize from tickers first
                if not allocations:
                    tickers = args.get("tickers", [])
                    if isinstance(tickers, str):
                        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
                    if not tickers:
                        return {"error": "provide allocations dict or tickers list for rebalance"}
                    analyze_r = await c.post(f"{base}/internal/analyze", json={
                        "tickers":        tickers,
                        "start_date":     args.get("start_date", "2023-01-01"),
                        "risk_tolerance": args.get("risk_tolerance", "medium"),
                    })
                    analyze_r.raise_for_status()
                    allocations = analyze_r.json().get("allocations", {})
                    if not allocations:
                        return {"error": "optimization returned no allocations"}

                payload = {
                    "allocations":          allocations,
                    "rebalance_threshold":  args.get("threshold", 0.05),
                    "paper":                args.get("paper", False),
                }
                r = await c.post(f"{base}/internal/rebalance", json=payload)
                r.raise_for_status()
                result = r.json()
                trades = result.get("trades", [])
                return {
                    "service":    "SmartPortfolio",
                    "action":     "rebalance",
                    "allocations": allocations,
                    "trades_placed": len([t for t in trades if t.get("status") not in ("failed",)]),
                    "trades_failed": len([t for t in trades if t.get("status") == "failed"]),
                    "trades":     trades,
                    "account":    result.get("account", {}),
                }

            # ── crypto-balances ────────────────────────────────────────────
            if action in ("crypto", "crypto-balances", "crypto_balances"):
                r = await c.get(f"{base}/crypto/balances")
                r.raise_for_status()
                return r.json()

            # ── crypto-analyze ─────────────────────────────────────────────
            if action in ("crypto-analyze", "crypto_analyze"):
                symbols = args.get("symbols", args.get("tickers", []))
                if isinstance(symbols, str):
                    symbols = [s.strip() for s in symbols.split(",") if s.strip()]
                if not symbols:
                    return {"error": "symbols required for crypto-analyze"}
                r = await c.post(f"{base}/crypto/analyze", json={
                    "symbols":        symbols,
                    "risk_tolerance": args.get("risk_tolerance", "medium"),
                    "lookback_days":  args.get("lookback_days", 365),
                })
                r.raise_for_status()
                return r.json()

            # ── crypto-rebalance ───────────────────────────────────────────
            if action in ("crypto-rebalance", "crypto_rebalance"):
                payload = {
                    "allocations":    args.get("allocations"),
                    "symbols":        args.get("symbols", args.get("tickers")),
                    "risk_tolerance": args.get("risk_tolerance", "medium"),
                    "threshold":      args.get("threshold", 0.05),
                    "dry_run":        args.get("dry_run", True),
                }
                r = await c.post(f"{base}/crypto/rebalance", json=payload)
                r.raise_for_status()
                return r.json()

            # ── default: full status ───────────────────────────────────────
            health_r, acct_r = await asyncio.gather(
                c.get(f"{base}/health"),
                c.get(f"{base}/internal/account"),
                return_exceptions=True,
            )
            health = health_r.json() if not isinstance(health_r, Exception) and health_r.status_code == 200 else {"error": str(health_r)}
            acct   = acct_r.json()   if not isinstance(acct_r,   Exception) and acct_r.status_code   == 200 else {"error": str(acct_r)}
            return {
                "service":    "SmartPortfolio",
                "health":     health,
                "account":    acct,
                "equity":     acct.get("equity")       or "?",
                "cash":       acct.get("cash")         or "?",
                "daily_pnl":  acct.get("daily_pnl")   or "?",
                "mode":       acct.get("mode")         or "?",
            }

    except httpx.HTTPStatusError as exc:
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:
            pass
        return {"service": "SmartPortfolio", "error": f"HTTP {exc.response.status_code}: {body}", "online": False}
    except Exception as exc:
        logger.warning("portfolio handler failed: %s", exc)
        return {"service": "SmartPortfolio", "error": str(exc), "online": False}


async def _handle_mexpay(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to MEXPAY20022 Mexican fintech service (:3010).

    Actions:
      health    — Service health check
      info      — System info and capabilities
      transfer  — Execute a SPEI real-time payment (requires from_clabe, to_clabe, amount, concept)
      history   — SPEI transfer history
      banks     — List of participating banks
      validate  — Validate a CLABE account number (requires clabe)
      status    — Health + info summary (default)
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _resolve_host(_MEXPAY_BASE)

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:

            if action == "health":
                r = await c.get(f"{base}/api/health")
                r.raise_for_status()
                return r.json()

            if action == "info":
                r = await c.get(f"{base}/api/info")
                r.raise_for_status()
                return r.json()

            if action == "transfer":
                body = {
                    "fromClabe": args.get("from_clabe") or args.get("fromClabe", ""),
                    "toClabe": args.get("to_clabe") or args.get("toClabe", ""),
                    "amount": args.get("amount", 0),
                    "concept": args.get("concept") or args.get("description", ""),
                    "referenceNumber": args.get("reference") or args.get("referenceNumber", ""),
                }
                r = await c.post(f"{base}/api/spei/transfer", json=body)
                r.raise_for_status()
                return r.json()

            if action == "history":
                r = await c.get(f"{base}/api/spei/history")
                r.raise_for_status()
                return r.json()

            if action == "banks":
                r = await c.get(f"{base}/api/spei/banks")
                r.raise_for_status()
                return r.json()

            if action == "validate":
                clabe = args.get("clabe", "")
                r = await c.post(f"{base}/api/spei/validate-clabe", json={"clabe": clabe})
                r.raise_for_status()
                return r.json()

            # Default: status (health + info)
            import asyncio
            health_r, info_r = await asyncio.gather(
                c.get(f"{base}/api/health"),
                c.get(f"{base}/api/info"),
                return_exceptions=True,
            )
            health = health_r.json() if not isinstance(health_r, Exception) and health_r.status_code == 200 else {"error": str(health_r)}
            info = info_r.json() if not isinstance(info_r, Exception) and info_r.status_code == 200 else {"error": str(info_r)}
            return {"service": "MEXPAY20022", "health": health, "info": info}

    except httpx.HTTPStatusError as exc:
        return {"service": "MEXPAY20022", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("mexpay handler failed: %s", exc)
        return {"service": "MEXPAY20022", "error": str(exc), "online": False}


async def _handle_hub(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS-HUB central orchestration service (:8003).

    Actions:
      nodes     — List all registered NIS nodes and their status
      fleet     — Fleet management summary (devices, products)
      missions  — Active missions status
      health    — Hub health check
      status    — Nodes + fleet summary (default)
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _resolve_host(_HUB_BASE)

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:

            if action == "nodes":
                r = await c.get(f"{base}/nodes/status")
                r.raise_for_status()
                return r.json()

            if action == "fleet":
                r = await c.get(f"{base}/api/v1/fleet/summary")
                r.raise_for_status()
                return r.json()

            if action == "fleet_devices":
                r = await c.get(f"{base}/api/v1/fleet/devices")
                r.raise_for_status()
                return r.json()

            if action == "missions":
                r = await c.get(f"{base}/missions")
                r.raise_for_status()
                return r.json()

            if action == "health":
                r = await c.get(f"{base}/health")
                r.raise_for_status()
                return r.json()

            # Default: status (nodes + fleet)
            import asyncio
            nodes_r, fleet_r = await asyncio.gather(
                c.get(f"{base}/nodes/status"),
                c.get(f"{base}/api/v1/fleet/summary"),
                return_exceptions=True,
            )
            nodes = nodes_r.json() if not isinstance(nodes_r, Exception) and nodes_r.status_code == 200 else {"error": str(nodes_r)}
            fleet = fleet_r.json() if not isinstance(fleet_r, Exception) and fleet_r.status_code == 200 else {"error": str(fleet_r)}
            return {"service": "NIS-HUB", "nodes": nodes, "fleet": fleet}

    except httpx.HTTPStatusError as exc:
        return {"service": "NIS-HUB", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("hub handler failed: %s", exc)
        return {"service": "NIS-HUB", "error": str(exc), "online": False}


async def _handle_moe(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS Mixture-of-Experts semantic embedding model (:8004).

    Actions:
      embed      — Embed a single text into a semantic vector (requires text)
      batch      — Embed a list of texts (requires texts list)
      similarity — Compute cosine similarity between two texts (requires text1, text2)
      info       — Model info and dimensions
      health     — Service health check (default)
    """
    import httpx

    action = args.get("action", "health").lower()
    base = _resolve_host(_MOE_BASE)

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:

            if action == "embed":
                text = args.get("text") or args.get("input") or args.get("query", "")
                if not text:
                    raise HTTPException(status_code=400, detail="text required for embed action")
                r = await c.post(f"{base}/embed", json={"text": text})
                r.raise_for_status()
                data = r.json()
                # Truncate embedding for readability
                emb = data.get("embedding", [])
                return {
                    "service": "NIS-MoE",
                    "embedding_dims": len(emb),
                    "embedding_preview": emb[:8],
                    "shape": data.get("shape", []),
                    "model": data.get("model", "unknown"),
                }

            if action == "batch":
                texts = args.get("texts") or args.get("inputs", [])
                if not texts:
                    raise HTTPException(status_code=400, detail="texts list required for batch action")
                r = await c.post(f"{base}/embed_batch", json={"texts": texts})
                r.raise_for_status()
                data = r.json()
                return {
                    "service": "NIS-MoE",
                    "count": data.get("count", len(texts)),
                    "shape": data.get("shape", []),
                }

            if action == "similarity":
                text1 = args.get("text1") or args.get("a", "")
                text2 = args.get("text2") or args.get("b", "")
                if not text1 or not text2:
                    raise HTTPException(status_code=400, detail="text1 and text2 required for similarity action")
                r = await c.post(f"{base}/similarity", json={"text1": text1, "text2": text2})
                r.raise_for_status()
                data = r.json()
                return {"service": "NIS-MoE", "similarity": data.get("similarity", 0.0)}

            if action == "info":
                r = await c.get(f"{base}/info")
                r.raise_for_status()
                return r.json()

            # Default: health
            r = await c.get(f"{base}/health")
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as exc:
        return {"service": "NIS-MoE", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("moe handler failed: %s", exc)
        return {"service": "NIS-MoE", "error": str(exc), "online": False}


async def _handle_auto(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to NIS-AUTO automotive AGI agent (:8005).

    Actions:
      health        — Service health check
      chat          — Send a message to the AUTO cognitive agent
      consciousness — Consciousness/self-model status
      agents        — List active cognitive agents
      process       — Process text through cognitive pipeline
      status        — Health + consciousness status (default)
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _resolve_host(_AUTO_BASE)

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:

            if action == "health":
                r = await c.get(f"{base}/health")
                r.raise_for_status()
                return r.json()

            if action == "chat":
                message = args.get("message") or args.get("query") or args.get("text", "")
                if not message:
                    raise HTTPException(status_code=400, detail="message required for chat action")
                r = await c.post(f"{base}/chat", json={"message": message})
                r.raise_for_status()
                data = r.json()
                return {
                    "service": "NIS-AUTO",
                    "response": data.get("response") or data.get("content") or str(data),
                }

            if action == "consciousness":
                r = await c.get(f"{base}/consciousness/status")
                r.raise_for_status()
                return r.json()

            if action == "agents":
                r = await c.get(f"{base}/agents")
                r.raise_for_status()
                return r.json()

            if action == "process":
                text = args.get("text") or args.get("input") or args.get("message", "")
                if not text:
                    raise HTTPException(status_code=400, detail="text required for process action")
                r = await c.post(f"{base}/process", json={"input": text})
                r.raise_for_status()
                return r.json()

            # Default: status (health + consciousness)
            import asyncio
            health_r, cns_r = await asyncio.gather(
                c.get(f"{base}/health"),
                c.get(f"{base}/consciousness/status"),
                return_exceptions=True,
            )
            health = health_r.json() if not isinstance(health_r, Exception) and health_r.status_code == 200 else {"error": str(health_r)}
            cns = cns_r.json() if not isinstance(cns_r, Exception) and cns_r.status_code == 200 else {"error": str(cns_r)}
            return {"service": "NIS-AUTO", "health": health, "consciousness": cns}

    except httpx.HTTPStatusError as exc:
        return {"service": "NIS-AUTO", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("auto handler failed: %s", exc)
        return {"service": "NIS-AUTO", "error": str(exc), "online": False}


async def _handle_organica_web(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to OrganicaAI website backend (:5001) — Gemini chat + auth.

    Actions:
      chat    — Send a chat message (requires token or user/pass for auto-login)
      login   — Login and get a JWT token (requires email, password)
      health  — Service health check (default)
    """
    import httpx

    action = args.get("action", "health").lower()
    base = _resolve_host(_ORGWEB_BASE)

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:

            if action == "login":
                email = args.get("email", "")
                password = args.get("password", "")
                if not email or not password:
                    raise HTTPException(status_code=400, detail="email and password required for login")
                r = await c.post(f"{base}/api/login", json={"email": email, "password": password})
                r.raise_for_status()
                data = r.json()
                return {
                    "service": "OrganicaWeb",
                    "token": data.get("token", ""),
                    "user": data.get("user", {}),
                }

            if action == "chat":
                message = args.get("message") or args.get("query") or args.get("text", "")
                token = args.get("token", "")
                if not message:
                    raise HTTPException(status_code=400, detail="message required for chat action")
                if not token:
                    raise HTTPException(status_code=400, detail="token required for chat (use login action first)")
                r = await c.post(
                    f"{base}/api/chat",
                    json={"message": message},
                    headers={"Authorization": f"Bearer {token}"},
                )
                r.raise_for_status()
                data = r.json()
                return {
                    "service": "OrganicaWeb",
                    "response": data.get("response") or data.get("message") or str(data),
                    "model": data.get("model", "gemini-pro"),
                }

            # Default: health check (hit root endpoint)
            r = await c.get(f"{base}/")
            return {
                "service": "OrganicaWeb",
                "online": r.status_code < 500,
                "status_code": r.status_code,
                "note": "Gemini-backed chat API — use login action to get token, then chat",
            }

    except httpx.HTTPStatusError as exc:
        return {"service": "OrganicaWeb", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("organica_web handler failed: %s", exc)
        return {"service": "OrganicaWeb", "error": str(exc), "online": False}


async def _handle_cryptobot(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to CryptoBot Alpaca trading service (:5002).

    Actions:
      health       — Service health check
      account      — Account info (balance, buying power)
      positions    — Current open positions
      trades       — Recent trade history
      market       — Market data and prices
      strategies   — List active trading strategies
      start        — Start the trading engine
      stop         — Stop the trading engine
      backtest     — Backtest a strategy (requires symbol, strategy, start_date, end_date)
      status       — Trading engine status (default)
    """
    import httpx

    action = args.get("action", "status").lower()
    base = _resolve_host(_CRYPTOBOT_BASE)
    api_base = f"{base}/api"

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:

            if action == "health":
                r = await c.get(f"{api_base}/health")
                r.raise_for_status()
                return r.json()

            if action == "account":
                r = await c.get(f"{api_base}/account")
                r.raise_for_status()
                return r.json()

            if action == "positions":
                r = await c.get(f"{api_base}/positions")
                r.raise_for_status()
                return r.json()

            if action == "trades":
                r = await c.get(f"{api_base}/trades")
                r.raise_for_status()
                return r.json()

            if action == "market":
                symbol = args.get("symbol", "BTC/USD")
                r = await c.get(f"{api_base}/market", params={"symbol": symbol})
                r.raise_for_status()
                return r.json()

            if action == "strategies":
                r = await c.get(f"{api_base}/strategies")
                r.raise_for_status()
                return r.json()

            if action == "start":
                r = await c.post(f"{api_base}/trading/start")
                r.raise_for_status()
                return r.json()

            if action == "stop":
                r = await c.post(f"{api_base}/trading/stop")
                r.raise_for_status()
                return r.json()

            if action == "backtest":
                body = {
                    "symbol": args.get("symbol", "BTC/USD"),
                    "strategy": args.get("strategy", "supertrend"),
                    "start_date": args.get("start_date", ""),
                    "end_date": args.get("end_date", ""),
                }
                r = await c.post(f"{api_base}/backtest", json=body)
                r.raise_for_status()
                return r.json()

            # Default: status (health + account summary)
            import asyncio
            health_r, acct_r, pos_r = await asyncio.gather(
                c.get(f"{api_base}/health"),
                c.get(f"{api_base}/account"),
                c.get(f"{api_base}/positions"),
                return_exceptions=True,
            )
            health = health_r.json() if not isinstance(health_r, Exception) and health_r.status_code == 200 else {"error": str(health_r)}
            acct = acct_r.json() if not isinstance(acct_r, Exception) and acct_r.status_code == 200 else {"error": str(acct_r)}
            positions = pos_r.json() if not isinstance(pos_r, Exception) and pos_r.status_code == 200 else []
            return {
                "service": "CryptoBot",
                "health": health,
                "account": acct,
                "open_positions": positions,
            }

    except httpx.HTTPStatusError as exc:
        return {"service": "CryptoBot", "error": f"HTTP {exc.response.status_code}", "online": False}
    except Exception as exc:
        logger.warning("cryptobot handler failed: %s", exc)
        return {"service": "CryptoBot", "error": str(exc), "online": False}


async def _handle_s3(args: Dict[str, Any]) -> Dict[str, Any]:
    """Browse and retrieve files from the NIS S3 training bucket (nis-finetuning-bucket-penti-1753760384).

    Actions:
      list      — List objects at a prefix path (default: top-level)
      download  — Download a file from S3 to local disk (requires s3_key, local_path)
      info      — Bucket summary (first 1000 objects: count + total size)
      dirs      — List top-level directories (common prefixes)
    """
    _BUCKET = "nis-finetuning-bucket-penti-1753760384"
    action = args.get("action", "list").lower()

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
        s3 = boto3.client("s3")

        if action == "dirs":
            resp = s3.list_objects_v2(Bucket=_BUCKET, Delimiter="/")
            dirs = [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])]
            return {"bucket": _BUCKET, "top_level_dirs": dirs}

        if action == "list":
            prefix = args.get("prefix", args.get("path", ""))
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            resp = s3.list_objects_v2(Bucket=_BUCKET, Prefix=prefix, Delimiter="/", MaxKeys=200)
            dirs = [p["Prefix"].rstrip("/").split("/")[-1] for p in resp.get("CommonPrefixes", [])]
            files = [
                {"key": o["Key"].split("/")[-1], "size_kb": round(o["Size"] / 1024, 1), "modified": str(o["LastModified"])[:10]}
                for o in resp.get("Contents", [])
                if not o["Key"].endswith("/")
            ]
            return {
                "bucket": _BUCKET,
                "prefix": prefix or "(root)",
                "dirs": dirs,
                "files": files,
                "truncated": resp.get("IsTruncated", False),
            }

        if action == "info":
            paginator = s3.get_paginator("list_objects_v2")
            total_size = 0
            total_count = 0
            for page in paginator.paginate(Bucket=_BUCKET, PaginationConfig={"MaxItems": 5000}):
                for obj in page.get("Contents", []):
                    total_size += obj["Size"]
                    total_count += 1
            return {
                "bucket": _BUCKET,
                "objects_sampled": total_count,
                "total_size_gb": round(total_size / 1e9, 2),
            }

        if action == "download":
            s3_key = args.get("s3_key") or args.get("key", "")
            local_path = args.get("local_path") or args.get("dest", "")
            if not s3_key or not local_path:
                raise HTTPException(status_code=400, detail="s3_key and local_path required for download")
            import os
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            s3.download_file(Bucket=_BUCKET, Key=s3_key, Filename=local_path)
            size = os.path.getsize(local_path)
            return {"status": "ok", "s3_key": s3_key, "local_path": local_path, "size_mb": round(size / 1e6, 1)}

        return {"error": f"Unknown action: {action}", "supported": ["list", "dirs", "info", "download"]}

    except Exception as exc:
        logger.warning("s3 handler failed: %s", exc)
        return {"service": "NIS-S3", "error": str(exc)}


# ──────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────

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

    if tool in ("nis_yolo", "yolo", "vision", "detect"):
        result = await _handle_yolo(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_alphacortex", "alphacortex", "alpha", "trading"):
        result = await _handle_alphacortex(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_arbitrage", "arbitrage", "arb", "crypto"):
        result = await _handle_arbitrage(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_stack", "nis_system", "stack", "health", "services", "system"):
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

    if tool in ("nis_s3", "s3", "bucket", "aws_s3"):
        result = await _handle_s3(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_memory", "memory", "mem", "recall", "remember"):
        result = await _handle_memory(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_reasoning", "reasoning", "reason", "think", "debate"):
        result = await _handle_reasoning(args)
        return {"status": "ok", "tool": tool, "result": result}

    if tool in ("nis_autonomy", "autonomy", "auto_engine", "engine"):
        result = await _handle_autonomy(args)
        return {"status": "ok", "tool": tool, "result": result}

    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown tool: '{tool}'. "
            "Supported: nis_chat, nis_cosmos_plan, nis_skills, nis_xarm, nis_yolo, "
            "nis_alphacortex, nis_arbitrage, nis_stack, nis_system, nis_neurokernel, nis_openfang, "
            "nis_organica, nis_orion, nis_portfolio, nis_mexpay, nis_hub, nis_moe, nis_auto, "
            "nis_organica_web, nis_cryptobot, nis_s3, nis_agent, "
            "nis_memory, nis_reasoning, nis_autonomy"
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
                "name": "nis_yolo",
                "description": "YOLO object detection on the Pi camera — returns detected objects with bounding boxes and color labels",
                "args": {
                    "action": "string? — detect (default) | status",
                },
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
                    "Control SmartPortfolio (:8002): analyze tickers with Markowitz optimization, "
                    "execute live Alpaca rebalances, check account equity/cash/daily P&L. "
                    "Uses env-stored Alpaca credentials with $10 daily-loss circuit breaker."
                ),
                "args": {
                    "action": (
                        "string — status | account | health | analyze | optimize | rebalance | "
                        "crypto | crypto-analyze | crypto-rebalance "
                        "(default: status)"
                    ),
                    "tickers": (
                        "list|string? — ticker symbols, e.g. ['AAPL','MSFT','SPY'] or 'AAPL,MSFT,SPY' "
                        "— required for analyze/rebalance when no allocations given"
                    ),
                    "allocations": (
                        "object? — {ticker: weight} e.g. {\"AAPL\": 0.4, \"MSFT\": 0.6} "
                        "— for rebalance; omit to auto-optimize from tickers"
                    ),
                    "risk_tolerance": "string? — low | medium | high (default: medium)",
                    "threshold":      "number? — rebalance threshold fraction (default: 0.05 = 5%)",
                    "start_date":     "string? — history start for optimization e.g. '2023-01-01'",
                    "paper":          "bool? — paper=true for dry run (default: false = live)",
                },
            },
            {
                "name": "nis_mexpay",
                "description": (
                    "MEXPAY20022 Mexican fintech service (:3010): SPEI real-time payments, "
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
            {
                "name": "nis_s3",
                "description": (
                    "AWS S3 browser for the NIS training bucket (s3://nis-finetuning-bucket-penti-1753760384/). "
                    "Contains all NVIDIA grant models, checkpoints, LoRA weights, datasets (3.6 TiB). "
                    "List, browse, and download models for local deployment."
                ),
                "args": {
                    "action": "string — list | dirs | info | download (default: list)",
                    "prefix": "string? — S3 key prefix to list (e.g. 'h100-backup-latest/NIS-MoE')",
                    "s3_key": "string? — full S3 key to download (required for download)",
                    "local_path": "string? — local destination path (required for download)",
                },
            },
            {
                "name": "nis_memory",
                "description": (
                    "NIS Protocol persistent memory system: store, retrieve, and query conversation "
                    "history and key-value memory. Namespaced storage with semantic retrieval."
                ),
                "args": {
                    "action": "string — query | store | retrieve | stats | list (default: query)",
                    "key": "string? — memory key (required for store/retrieve)",
                    "value": "string? — value to store (required for store)",
                    "namespace": "string? — namespace for list action (default: global)",
                    "query": "string? — query text for semantic retrieval",
                },
            },
            {
                "name": "nis_reasoning",
                "description": (
                    "NIS Protocol multi-agent reasoning: collaborative reasoning chains or structured "
                    "debate between agents. Use for complex analysis or decisions requiring multiple perspectives."
                ),
                "args": {
                    "action": "string — collaborative | debate (default: collaborative)",
                    "topic": "string — topic or question to reason about (required)",
                    "mode": "string? — reasoning mode for collaborative (default: analytical)",
                    "rounds": "integer? — number of reasoning rounds (default: 3)",
                },
            },
            {
                "name": "nis_autonomy",
                "description": (
                    "NIS Protocol Autonomy Engine on the Pi: start/stop autonomous task execution, "
                    "submit tasks, check engine status, view history, manage goals and watchdog."
                ),
                "args": {
                    "action": "string — status | start | stop | task | history | watchdog | goal (default: status)",
                    "text": "string? — task description (required for task and goal actions)",
                    "priority": "integer? — task priority 1-10 (default: 5, for task action)",
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
        alpha_ok, arb_ok, sp_ok, nk_ok, of_ok, org_ok, orion_ok,
        mexpay_ok, hub_ok, moe_ok, auto_ok, orgweb_ok, cryptobot_ok, yolo_ok,
        memory_ok, autonomy_ok,
    ) = await asyncio.gather(
        _ping(f"{_ALPHA_BASE}/health"),
        _ping(f"{_ARB_BASE}/api/health"),
        _ping(f"{_SP_BASE}/health"),
        _ping(f"{_PI_NIS_BASE}/neurokernel/health"),
        _ping(f"{_PI_NIS_BASE}/openfang/status"),
        _ping(f"{_ORG_BASE}/health"),
        _ping(f"{_ORION_BASE}"),
        _ping(f"{_MEXPAY_BASE}/api/health"),
        _ping(f"{_HUB_BASE}/health"),
        _ping(f"{_MOE_BASE}/health"),
        _ping(f"{_AUTO_BASE}/health"),
        _ping(f"{_ORGWEB_BASE}/"),
        _ping(f"{_CRYPTOBOT_BASE}/api/health"),
        _ping(f"{_PI_NIS_BASE}/yolo/status"),
        _ping(f"{_resolve_host('http://localhost:8000')}/memory/stats"),
        _ping(f"{_PI_NIS_BASE}/autonomy/status"),
        return_exceptions=False,
    )

    return {
        "status": "ok",
        "capabilities": {
            "nis_chat": llm_ok,
            "nis_cosmos_plan": cosmos_ok,
            "nis_xarm": xarm_ok,
            "nis_skills": skills_count > 0,
            "nis_alphacortex": bool(alpha_ok),
            "nis_arbitrage": bool(arb_ok),
            "nis_yolo": bool(yolo_ok),
            "nis_stack": True,
            "nis_system": True,
            "nis_neurokernel": bool(nk_ok),
            "nis_openfang": bool(of_ok),
            "nis_organica": bool(org_ok),
            "nis_orion": bool(orion_ok),
            "nis_portfolio": bool(sp_ok),
            "nis_mexpay": bool(mexpay_ok),
            "nis_hub": bool(hub_ok),
            "nis_moe": bool(moe_ok),
            "nis_auto": bool(auto_ok),
            "nis_organica_web": bool(orgweb_ok),
            "nis_cryptobot": bool(cryptobot_ok),
            "nis_agent": llm_ok,
            "nis_memory": bool(memory_ok),
            "nis_reasoning": llm_ok,
            "nis_autonomy": bool(autonomy_ok),
        },
        "services": {
            "alphacortex": bool(alpha_ok),
            "arbitrage": bool(arb_ok),
            "smartportfolio": bool(sp_ok),
            "organica": bool(org_ok),
            "orion": bool(orion_ok),
            "yolo_pi": bool(yolo_ok),
            "neurokernel_pi": bool(nk_ok),
            "openfang_pi": bool(of_ok),
            "mexpay": bool(mexpay_ok),
            "nis_hub": bool(hub_ok),
            "nis_moe": bool(moe_ok),
            "nis_auto": bool(auto_ok),
            "organica_web": bool(orgweb_ok),
            "cryptobot": bool(cryptobot_ok),
            "memory_pi": bool(memory_ok),
            "autonomy_pi": bool(autonomy_ok),
        },
        "skills_loaded": skills_count,
        "bridge_version": "1.9",
    }
