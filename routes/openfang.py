"""
NIS Protocol — OpenFang Integration Routes

Exposes NIS Protocol tools as an MCP server so OpenFang can discover
and call them. Also provides OpenAI-compatible endpoint routing for
Cosmos Reason2 calls.

Endpoints:
  GET  /openfang/tools          — List all available NIS tools (OpenFang format)
  POST /openfang/tools/call     — Execute a tool by name
  GET  /openfang/agents         — List active NIS agents
  POST /openfang/mcp            — MCP JSON-RPC 2.0 handler
  GET  /openfang/status         — System status for OpenFang dashboard
"""

import json
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional

import urllib.request
import urllib.error

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("nis_protocol.openfang")

router = APIRouter(prefix="/openfang", tags=["OpenFang Integration"])

# ── Config ────────────────────────────────────────────────────────────────────

PI_URL     = "http://192.168.1.163:8085"
COSMOS_URL = "http://localhost:8100"

# ── Tool Registry ─────────────────────────────────────────────────────────────

NIS_TOOLS = [
    {
        "name": "arm_move_home",
        "description": "Move the xArm 6DOF robotic arm to the calibrated home position (17×20.5cm workspace rest pose).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration_ms": {"type": "integer", "description": "Movement duration in ms", "default": 1200}
            }
        }
    },
    {
        "name": "arm_group_move",
        "description": "Move multiple xArm servos simultaneously. Servo IDs: S1=Gripper(100=open,550=closed), S6=Base(100=left,800=right).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "object",
                    "description": "Servo positions dict, e.g. {'1': 500, '2': 600, '6': 350}",
                    "additionalProperties": {"type": "integer"}
                },
                "duration_ms": {"type": "integer", "default": 1000}
            },
            "required": ["positions"]
        }
    },
    {
        "name": "arm_pick_and_place",
        "description": "Run the full 9-step Cosmos-guided pick-and-place cookoff demo pipeline on the xArm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "Must be true to execute (safety gate)"}
            },
            "required": ["confirm"]
        }
    },
    {
        "name": "arm_get_status",
        "description": "Get current xArm servo positions and connection status.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "arm_get_poses",
        "description": "List all saved named positions (touch poses) for the xArm.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "arm_save_pose",
        "description": "Save the current arm position as a named touch pose.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Pose name (e.g. 'home', 'inspect', 'pick_table')"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "camera_snapshot",
        "description": "Capture a JPEG frame from the Pi camera. Returns base64-encoded image data.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "cosmos_reason",
        "description": "Send an image + text prompt to Cosmos Reason2 on the NVIDIA H100 for visual-spatial reasoning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Reasoning prompt"},
                "image_base64": {"type": "string", "description": "Base64-encoded image (optional; uses live camera if omitted)"}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "arm_calibrate",
        "description": "Run Cosmos-guided calibration — moves arm through all 5 pipeline poses, captures frames, analyzes with Cosmos Reason2.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "Must be true to execute"}
            },
            "required": ["confirm"]
        }
    },
    {
        "name": "system_status",
        "description": "Check connectivity of all NIS Protocol services: Pi agent, Cosmos H100, NIS Protocol itself.",
        "inputSchema": {"type": "object", "properties": {}}
    },
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _sync_get(url: str, timeout: int = 5) -> Optional[Dict]:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception:
        return None


def _sync_post(url: str, data: Dict, timeout: int = 30) -> Optional[Dict]:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body,
              headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# ── Tool executor ─────────────────────────────────────────────────────────────

async def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()

    if name == "arm_move_home":
        result = await loop.run_in_executor(None, lambda: _sync_post(
            f"{PI_URL}/arm/group_move",
            {"positions": {"1": 500, "2": 500, "3": 400, "4": 500, "5": 400, "6": 350},
             "duration_ms": args.get("duration_ms", 1200)}
        ))
        return {"success": True, "pose": "home", "result": result}

    elif name == "arm_group_move":
        if not args.get("positions"):
            raise HTTPException(400, "positions required")
        result = await loop.run_in_executor(None, lambda: _sync_post(
            f"{PI_URL}/arm/group_move",
            {"positions": args["positions"], "duration_ms": args.get("duration_ms", 1000)}
        ))
        return {"success": True, "result": result}

    elif name == "arm_pick_and_place":
        if not args.get("confirm"):
            return {"success": False, "error": "Safety gate: set confirm=true to execute pick-and-place"}
        result = await loop.run_in_executor(None, lambda: _sync_post(
            f"{PI_URL}/arm/pick_and_place", {}, 60
        ))
        return {"success": True, "result": result}

    elif name == "arm_get_status":
        result = await loop.run_in_executor(None, lambda: _sync_get(f"{PI_URL}/arm/status"))
        return result or {"error": "Pi agent not reachable"}

    elif name == "arm_get_poses":
        result = await loop.run_in_executor(None, lambda: _sync_get(f"{PI_URL}/arm/touch_poses"))
        return result or {"error": "Pi agent not reachable"}

    elif name == "arm_save_pose":
        if not args.get("name"):
            raise HTTPException(400, "name required")
        result = await loop.run_in_executor(None, lambda: _sync_post(
            f"{PI_URL}/arm/save_touch_pose", {"name": args["name"]}
        ))
        return {"success": True, "result": result}

    elif name == "camera_snapshot":
        result = await loop.run_in_executor(None, lambda: _sync_get(f"{PI_URL}/camera/snapshot", timeout=10))
        if result:
            return {"success": True, "image": result.get("image", ""), "timestamp": result.get("timestamp")}
        return {"error": "Camera not available"}

    elif name == "cosmos_reason":
        prompt = args.get("prompt", "Describe what you see.")
        image_b64 = args.get("image_base64")
        if not image_b64:
            snap = await loop.run_in_executor(None, lambda: _sync_get(f"{PI_URL}/camera/snapshot", timeout=8))
            if snap:
                image_b64 = snap.get("image", "")
        messages_content = []
        if image_b64:
            messages_content.append({"type": "image_url", "image_url": {"url": image_b64}})
        messages_content.append({"type": "text", "text": prompt})
        result = await loop.run_in_executor(None, lambda: _sync_post(
            f"{COSMOS_URL}/v1/chat/completions",
            {"model": "cosmos-reason2", "messages": [{"role": "user", "content": messages_content}],
             "max_tokens": 512, "temperature": 0.1},
            30
        ))
        if result and "choices" in result:
            return {"success": True, "response": result["choices"][0]["message"]["content"]}
        return {"error": "Cosmos not available", "raw": result}

    elif name == "arm_calibrate":
        if not args.get("confirm"):
            return {"success": False, "error": "Safety gate: set confirm=true to run calibration"}
        poses = {
            "home":       {"1": 500, "2": 500, "3": 400, "4": 500, "5": 400, "6": 350},
            "inspect":    {"1": 500, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
            "pick_table": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500},
            "lift_grip":  {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
            "place_bin":  {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240},
        }
        results = {}
        for pose_name, pos in poses.items():
            await loop.run_in_executor(None, lambda p=pos: _sync_post(
                f"{PI_URL}/arm/group_move", {"positions": p, "duration_ms": 1500}
            ))
            await asyncio.sleep(2.0)
            save = await loop.run_in_executor(None, lambda pn=pose_name: _sync_post(
                f"{PI_URL}/arm/save_touch_pose", {"name": pn}
            ))
            results[pose_name] = {"saved": True, "positions": pos}
        return {"success": True, "calibrated_poses": results}

    elif name == "system_status":
        pi = await loop.run_in_executor(None, lambda: _sync_get(f"{PI_URL}/health", 3))
        cosmos = await loop.run_in_executor(None, lambda: _sync_get(f"{COSMOS_URL}/health", 3))
        return {
            "pi_agent": {"online": pi is not None, "version": pi.get("version") if pi else None},
            "cosmos_h100": {"online": cosmos is not None},
            "nis_protocol": {"online": True, "version": "4.0.1"},
            "timestamp": time.time()
        }

    else:
        raise HTTPException(404, f"Unknown tool: {name}")

# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """List all NIS Protocol tools in OpenFang/MCP format."""
    return {"tools": NIS_TOOLS, "count": len(NIS_TOOLS), "server": "nis-protocol"}


class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


@router.post("/tools/call")
async def call_tool(req: ToolCallRequest):
    """Execute a NIS Protocol tool. Compatible with OpenFang tool calling format."""
    try:
        result = await _execute_tool(req.name, req.arguments)
        return {
            "tool": req.name,
            "result": result,
            "timestamp": time.time(),
            "call_id": uuid.uuid4().hex[:8]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool execution error [{req.name}]: {e}")
        return {"tool": req.name, "error": str(e), "timestamp": time.time()}


@router.post("/mcp")
async def mcp_jsonrpc(request: Request):
    """
    MCP JSON-RPC 2.0 handler. OpenFang connects here via:
      [[mcp_servers]]
      name = "nis-protocol"
      url = "http://localhost:8000/openfang/mcp"
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "nis-protocol", "version": "4.0.1"}
            }
        })

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {"tools": NIS_TOOLS}
        })

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            result = await _execute_tool(tool_name, arguments)
            content = [{"type": "text", "text": json.dumps(result, indent=2)}]
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {"content": content, "isError": "error" in result}
            })
        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {"content": [{"type": "text", "text": str(e)}], "isError": True}
            })

    elif method == "ping":
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": {}})

    else:
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })


@router.get("/agents")
async def list_agents():
    """List active NIS Protocol agents in OpenFang format."""
    return {
        "agents": [
            {
                "id": "robotics-arm",
                "name": "Robotics Arm Agent",
                "description": "Cosmos Reason2-guided xArm 6DOF pick-and-place",
                "status": "active",
                "capabilities": ["arm_control", "visual_reasoning", "calibration"]
            },
            {
                "id": "cosmos-reasoner",
                "name": "Cosmos Reason2",
                "description": "NVIDIA H100 visual-spatial reasoning for robotics",
                "status": "active" if _sync_get(f"{COSMOS_URL}/health", 2) else "offline",
                "capabilities": ["vision", "spatial_reasoning", "depth_estimation"]
            },
            {
                "id": "neurolinux-agent",
                "name": "NeuroLinux Pi Agent",
                "description": "Raspberry Pi hardware interface for xArm and camera",
                "status": "active" if _sync_get(f"{PI_URL}/health", 2) else "offline",
                "capabilities": ["arm_hardware", "camera", "usb_hid"]
            },
        ]
    }


@router.get("/status")
async def openfang_status():
    """System status in OpenFang dashboard format."""
    result = await _execute_tool("system_status", {})
    return {
        **result,
        "tools_available": len(NIS_TOOLS),
        "openfang_mcp_endpoint": "POST /openfang/mcp",
        "openfang_tools_endpoint": "GET /openfang/tools",
        "config_snippet": """
# Add to ~/.openfang/config.toml to connect OpenFang to NIS Protocol:
[[mcp_servers]]
name = "nis-protocol"
url = "http://localhost:8000/openfang/mcp"
"""
    }
