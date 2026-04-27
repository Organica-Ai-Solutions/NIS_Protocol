#!/usr/bin/env python3
"""
NIS Protocol MCP Bridge for Cursor
====================================
Speaks standard MCP stdio protocol (JSON-RPC 2.0).
Forwards tool calls to NIS Protocol at localhost:8090 (SSH tunnel to H100).

Cursor adds this via ~/.cursor/mcp.json:
  "nis-protocol": { "command": "python", "args": ["<path>/nis_mcp_bridge.py"] }
"""

import sys
import json
import urllib.request
import urllib.error
import logging

NIS_BASE = "http://localhost:8090"   # SSH tunnel: Windows→H100:8090
PI_BASE  = "http://192.168.1.163:8085"  # Pi direct (Windows is on same LAN)

logging.basicConfig(filename="C:/Users/DiegoTorres/AppData/Local/Temp/nis_mcp_bridge.log",
                    level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("nis_mcp")


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _post(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Tool definitions (MCP schema) ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "nis_chat",
        "description": (
            "Send a message to NIS Protocol (Neural Intelligence System) running on H100 GPU. "
            "Uses Qwen3.5-35B with LoRA adapter. Supports agentic mode with tool use. "
            "NIS controls an xArm robot, Pi camera, and Cosmos Reason2 vision model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Your message or question"},
                "use_tools": {"type": "boolean", "description": "Enable NIS agentic tools", "default": True},
                "provider": {"type": "string", "description": "LLM provider (default: openai=Qwen3.5)", "default": "openai"}
            },
            "required": ["message"]
        }
    },
    {
        "name": "nis_status",
        "description": (
            "Get full system status: H100 GPU services (Cosmos Reason2, Predict2.5, Transfer2.5, VLA xArm), "
            "Pi arm agent, NIS server health, and all service connectivity."
        ),
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "nis_arm_goto",
        "description": (
            "Move xArm to a named position via NIS→Pi tunnel. "
            "Named positions: home, ready, pick_table, place_bin, reach_forward, wave_up, wave_side, "
            "grip_open, grip_close, pick_blue, pick_yellow, pick_green."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Named position (e.g. 'home', 'pick_blue')"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "nis_arm_move",
        "description": (
            "Move individual xArm servos. Servo map: "
            "s1=gripper(100=open,500=close), s2=base_pan(500=center), "
            "s3=shoulder, s4=elbow, s5=shoulder_rot(keep 500), s6=main_reach(500=compact,800=extended)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "object",
                    "description": "Servo positions dict, e.g. {\"1\": 100, \"2\": 500}. Range: 0-1000."
                },
                "duration_ms": {"type": "integer", "description": "Move duration in ms", "default": 1000}
            },
            "required": ["positions"]
        }
    },
    {
        "name": "nis_arm_snapshot",
        "description": "Capture a snapshot from the Pi camera (Logitech C270). Returns base64 JPEG.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "nis_cosmos_plan",
        "description": (
            "Query Cosmos Reason2 (8B VLM on H100) for a robot action plan. "
            "Describe the scene or task and get a step-by-step manipulation plan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Scene description or task to plan"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "nis_cookoff_demo",
        "description": (
            "Run the full cookoff pick-and-place demo pipeline: "
            "Cosmos Reason2 plans → VLA xArm model executes → Pi arm moves → verify. "
            "Use dry_run=true to get the plan without physically moving the arm."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "description": "Plan only, don't move arm", "default": True}
            }
        }
    },
    {
        "name": "nis_cookoff_run",
        "description": (
            "FULL AUTONOMOUS cookoff pipeline with YOLO+GDINO scene detection. "
            "Supports natural language: 'pick all lighters and put them in the bin', "
            "'pick the blue lighter', 'pick every object'. "
            "Detects ALL objects in scene with YOLOv8x + Grounding DINO, "
            "Cosmos Reason2 analyzes scene, then picks each target and places in bin. "
            "Use simulation=true to plan without moving arm."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Natural language task (e.g. 'pick all lighters and put them in the bin')"},
                "execute_arm": {"type": "boolean", "description": "Execute real arm movement", "default": True},
                "simulation": {"type": "boolean", "description": "Simulate without moving arm", "default": False}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "nis_autonomous",
        "description": (
            "FULL AUTONOMOUS lighter sweep — scans table, picks ALL lighters with retry, places each in bin. "
            "Uses YOLOv8x + GDINO + YOLO-World (open-vocab 'lighter') + Cosmos Reason2 vision chain. "
            "Retries each pick up to 3×, adjusting S6 servo ±15px per retry. "
            "Keeps looping until table is clear or max_picks reached. "
            "Best choice for cookoff demo: 'pick all lighters from random positions'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "High-level task (default: 'pick all lighters and put them in the bin')",
                    "default": "pick all lighters and put them in the bin"
                },
                "max_picks": {"type": "integer", "description": "Max total picks (default: 10)", "default": 10},
                "max_retries": {"type": "integer", "description": "Max retries per lighter (default: 3)", "default": 3},
                "execute_arm": {"type": "boolean", "description": "False = simulation/dry-run", "default": True},
                "conf": {"type": "number", "description": "YOLO confidence threshold (default: 0.08)", "default": 0.08}
            }
        }
    },
    {
        "name": "nis_web_search",
        "description": "Search the web via DuckDuckGo through NIS Protocol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "nis_research",
        "description": "Run a deep research task via NIS autonomous research agent (uses web search + Qwen3.5).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic"},
                "depth": {"type": "string", "enum": ["quick", "deep"], "default": "quick"}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "nis_arm_read_positions",
        "description": "Read current servo positions from xArm (live hardware state).",
        "inputSchema": {"type": "object", "properties": {}}
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

def handle_nis_chat(args):
    result = _post(f"{NIS_BASE}/chat", {
        "message": args["message"],
        "use_tools": args.get("use_tools", True),
        "enable_agents": True,
        "provider": args.get("provider", "openai"),
    }, timeout=60)
    if "error" in result:
        return result["error"]
    resp = result.get("response", "")
    model = result.get("model", "?")
    tools_used = result.get("tools_used", [])
    suffix = f"\n\n_(provider: {model}"
    if tools_used:
        suffix += f" | tools: {', '.join(tools_used)}"
    suffix += ")_"
    return resp + suffix


def handle_nis_status(args):
    health = _get(f"{NIS_BASE}/health")
    cookoff = _get(f"{NIS_BASE}/cookoff/status")
    pi = _get(f"{PI_BASE}/health", timeout=5)

    lines = [
        f"**NIS Protocol** v{health.get('version','?')} — {health.get('status','?')}",
        f"  Routes: {health.get('modular_routes','?')} | Uptime: {health.get('uptime_seconds',0):.0f}s",
        "",
        "**H100 Services:**",
    ]
    for svc, info in cookoff.get("h100_services", {}).items():
        ok = "OK" if info.get("healthy") else "FAIL"
        lines.append(f"  {svc}: {ok}")
    lines += [
        "",
        "**Pi Agent (192.168.1.163:8085):**",
        f"  status: {pi.get('status','unreachable')} | xarm: {pi.get('xarm','?')} | camera: {pi.get('camera','?')}",
    ]
    return "\n".join(lines)


def handle_nis_arm_goto(args):
    name = args["name"]
    result = _post(f"{PI_BASE}/arm/goto", {"name": name}, timeout=10)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Arm moved to '{name}': {result.get('status', result)}"


def handle_nis_arm_move(args):
    positions = {str(k): int(v) for k, v in args["positions"].items()}
    duration = args.get("duration_ms", 1000)
    result = _post(f"{PI_BASE}/arm/group_move",
                   {"positions": positions, "duration_ms": duration}, timeout=10)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Servos moved: {positions} in {duration}ms — {result.get('status', 'ok')}"


def handle_nis_arm_snapshot(args):
    result = _get(f"{PI_BASE}/camera/snapshot", timeout=10)
    if "error" in result:
        return f"Camera error: {result['error']}"
    b64 = result.get("image_base64", "")
    if b64:
        return f"[Camera snapshot: {len(b64)} bytes base64 JPEG captured from Pi]"
    return f"Snapshot result: {result}"


def handle_nis_cosmos_plan(args):
    result = _post(f"{NIS_BASE}/cookoff/robot-plan",
                   {"query": args["query"]}, timeout=45)
    if "error" in result:
        return f"Error: {result['error']}"
    plan = result.get("plan", result.get("cosmos_reasoning", result))
    confidence = result.get("confidence", "?")
    return f"**Cosmos Reason2 Plan** (confidence: {confidence}):\n{json.dumps(plan, indent=2)}"


def handle_nis_cookoff_run(args):
    result = _post(f"{NIS_BASE}/cookoff/run", {
        "prompt":      args.get("prompt", "pick all lighters and put them in the bin"),
        "execute_arm": args.get("execute_arm", True),
        "simulation":  args.get("simulation", False),
    }, timeout=120)
    if "error" in result:
        return f"Error: {result['error']}"
    picks = result.get("pick_results", [])
    n_ok = sum(1 for p in picks if p.get("ok"))
    lines = [
        f"**Cookoff Run** — prompt: \"{args.get('prompt')}\"",
        f"  targets: {[p.get('target',{}).get('object','?') for p in picks]}",
        f"  picks: {n_ok}/{len(picks)} OK",
        f"  r2_reasoning: {str(result.get('r2_reasoning','?'))[:200]}",
        f"  scene: {str(result.get('scene0',''))[:100]}",
        f"  latency: {result.get('latency_ms','?')}ms",
    ]
    for p in picks:
        t = p.get("target", {})
        lines.append(f"  → {t.get('object','?')} s6={p.get('s6')} place={p.get('place_zone')} ok={p.get('ok')}")
    return "\n".join(lines)


def handle_nis_autonomous(args):
    result = _post(f"{NIS_BASE}/cookoff/autonomous", {
        "task":        args.get("task", "pick all lighters and put them in the bin"),
        "max_picks":   args.get("max_picks", 10),
        "max_retries": args.get("max_retries", 3),
        "execute_arm": args.get("execute_arm", True),
        "conf":        args.get("conf", 0.08),
    }, timeout=300)
    if "error" in result:
        return f"Autonomous error: {result['error']}"
    lines = [
        f"**AUTONOMOUS SWEEP** — {'TABLE CLEAR [OK]' if result.get('table_clear') else 'STOPPED'}",
        f"Picks: {result.get('picks_success',0)}/{result.get('picks_total',0)} succeeded",
        f"Retries used: {result.get('total_retries', 0)}",
        f"Latency: {result.get('latency_ms',0)}ms",
        "",
        "**Pick log:**",
    ]
    for p in result.get("all_picks", []):
        status = "[OK]" if p.get("success") else "[FAIL]"
        lines.append(f"  {status} #{p['lighter_idx']+1} {p.get('label','')} cx={p.get('yolo_cx')} cosmos_cx={p.get('cosmos_cx')} place={p.get('place_zone')} attempts={len(p.get('attempts',[]))}")
    lines.append("")
    for log in result.get("logs", []):
        lines.append(f"  {log}")
    return "\n".join(lines)


def handle_nis_cookoff_demo(args):
    result = _post(f"{NIS_BASE}/cookoff/demo",
                   {"dry_run": args.get("dry_run", True)}, timeout=60)
    if "error" in result:
        return f"Error: {result['error']}"
    lines = [
        f"**Cookoff Demo** — {result.get('task', '?')}",
        f"  plan_source: {result.get('plan_source', '?')}",
        f"  confidence: {result.get('confidence', '?')}",
        f"  safe: {result.get('safe', '?')}",
        f"  steps: {result.get('steps_ok', '?')}/{result.get('steps_total', '?')}",
        f"  cosmos_reasoning: {result.get('cosmos_reasoning', '?')}",
        f"  yolo_scene: {result.get('yolo_scene', '?')}",
    ]
    exec_info = result.get("execution", {})
    if exec_info:
        lines.append(f"  execution: {exec_info.get('status', exec_info)}")
    return "\n".join(lines)


def handle_nis_web_search(args):
    result = _post(f"{NIS_BASE}/protocol/mcp/execute",
                   {}, timeout=20)
    # Fallback: use research endpoint
    result = _post(f"{NIS_BASE}/research/search",
                   {"query": args["query"]}, timeout=20)
    if "error" in result or not result:
        # Final fallback: send to NIS chat
        result = _post(f"{NIS_BASE}/chat",
                       {"message": f"Search the web for: {args['query']}",
                        "use_tools": True}, timeout=30)
        return result.get("response", str(result))
    return json.dumps(result, indent=2)[:2000]


def handle_nis_research(args):
    result = _post(f"{NIS_BASE}/research/deep",
                   {"topic": args["topic"], "depth": args.get("depth", "quick")},
                   timeout=60)
    if "error" in result:
        # Fallback to chat
        result = _post(f"{NIS_BASE}/chat",
                       {"message": f"Do deep research on: {args['topic']}",
                        "use_tools": True, "enable_agents": True}, timeout=60)
        return result.get("response", str(result))[:3000]
    return json.dumps(result, indent=2)[:3000]


def handle_nis_arm_read(args):
    result = _get(f"{PI_BASE}/arm/positions", timeout=8)
    if "error" in result:
        return f"Error reading arm: {result['error']}"
    positions = result.get("positions", result)
    lines = ["**Current xArm Servo Positions:**"]
    servo_names = {1: "gripper", 2: "base_pan", 3: "shoulder2",
                   4: "elbow", 5: "shoulder_rot", 6: "main_reach"}
    if isinstance(positions, dict):
        for sid, pos in sorted(positions.items(), key=lambda x: int(x[0])):
            name = servo_names.get(int(sid), f"s{sid}")
            lines.append(f"  S{sid} ({name}): {pos}")
    else:
        lines.append(str(positions))
    return "\n".join(lines)


HANDLERS = {
    "nis_chat":           handle_nis_chat,
    "nis_status":         handle_nis_status,
    "nis_arm_goto":       handle_nis_arm_goto,
    "nis_arm_move":       handle_nis_arm_move,
    "nis_arm_snapshot":   handle_nis_arm_snapshot,
    "nis_cosmos_plan":    handle_nis_cosmos_plan,
    "nis_cookoff_demo":   handle_nis_cookoff_demo,
    "nis_cookoff_run":    handle_nis_cookoff_run,
    "nis_autonomous":     handle_nis_autonomous,
    "nis_web_search":     handle_nis_web_search,
    "nis_research":       handle_nis_research,
    "nis_arm_read_positions": handle_nis_arm_read,
}


# ── MCP stdio server ──────────────────────────────────────────────────────────

def send(obj):
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log.info(f">> {line[:200]}")


def handle(msg):
    mid = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {})
    log.info(f"<< {method} id={mid}")

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nis-protocol", "version": "4.0.1"}
        }})

    elif method == "notifications/initialized":
        pass  # no response needed

    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})

    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        handler = HANDLERS.get(name)
        if not handler:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": f"Unknown tool: {name}"
            }})
            return
        try:
            result_text = handler(args)
        except Exception as e:
            log.exception(f"handler {name} failed")
            result_text = f"Tool error: {e}"
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": str(result_text)}]
        }})

    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": f"Method not found: {method}"
        }})


def main():
    log.info("NIS MCP bridge started")
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = json.loads(raw_line)
            handle(msg)
        except json.JSONDecodeError as e:
            log.error(f"JSON parse error: {e} — raw: {raw_line[:100]}")
        except Exception as e:
            log.exception("Unhandled error")


if __name__ == "__main__":
    main()
