#!/usr/bin/env python3
"""
NIS Protocol CLI — Console-first agentic interface
====================================================
Works like Claude Code / OpenClaw in the terminal.

Usage
-----
  python nis_cli.py                        # interactive session
  python nis_cli.py "pick up the cube"     # single command
  python nis_cli.py --server ws://192.168.1.163:8000/ws/agentic "wave"
  python nis_cli.py --robot "pick and place demo"
  python nis_cli.py --status              # system health check
  python nis_cli.py --skills              # list available tools

Environment
-----------
  NIS_SERVER   ws://host:port/ws/agentic  (default: ws://localhost:8000/ws/agentic)
  NIS_API_KEY  optional bearer token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Optional

# ── Colour helpers (no external deps) ────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAGENTA= "\033[35m"
BLUE   = "\033[34m"

def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


def _wrap(text: str, prefix: str = "  ", width: int = 100) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if len(line) <= width - len(prefix):
            out.append(prefix + line)
        else:
            wrapped = textwrap.fill(line, width=width - len(prefix),
                                    initial_indent=prefix, subsequent_indent=prefix)
            out.append(wrapped)
    return "\n".join(out)


# ── Event renderers ───────────────────────────────────────────────────────────

def render_thinking(step: dict) -> None:
    title   = step.get("title", "Thinking")
    content = step.get("content", "")
    print(_c(f"  ~ {title}", DIM, CYAN))
    if content:
        print(_c(_wrap(content, "    "), DIM))


def render_tool_call(step: dict) -> None:
    tool    = step.get("tool", step.get("tool_id", "tool"))
    args    = step.get("args", {})
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
    print(_c(f"  > {tool}({arg_str})", BOLD, YELLOW))


def render_tool_result(step: dict) -> None:
    result  = step.get("result", "")
    ok_icon = _c("[ok]", GREEN) if step.get("ok", True) else _c("[er]", RED)
    # Trim very long results
    if isinstance(result, str) and len(result) > 400:
        result = result[:400] + "..."
    print(f"  {ok_icon} {_c(str(result), DIM)}")


def render_agent_activation(step: dict) -> None:
    name = step.get("agent_name", "agent")
    task = step.get("task", "")
    print(_c(f"  * {name}: {task}", DIM, MAGENTA))


def render_response(step: dict) -> None:
    content = step.get("content", "")
    meta    = step.get("metadata", {})
    provider = meta.get("provider", step.get("provider", "nis"))
    intent   = meta.get("intent", "")
    tools    = meta.get("tools_used") or intent or ""

    tag = f"[{provider}]"
    if tools and tools != "chat":
        tag += f" [{tools}]"

    print()
    print(_c(f"NIS {tag}", BOLD, CYAN))
    print(_c("─" * 60, DIM))
    print(_wrap(content, "  "))
    # Inline image notice
    if step.get("image_base64"):
        print(_c("  [image attached — base64 available in raw JSON]", DIM))
    print()


def render_error(msg: str) -> None:
    print(_c(f"  [!] {msg}", BOLD, RED))


# ── WebSocket session ─────────────────────────────────────────────────────────

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


async def _ws_session(server: str, message: str,
                      raw: bool = False,
                      image_b64: Optional[str] = None) -> int:
    """Send one message and stream all events until TEXT_MESSAGE_CONTENT arrives."""
    if not _WS_AVAILABLE:
        print(_c("[!] 'websockets' package not installed.\n"
                 "    Run: pip install websockets", BOLD, RED))
        return 1

    payload: dict = {"type": "message", "message": message}
    if image_b64:
        payload["image_base64"] = image_b64

    try:
        async with websockets.connect(server, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(json.dumps(payload))

            async for raw_msg in ws:
                if isinstance(raw_msg, bytes):
                    raw_msg = raw_msg.decode()
                try:
                    event = json.loads(raw_msg)
                except json.JSONDecodeError:
                    print(raw_msg)
                    continue

                if raw:
                    print(json.dumps(event, indent=2))
                    continue

                etype = event.get("type", "")

                if etype == "THINKING_STEP":
                    render_thinking(event)
                elif etype == "TOOL_CALL":
                    render_tool_call(event)
                elif etype == "TOOL_RESULT":
                    render_tool_result(event)
                elif etype == "AGENT_ACTIVATION":
                    render_agent_activation(event)
                elif etype == "AGENT_DEACTIVATION":
                    pass  # silent
                elif etype == "TEXT_MESSAGE_CONTENT":
                    render_response(event)
                    return 0   # done
                elif etype == "error":
                    render_error(event.get("message", str(event)))
                    return 1
                # Ignore pong, ack, etc.

    except (ConnectionRefusedError, OSError) as e:
        render_error(f"Cannot connect to {server}\n  {e}")
        print(_c("  Start NIS Protocol with: python main.py", DIM))
        return 1
    except Exception as e:
        render_error(f"WebSocket error: {e}")
        return 1

    return 0


# ── HTTP fallback (REST /chat) ────────────────────────────────────────────────

async def _http_session(server_http: str, message: str) -> int:
    """Fall back to POST /chat when WebSocket is unavailable."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{server_http}/chat",
                             json={"message": message,
                                   "use_tools": True,
                                   "enable_agents": True})
            if r.status_code == 200:
                data = r.json()
                fake_event = {
                    "content": data.get("response", ""),
                    "metadata": {
                        "provider": data.get("provider", "nis"),
                        "tools_used": ", ".join(data.get("tools_used", [])) or None,
                    }
                }
                render_response(fake_event)
                return 0
            render_error(f"HTTP {r.status_code}: {r.text[:200]}")
            return 1
    except ImportError:
        render_error("Neither 'websockets' nor 'httpx' is installed.\n  pip install websockets httpx")
        return 1
    except Exception as e:
        render_error(str(e))
        return 1


# ── Interactive REPL ──────────────────────────────────────────────────────────

BANNER = f"""
{BOLD}{CYAN}  NIS Protocol v4.0 — Agentic Console{RESET}
{DIM}  Organica AI Solutions · neurolinux-on-rpi5
  Type a command or question. Special commands:
    /status     system health check
    /skills     list available tools
    /arm home   move arm to home position
    /demo       run pick-and-place demo
    /history    show conversation history
    /raw        toggle raw JSON output
    /clear      clear screen
    /exit       quit{RESET}
"""

async def _interactive(server_ws: str, server_http: str, raw: bool = False) -> None:
    print(BANNER)
    history: list[str] = []
    _raw = raw

    while True:
        try:
            line = input(_c("nis> ", BOLD, CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # ── Built-in commands ─────────────────────────────────────────────
        if line in ("/exit", "/quit", "exit", "quit"):
            break
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue
        if line == "/raw":
            _raw = not _raw
            print(_c(f"  Raw JSON output: {'ON' if _raw else 'OFF'}", DIM))
            continue
        if line == "/history":
            if not history:
                print(_c("  No history yet.", DIM))
            else:
                for i, h in enumerate(history, 1):
                    print(_c(f"  {i:2}. {h[:80]}", DIM))
            continue

        # Map shortcuts to NL commands the agentic WS understands
        shortcuts = {
            "/status": "what is the system status",
            "/skills": "what skills and tools are available",
            "/demo":   "run pick and place demo",
            "/home":   "move arm to home position",
        }
        if line.startswith("/arm "):
            line = "arm " + line[5:]
        msg = shortcuts.get(line, line)

        history.append(msg)
        rc = await _ws_session(server_ws, msg, raw=_raw)
        if rc != 0:
            # WS failed — try HTTP
            print(_c("  [!] WebSocket unavailable, falling back to REST /chat", DIM))
            await _http_session(server_http, msg)


# ── Single-shot mode ──────────────────────────────────────────────────────────

async def _run_once(server_ws: str, server_http: str,
                    message: str, raw: bool = False) -> int:
    print(_c(f"\n  {message}", BOLD))
    print(_c("  " + "─" * 60, DIM))
    rc = await _ws_session(server_ws, message, raw=raw)
    if rc != 0:
        print(_c("  [!] WS unavailable, falling back to REST", DIM))
        rc = await _http_session(server_http, message)
    return rc


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nis",
        description="NIS Protocol CLI — agentic console for NeuroLinux on Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          python nis_cli.py                          # interactive REPL
          python nis_cli.py "wave the arm"           # single command
          python nis_cli.py --status                 # health check
          python nis_cli.py --skills                 # list tools
          python nis_cli.py --robot "pick up cube"   # robot control
          python nis_cli.py --raw "what do you see"  # raw JSON events

        Connect to a remote NIS server
        --------------------------------
          python nis_cli.py --server ws://192.168.1.163:8000/ws/agentic "hello"
          NIS_SERVER=ws://192.168.1.163:8000/ws/agentic python nis_cli.py
        """)
    )
    p.add_argument("message", nargs="?", default=None,
                   help="Message to send (omit for interactive mode)")
    p.add_argument("--server", "-s",
                   default=os.getenv("NIS_SERVER", "ws://localhost:8000/ws/agentic"),
                   help="WebSocket server URL (default: ws://localhost:8000/ws/agentic)")
    p.add_argument("--raw", "-r", action="store_true",
                   help="Print raw JSON event stream instead of formatted output")
    p.add_argument("--status", action="store_true",
                   help="Check system health and exit")
    p.add_argument("--skills", action="store_true",
                   help="List available tools / skills and exit")
    p.add_argument("--robot", action="store_true",
                   help="Hint: this command involves robot hardware (adds intent context)")
    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()

    # Derive HTTP URL from WS URL
    server_ws = args.server
    server_http = server_ws.replace("ws://", "http://").replace("wss://", "https://")
    server_http = server_http.rsplit("/ws", 1)[0]  # strip /ws/agentic

    # Prefixes for special flags
    if args.status:
        args.message = "what is the system status"
    elif args.skills:
        args.message = "what skills and tools are available"
    elif args.robot and args.message:
        args.message = f"[robot] {args.message}"

    if args.message:
        rc = asyncio.run(_run_once(server_ws, server_http, args.message, raw=args.raw))
        sys.exit(rc)
    else:
        asyncio.run(_interactive(server_ws, server_http, raw=args.raw))


if __name__ == "__main__":
    main()
