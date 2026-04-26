# NIS Protocol — Quick Start Guide

> **Neural Intelligence System v4.0** · Organica AI Solutions  
> An open AGI foundation — biologically inspired, hardware-ready, console-first.

---

## What is NIS Protocol?

NIS Protocol is a modular AI operating system that lets you build agentic systems that **do things in the real world** — control robot arms, analyze camera feeds with Cosmos VLMs, coordinate multi-agent pipelines, and talk to hardware over USB/CAN — all through a clean Python API and a Claude-Code-style terminal interface.

```
User prompt → Intent detection → Tool dispatch → LLM synthesis → Action
              (vision / xarm / cosmos / status)   (ChromaDB memory injected)
```

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│  NIS CLI (nis_cli.py)                                           │
│  Web Dashboard (static/)   ←→  POST /chat  ←→  WebSocket /ws/agentic │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  NIS Protocol API  │   main.py (FastAPI)
                    │  port 8000         │   290+ endpoints, 30 routers
                    └────┬───────┬───────┘
                         │       │
              ┌──────────▼─┐   ┌─▼────────────┐
              │ Tool Executor│   │ LLM Manager  │
              │ (src/core/  │   │ (Anthropic / │
              │  tool_executor│   │  OpenAI /    │
              │  .py)        │   │  NVIDIA NIM) │
              └──────┬──────┘   └──────────────┘
                     │
        ┌────────────┼────────────────────┐
        │            │                    │
┌───────▼──┐  ┌──────▼──────┐  ┌─────────▼──────┐
│NeuroLinux│  │ NVIDIA       │  │ ChromaDB        │
│ Agent    │  │ Cosmos H100  │  │ Persistent      │
│ port 8085│  │ Reason2      │  │ Memory          │
│ (Pi/USB) │  │ port 8100    │  │                 │
└──────────┘  └─────────────┘  └────────────────┘
     │
  xArm 6DOF + Pi Camera
```

---

## 1. Install

```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
pip install -r requirements.txt

# Optional: NeuroLinux agent (Raspberry Pi only)
# git clone https://github.com/Organica-Ai-Solutions/NeuroLinux.git
```

### Required environment variables

Create a `.env` file (never commit this):

```env
# At least one LLM provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or (for local)
OLLAMA_BASE_URL=http://localhost:11434

# Optional: NVIDIA stack
NVIDIA_API_KEY=nvapi-...

# Optional: ChromaDB path (defaults to ./data/chromadb)
CHROMA_DB_PATH=./data/chromadb
```

---

## 2. Run the NIS Protocol Server

```bash
python main.py
# Server starts at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

---

## 3. Use the CLI (like Claude Code)

```bash
# Install WebSocket client (one-time)
pip install websockets httpx

# Interactive REPL — Claude Code style
python nis_cli.py

# Single commands
python nis_cli.py "what is the status of all services"
python nis_cli.py "wave the robot arm"
python nis_cli.py "take a photo and describe the scene"
python nis_cli.py "run the pick-and-place demo"
python nis_cli.py --status
python nis_cli.py --skills

# Connect to a remote Pi running NIS
python nis_cli.py --server ws://192.168.1.163:8000/ws/agentic "pick up the cube"

# Raw JSON event stream (for debugging / tool integration)
python nis_cli.py --raw "plan a robot grasp"
```

The CLI streams live tool calls, agent activations, and the final response — just like Claude Code.

---

## 4. REST API — POST /chat (fully agentic)

Every `POST /chat` request now automatically:
1. Retrieves relevant memories from ChromaDB
2. Detects intent (vision / xarm / cosmos / status / skills)
3. Dispatches the right tool and injects results into the LLM prompt
4. Stores the interaction back in memory

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "take a snapshot and plan a robot grasp", "use_tools": true}'
```

Response includes `tools_used`, `tool_results`, and `intent`.

---

## 5. NeuroLinux Agent (Raspberry Pi / Robot Hardware)

The agent runs on the Pi at **port 8085** and exposes:

| Endpoint | Description |
|---|---|
| `POST /arm/named/{name}` | Move to calibrated named position |
| `POST /arm/group_move` | Move multiple servos (servo-unit 0–1000) |
| `POST /arm/gripper/open` | Open gripper (S1 → 100) |
| `POST /arm/gripper/close` | Close gripper (S1 → 550) |
| `POST /arm/pick_and_place` | Full 6-step pick-and-place pipeline |
| `POST /arm/save_touch_pose` | Save current arm position as named pose |
| `GET /arm/touch_poses` | List all calibrated positions |
| `POST /arm/train` | Enable compliant mode for hand-teaching |
| `GET /camera/status` | Camera availability and type |
| `GET /camera/snapshot` | JPEG snapshot |
| `POST /camera/cosmos-arm` | Snapshot → Cosmos reasoning → arm execution |
| `POST /agent/chat` | NL chat with intent routing |

### Calibrate your robot arm (first-time setup)

```python
import requests

BASE = "http://192.168.1.163:8085"   # Pi IP

# 1. Enable training mode (servos go compliant)
requests.post(f"{BASE}/arm/train")
# 2. Physically move arm to the desired position
# 3. Save it
requests.post(f"{BASE}/arm/save_touch_pose", json={"name": "home"})
# 4. Repeat for: inspect, pick_table, lift_grip, place_bin
# 5. Exit training mode
requests.post(f"{BASE}/arm/train/stop")
```

---

## 6. Key Files for Contributors

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, `/chat`, all WebSocket endpoints |
| `nis_cli.py` | Terminal CLI (Claude Code style) |
| `src/core/tool_executor.py` | **Shared HTTP tool runner** — add new tools here |
| `routes/` | 30 modular FastAPI routers |
| `src/memory/persistent_memory.py` | ChromaDB long-term memory |
| `src/llm/llm_manager.py` | Multi-provider LLM (OpenAI, Anthropic, NVIDIA, Ollama) |
| `src/agents/` | All cognitive agents (planning, curiosity, consciousness…) |
| `routes/openclaw.py` | OpenClaw bridge — maps NIS tools to OpenClaw protocol |

---

## 7. Adding a New Tool

1. Add a function to `src/core/tool_executor.py`:

```python
async def my_new_tool(query: str) -> dict:
    """Tool doc — returned dict must have 'ok', 'summary', 'tool' keys."""
    result = do_something(query)
    return {"tool": "my_new_tool", "ok": True, "summary": result}
```

2. Register it in `detect_intent` and `dispatch` in the same file:

```python
INTENT_KEYWORDS["my_intent"] = ["keyword1", "keyword2"]

# In dispatch():
if intent == "my_intent":
    return await my_new_tool(message)
```

3. The tool is now automatically available in:
   - `POST /chat` (REST)
   - `/ws/agentic` (WebSocket)
   - `python nis_cli.py "keyword1 something"` (CLI)

---

## 8. Project Vision

NIS Protocol is designed to become the **Linux of AI operating systems** — a composable, hardware-aware AGI foundation that:

- Works on a $35 Raspberry Pi **and** an NVIDIA H100 cluster
- Understands natural language → executes real physical actions
- Remembers every interaction via semantic memory
- Self-optimizes through the built-in self-modifier
- Stays open and extensible via 30 modular routers

**Current hardware integrations:**
- Hiwonder xArm 6-DOF (USB HID)
- Pi Camera 3 / Logitech C270
- NVIDIA Cosmos Reason2 (8B VLM on H100)
- CAN bus (NeuroKernel layer)

---

## 9. Related Repositories

| Repo | What it is |
|---|---|
| [NeuroLinux](https://github.com/Organica-Ai-Solutions/NeuroLinux) | Custom Linux OS for NeuroLinux Agent on Pi |
| [NIS-TOOLKIT-SUIT](https://github.com/Organica-Ai-Solutions/NIS-TOOLKIT-SUIT) | Full developer toolkit (NDT + NAT) |
| [NIS-HUB](https://github.com/Organica-Ai-Solutions/NIS-HUB) | Enterprise coordination hub for distributed NIS deployments |

---

## License

NIS Protocol is released under the [Business Source License (BSL)](LICENSE).  
Free for personal use, research, and education.

---

*Built by [Organica AI Solutions](https://github.com/Organica-Ai-Solutions)*
