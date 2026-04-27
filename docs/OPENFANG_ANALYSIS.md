# OpenFang — Analysis & NIS Protocol Integration Plan

> Analyzed: February 17, 2026 | Repo: RightNow-AI/openfang (v0.1.3, 2.1k ⭐)

---

## What OpenFang Is

OpenFang is not a chatbot framework. It is a **full Agent Operating System** compiled into a single 32MB Rust binary.

Key distinction: **Traditional agents wait for you to type. OpenFang Hands work for you.**

This is exactly the vision the user has for NIS Protocol: *"my dream is for the nis protocol to work like claude code in the console or like openclaw/clawbot."*

---

## Why This Matters for NIS Protocol

### The OpenFang Architecture Maps 1:1 to NIS Protocol Concepts

| OpenFang Concept | NIS Protocol Equivalent | Status |
|---|---|---|
| **Kernel** (`openfang-kernel`) | `main.py` orchestration | Exists, less structured |
| **Hands** (autonomous scheduled tasks) | Pick-and-place pipeline | Exists as HTTP endpoints only |
| **Runtime** (agent loop + tools) | Agent routes (`/agents/*`) | Partial |
| **Channels** (40 messaging adapters) | Only HTTP REST | Missing |
| **Memory** (SQLite + vector) | In-memory only | Missing persistence |
| **CLI** (`openfang chat researcher`) | No console mode | **MISSING — user's dream** |
| **SKILL.md** (domain expertise injection) | No skill system | Missing |
| **HAND.toml** (declarative agent manifest) | No declarative config | Missing |
| **MCP integration** | `/mcp/*` routes | Exists |
| **OFP P2P protocol** | Pi ↔ Windows bridge | HTTP only |

### The Critical Gap: Console Mode

The user wants NIS Protocol to work like Claude Code in the console. OpenFang has this:
```bash
openfang chat researcher
> "What are the emerging trends in AI agent frameworks?"
```

NIS Protocol has no equivalent. Everything is HTTP endpoints + web dashboard.

---

## OpenFang's `agent.toml` Pattern (Game Changer)

The `home-automation` agent is almost identical to what NeuroLinux Agent should be:

```toml
name = "home-automation"
module = "builtin:chat"

[model]
provider = "groq"
model = "llama-3.3-70b-versatile"
system_prompt = """You are a smart home specialist..."""

[capabilities]
tools = ["file_read", "file_write", "shell_exec", "web_fetch"]
shell = ["curl *", "python *"]
```

vs. what NIS Protocol `neurolinux_agent.py` does:
- 600+ lines of FastAPI
- No declarative manifest
- No skill injection
- No model routing
- No fallback providers

### What a `robotics-arm.toml` would look like:

```toml
name = "robotics-arm"
description = "xArm 6DOF pick-and-place agent with Cosmos Reason2 visual reasoning"

[model]
provider = "openai"          # OpenAI-compatible → points to local Cosmos Reason2
base_url = "http://localhost:8100/v1"
system_prompt = """..."""     # 500-word procedure for pick-and-place

[hand]
schedule = { on_command = true }
approval_required = ["pick_and_place"]  # Approval gate before physical movement

[capabilities]
tools = ["shell_exec", "web_fetch", "memory_store", "memory_recall"]
shell = ["curl http://192.168.1.163:8085/*", "python *"]
```

---

## Three Integration Strategies (Ranked by ROI)

### Strategy 1 (BEST): Use NIS Protocol as an OpenFang MCP Server ⭐

OpenFang already supports MCP (Model Context Protocol). NIS Protocol already has 140+ endpoints.

**Connect them:**
```toml
# In ~/.openfang/config.toml
[[mcp_servers]]
name = "nis-protocol"
command = "python"
args = ["-m", "main"]
env = { NIS_PORT = "8000" }
```

Or via network (NIS Protocol already running):
```toml
[[mcp_servers]]
name = "nis-protocol-remote"
url = "http://localhost:8000/mcp"
```

**Result**: Any OpenFang agent (orchestrator, coder, researcher) can call:
- `arm_move`, `arm_pick_place`, `cosmos_reason`, `get_camera_frame`
- Via the OpenFang kernel, with audit trail, rate limiting, WASM sandbox

**Effort**: LOW (NIS Protocol likely needs a `/mcp` SSE endpoint added)

---

### Strategy 2: Port NIS Protocol to `agent.toml` Declarative Format

Replace Python class definitions with `agent.toml` manifests:

```
agents/
  xarm-controller/agent.toml     ← replaces neurolinux_agent.py
  cosmos-reasoner/agent.toml     ← replaces relay_h100.py
  nis-orchestrator/agent.toml    ← replaces main.py routing
```

**Effort**: MEDIUM (requires restructuring, preserves all existing logic)

---

### Strategy 3: Migrate Fully to OpenFang Runtime

Use `openfang migrate --from openclaw` (since NIS Protocol borrows from OpenClaw patterns) and rebuild the Pi agent as an OpenFang node.

**Effort**: HIGH — but OpenFang's P2P protocol (`openfang-wire`) would replace the current HTTP-based Pi ↔ Windows bridge with HMAC-authenticated OFP.

---

## What NIS Protocol Has That OpenFang Does NOT

| NIS Protocol Feature | Description |
|---|---|
| **Cosmos Reason2 integration** | VLM for spatial reasoning on H100 |
| **xArm physical control** | Real hardware via USB HID + serial |
| **NeuroKernel** | Custom biologically-inspired DIKW kernel |
| **Consciousness engine** | Attention, awareness, emotional weighting |
| **Isaac Lab sim** | NVIDIA simulation integration |
| **Real-time camera** | Pi camera + MJPEG streaming |
| **NVIDIA Cosmos stack** | Transfer, Predict, World Model integration |

**These are NIS Protocol's competitive moat.** OpenFang has no robotics or physical embodiment layer. NIS Protocol should NOT migrate away from this — it should EXPOSE these as OpenFang tools.

---

## What NIS Protocol Needs from OpenFang

### 1. Console Mode (User's Explicit Dream)

OpenFang Python SDK pattern — implement this for NIS Protocol:

```python
# nis_console.py — "Claude Code in the console"
from openfang_sdk import Agent

agent = Agent()

@agent.on_message
def handle(message: str, context: dict) -> str:
    # Route to NIS Protocol endpoints
    # Call Cosmos Reason2
    # Control xArm
    # Return reasoning + action taken
    return result

agent.run()
```

Run it:
```bash
python nis_console.py
> pick up the red block and place it in the bin
```

### 2. Declarative Agent Manifests (`agent.toml` style)

Replace inline Python dicts with `agent.toml`:

```toml
name = "xarm-cookoff-demo"
description = "Cosmos-guided pick-and-place for NVIDIA Cookoff"

[hand]
schedule = { continuous = { check_interval_secs = 0 } }
approval_required = ["pick_and_place"]

[model]
provider = "openai"
base_url = "http://172.16.1.83:8100/v1"
model = "cosmos-reason2"
```

### 3. MCP Endpoint on NIS Protocol

Add `/mcp` SSE endpoint to `main.py` so OpenFang can discover NIS Protocol's tools automatically.

### 4. Merkle Audit Trail

OpenFang's hash-chain audit trail is exactly what's needed for the cookoff demo — every arm movement cryptographically logged.

---

## Benchmark Context: Where NIS Protocol Sits

| Metric | OpenFang | NIS Protocol | Gap |
|---|---|---|---|
| Cold Start | 180ms | ~3s (Python FastAPI) | 16x slower |
| Memory | 40MB | ~180MB (Python) | 4.5x more |
| Security Layers | 16 | ~3 (auth, rate limit, CORS) | 13 missing |
| Channel Adapters | 40 | 0 (HTTP REST only) | All missing |
| Console Mode | Full CLI | None | All missing |
| Robotics | None | Full xArm + Cosmos | NIS leads |
| Physical Embodiment | None | Pi + Camera + H100 | NIS leads |

---

## Immediate Action Plan

### This Week (Cookoff Prep):

1. **Add `/mcp` SSE endpoint to NIS Protocol** → OpenFang can call arm endpoints
2. **Create `nis_console.py`** using OpenFang Python SDK pattern → Console mode
3. **Create `robotics-arm/agent.toml`** → Declarative pipeline manifest
4. **Install OpenFang locally** → `curl -fsSL https://openfang.sh/install | sh`

### After Cookoff:

5. Port agent configs to `agent.toml` format
6. Use OpenFang's orchestrator + NIS Protocol's physical layer as a combined system
7. Register NIS Protocol on FangHub marketplace

---

## One-Line Summary

> OpenFang is the **operating system layer** NIS Protocol is missing. NIS Protocol has the **physical intelligence layer** OpenFang cannot do. The right move is to run OpenFang as the runtime kernel and expose NIS Protocol's Cosmos + xArm + NeuroKernel as OpenFang tools via MCP.
