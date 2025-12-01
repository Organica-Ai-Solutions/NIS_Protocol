# NIS Protocol v4.0

**Enterprise AI Operating System**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Flutter](https://img.shields.io/badge/Flutter-Desktop-02569B?logo=flutter&logoColor=white)](https://flutter.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Inception-76B900?logo=nvidia&logoColor=white)](https://nvidia.com)

<p align="center">
  <img src="system/assets/images_organized/nis-protocol-logov1.png" alt="NIS Protocol" width="200"/>
</p>

<p align="center">
  <strong>Physics-validated AI coordination for autonomous systems</strong>
</p>

---

## Quick Start

```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
./start.sh

# Access points
# API:     http://localhost:8000
# Docs:    http://localhost:8000/docs
# Console: http://localhost:8000/console
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NIS Protocol v4.0                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Flutter   │  │   REST API  │  │  WebSocket  │  │     MCP     │        │
│  │   Desktop   │  │   Clients   │  │   Clients   │  │   Clients   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  ┌─────────────────────────────────▼─────────────────────────────────────┐  │
│  │                        FastAPI Backend (main.py)                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    10-Phase Consciousness Pipeline                │ │  │
│  │  │  Genesis → Plan → Collective → Multipath → Ethics → Embodiment  │ │  │
│  │  │  Evolution → Reflection → Marketplace → Debug                    │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐         │  │
│  │  │  Laplace   │ │    KAN     │ │    PINN    │ │  Robotics  │         │  │
│  │  │  Signal    │ │  Reasoning │ │  Physics   │ │   Agent    │         │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘         │  │
│  │                                                                        │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐         │  │
│  │  │   Vision   │ │  Research  │ │   Voice    │ │   BitNet   │         │  │
│  │  │   Agent    │ │   Agent    │ │   Agent    │ │   Local    │         │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─────────────────────────────────▼─────────────────────────────────────┐  │
│  │                         LLM Provider Layer                             │  │
│  │   OpenAI  │  Anthropic  │  Google  │  DeepSeek  │  NVIDIA  │  BitNet  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─────────────────────────────────▼─────────────────────────────────────┐  │
│  │                          Infrastructure                                │  │
│  │        Redis  │  Kafka  │  PostgreSQL  │  Docker  │  Nginx            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

<p align="center">
  <img src="system/assets/images_organized/system_screenshots/diagram-nis-core%20agents%20v2.png" alt="Core Agents" width="700"/>
</p>

---

## Core Capabilities

| Category | Features |
|----------|----------|
| **Consciousness** | 10-phase cognitive pipeline: Genesis, Plan, Collective, Multipath, Ethics, Embodiment, Evolution, Reflection, Marketplace, Debug |
| **Physics** | PINN validation, Laplace transforms, KAN reasoning, conservation law enforcement |
| **Robotics** | Forward/Inverse Kinematics, trajectory planning, TMR redundancy, MAVLink/ROS support |
| **LLM** | OpenAI, Anthropic, Google, DeepSeek, NVIDIA NIM, BitNet (offline) |
| **Vision** | Image analysis, generation, document processing |
| **Voice** | Multi-speaker TTS, real-time STT, WebSocket streaming |
| **Research** | Web search, ArXiv integration, deep analysis |
| **Auth** | JWT tokens, API keys, role-based access, dev bypass |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v4.0** | Nov 2025 | 10-phase consciousness, full robotics, Flutter frontend, auth system, BitNet local AI |
| **v3.2** | Jan 2025 | Intelligent query router, multimodal vision, NVIDIA NeMo, MCP integration |
| **v3.1** | Dec 2024 | Voice processing, enhanced memory, protocol bridge |
| **v3.0** | Nov 2024 | Multi-LLM coordination, physics validation, Docker deployment |
| **v2.0** | Sep 2024 | KAN reasoning, Laplace transforms, agent orchestration |
| **v1.0** | Jun 2024 | Initial release, basic agent framework |

### v4.0 Changelog

- **10-Phase Consciousness Pipeline** - Full cognitive architecture from idea generation to physical embodiment
- **Robotics Integration** - FK/IK with Denavit-Hartenberg, trajectory planning, NASA-grade TMR
- **BitNet Local Intelligence** - 1.58-bit quantized models for offline operation
- **Authentication System** - User management, API keys, JWT tokens
- **Flutter Desktop App** - Cross-platform frontend with agentic chat
- **214+ API Endpoints** - Comprehensive REST API coverage
- **Protocol Integration** - ACP (IBM), A2A (Google), MCP (Anthropic), S2A support
- **Agent Collaboration** - Multi-agent workflows with consensus synthesis
- **Persistent Memory** - Cross-session memory with namespaces and TTL
- **Real-time Dashboard** - System monitoring UI at `/static/dashboard.html`
- **Local MCP Server** - Built-in MCP server for tool sharing

---

## API Overview

### Consciousness Endpoints
```bash
POST /v4/consciousness/genesis     # Idea generation
POST /v4/consciousness/plan        # Strategic planning
POST /v4/consciousness/collective  # Multi-agent consensus
POST /v4/consciousness/multipath   # Parallel reasoning
POST /v4/consciousness/ethics      # Ethical evaluation
POST /v4/consciousness/embodiment  # Physical integration
```

### Robotics Endpoints
```bash
POST /robotics/forward_kinematics  # Joint angles → end effector
POST /robotics/inverse_kinematics  # Target pose → joint angles
POST /robotics/plan_trajectory     # Physics-validated paths
GET  /robotics/capabilities        # System capabilities
WS   /ws/robotics/control/{id}     # Real-time control
```

### Physics Endpoints
```bash
POST /physics/validate             # Constraint validation
POST /physics/solve/heat-equation  # PINN heat solver
POST /physics/solve/wave-equation  # PINN wave solver
GET  /physics/constants            # Physical constants
```

### Auth Endpoints
```bash
POST /auth/signup                  # Create account
POST /auth/login                   # Authenticate
POST /auth/logout                  # Invalidate token
GET  /auth/verify                  # Verify token
POST /users/api-keys               # Create API key
```

### Protocol Endpoints (ACP/A2A/MCP)
```bash
# ACP - Agent Communication Protocol (IBM/Linux Foundation)
GET  /agents                       # List agent manifests
POST /runs                         # Create agent run
GET  /runs/{run_id}                # Get run status
GET  /sessions/{session_id}        # Get session history

# A2A - Agent-to-Agent (Google)
POST /protocol/a2a/create-task     # Create A2A task
GET  /protocol/a2a/task/{id}       # Get task status

# MCP - Model Context Protocol (Anthropic)
GET  /protocol/mcp/tools           # Discover tools
POST /protocol/mcp/initialize      # Initialize server
POST /api/mcp/tools                # Execute tools
```

### Agent Collaboration
```bash
POST /agents/collaborate           # Multi-agent task
POST /memory/store                 # Store persistent memory
GET  /memory/retrieve/{ns}/{key}   # Retrieve memory
GET  /memory/list/{namespace}      # List memory keys
```

---

## Deployment

### Docker (Recommended)
```bash
# Development (safe mode - mock responses)
./start_safe.sh

# Production
cp .env.example .env
# Edit .env with your API keys
./start.sh
```

### Requirements
- Docker 20.10+
- 8GB RAM minimum
- 10GB disk space

### Environment Variables
```bash
# Required for full functionality
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
GOOGLE_API_KEY=...
NVIDIA_API_KEY=...
```

---

## Flutter Frontend

The NIS Protocol includes a full-featured Flutter desktop application.

```bash
cd ../NIS_Flutter_Chat/nis_chat_app
flutter run -d macos  # or windows, linux
```

**Features:**
- Agentic chat with natural language commands
- Real-time telemetry dashboard
- Robotics control panel
- BitNet status monitoring
- User authentication

---

## Modular Architecture

NIS Protocol v4.0 uses a modular route architecture for improved maintainability:

```
routes/
├── consciousness.py # V4.0 evolution, genesis, embodiment (28 endpoints)
├── protocols.py     # MCP, A2A, ACP integration (22 endpoints)
├── system.py        # Configuration, state, edge AI (18 endpoints)
├── monitoring.py    # Health, metrics (16 endpoints)
├── memory.py        # Persistent memory (15 endpoints)
├── utilities.py     # Cost, cache, templates, code exec (14 endpoints)
├── vision.py        # Image analysis, generation (12 endpoints)
├── agents.py        # Agent management (11 endpoints)
├── auth.py          # Authentication, user mgmt (11 endpoints)
├── v4_features.py   # V4.0 memory, self-modification (11 endpoints)
├── chat.py          # Chat endpoints (9 endpoints)
├── nvidia.py        # NVIDIA Inception, NeMo (8 endpoints)
├── voice.py         # STT, TTS, voice chat (7 endpoints)
├── llm.py           # LLM optimization, analytics (7 endpoints)
├── physics.py       # PINN validation, equations (6 endpoints)
├── bitnet.py        # BitNet training (6 endpoints)
├── robotics.py      # FK/IK, trajectory planning (5 endpoints)
├── research.py      # Deep research (4 endpoints)
├── unified.py       # Unified pipeline, autonomous (4 endpoints)
├── reasoning.py     # Collaborative reasoning (3 endpoints)
├── webhooks.py      # Webhook management (3 endpoints)
└── core.py          # Root, health check (2 endpoints)
```

**Benefits:**
- **Testability**: Each module can be tested independently
- **Maintainability**: Smaller, focused files (~10-30KB each)
- **Dependency Injection**: Clean separation of concerns
- **Documentation**: See `docs/organized/architecture/ROUTE_MIGRATION.md`

---

## Ecosystem

| Project | Description |
|---------|-------------|
| [NIS-AUTO](https://github.com/Organica-Ai-Solutions/NIS-AUTO) | Automotive AI systems |
| [NIS-DRONE](https://github.com/Organica-Ai-Solutions/NIS-DRONE) | Aerospace and drone control |
| [NIS-CITY](https://github.com/Organica-Ai-Solutions/NIS-CITY) | Smart city infrastructure |
| [NIS-X](https://github.com/Organica-Ai-Solutions/NIS-X) | Space exploration (NeurIPS 2025) |
| [AlphaCortex](https://github.com/Organica-Ai-Solutions/AlphaCortex) | Autonomous trading |

---

## Performance

| Metric | Value |
|--------|-------|
| API Endpoints | 236+ |
| Modular Routes | 222 (100% complete) |
| Response Time | <50ms avg |
| Uptime | 99.9% |
| Test Coverage | 100% critical paths |
| LLM Providers | 6 (OpenAI, Anthropic, DeepSeek, Google, NVIDIA, BitNet) |
| Agents | 14 specialized |
| Protocols | 4 (ACP, A2A, MCP, S2A) |

## Quick Links

| Resource | URL |
|----------|-----|
| **Dashboard** | http://localhost/static/dashboard.html |
| **Chat UI** | http://localhost/static/modern_chat.html |
| **Voice Chat** | http://localhost/static/chat_console.html |
| **API Docs** | http://localhost/docs |
| **Health** | http://localhost/health |

---

## License

Apache License 2.0

**Commercial licensing available.** Contact: diego.torres@organicaai.com

---

## Contact

- **Website**: [organicaai.com](https://organicaai.com)
- **GitHub**: [Organica-Ai-Solutions](https://github.com/Organica-Ai-Solutions)
- **Email**: contact@organicaai.com

---

<p align="center">
  <strong>NIS Protocol v4.0</strong><br/>
  Enterprise AI Operating System<br/>
  © 2024-2025 Organica AI Solutions
</p>
