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
- **192 API Endpoints** - Comprehensive REST API coverage

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
| API Endpoints | 192 |
| Response Time | <50ms avg |
| Uptime | 99.9% |
| Test Coverage | 100% critical paths |

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
