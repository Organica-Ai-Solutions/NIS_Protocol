# NIS Protocol v4.0

**Physics-Validated AI Orchestration for Robotics, Edge Systems, and Multi-LLM Applications**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-BSL-blue.svg)](LICENSE)

NIS Protocol is a production-oriented AI orchestration platform that connects large language models, real-time perception, physics validation, and physical robot control through a unified FastAPI backend.

The project is designed for systems where AI output must do more than generate text. It must reason over sensor data, call tools, validate actions, and control hardware safely.

---

## What NIS Protocol Does

NIS Protocol provides an application layer for embodied AI workflows:

- **Natural language to robot action**
- **Camera perception to grounded scene understanding**
- **LLM planning to executable tool calls**
- **Physics-aware validation for robotics and simulation**
- **Edge deployment on Raspberry Pi with optional H100/GPU acceleration**
- **REST and WebSocket APIs for dashboards, agents, and real-time clients**

It is not positioned as AGI or a consciousness system. It is a practical engineering stack for AI-assisted robotics and edge automation.

---

## Core Capabilities

### Multi-LLM Gateway

NIS Protocol includes a provider abstraction for multiple model backends:

- OpenAI
- Anthropic
- Google Gemini
- DeepSeek
- NVIDIA NIM
- Kimi / Moonshot
- Local BitNet-style edge models

The gateway supports provider selection, fallback behavior, model metadata, and unified chat-style request handling.

### Robotics Control

The robotics layer supports direct integration with physical robot hardware and simulation-friendly workflows:

- Hiwonder xArm 1S control
- Forward and inverse kinematics
- Named arm poses
- Gripper control
- Trajectory planning
- Safety-aware execution paths
- REST endpoints for direct command dispatch

### Vision and Perception

NIS Protocol integrates camera input and visual grounding:

- Pi Camera / USB camera support
- Snapshot and MJPEG stream endpoints
- YOLO-based object detection
- Grounding DINO integration paths
- Scene context extraction for robot planning

### Physics Validation

The platform includes modules for physics-aware reasoning and validation:

- Physics-Informed Neural Network workflows
- Laplace transform utilities
- KAN reasoning components
- Constraint checks for physically plausible outputs
- Simulation-oriented validation hooks

### H100 / GPU-Accelerated Reasoning

For heavier robotics reasoning workloads, NIS Protocol can proxy planning and perception tasks to GPU services:

- Cosmos Reason2-style planning endpoints
- Video/world prediction service integration
- VLA-style image-to-action workflows
- Remote GPU service health checks and fallbacks

### Real-Time Interfaces

The API exposes real-time communication channels for interactive clients:

- Agentic WebSocket for tool-driven interaction
- Voice WebSocket for STT / LLM / TTS workflows
- Camera streaming
- Dashboard-ready health and telemetry endpoints

---

## Architecture

```text
Client / Dashboard / Operator
        |
        v
FastAPI Backend
        |
        +-- LLM Gateway
        |     +-- OpenAI / Anthropic / Gemini / DeepSeek / NVIDIA / BitNet
        |
        +-- Vision Layer
        |     +-- Camera snapshots
        |     +-- YOLO / grounding models
        |
        +-- Planning Layer
        |     +-- Cosmos / H100 services
        |     +-- Tool routing
        |
        +-- Physics Layer
        |     +-- PINN / KAN / Laplace validation
        |
        +-- Robotics Layer
              +-- xArm control
              +-- FK / IK
              +-- trajectory execution
```

---

## Repository Layout

```text
.
├── main.py                  # Full FastAPI application entry point
├── main_pi.py               # Raspberry Pi / edge-optimized entry point
├── routes/                  # Modular API route definitions
│   ├── chat.py              # Chat and LLM endpoints
│   ├── robotics.py          # Robotics APIs
│   ├── physics.py           # Physics validation APIs
│   ├── vision.py            # Vision and image APIs
│   ├── voice.py             # Voice and real-time audio APIs
│   ├── cookoff.py           # End-to-end robot demo flows
│   └── openclaw.py          # Robot tool gateway
├── src/
│   ├── llm/                 # Provider abstraction and LLM routing
│   ├── physics/             # Physics, PINN, KAN, Laplace modules
│   ├── agents/              # Specialized agent implementations
│   ├── neurolinux/          # Edge/robot hardware integration
│   └── chat/                # Persistent chat memory components
├── static/                  # Browser dashboards and console UIs
├── tests/                   # Smoke, integration, and robotics tests
├── tools/                   # Deployment, diagnostics, calibration, ops
└── docs/                    # Technical documentation
```

---

## Quick Start

### Requirements

- Python 3.8+
- FastAPI / Uvicorn runtime
- Optional: Raspberry Pi 5 for edge robotics deployment
- Optional: NVIDIA GPU or remote H100 service for accelerated perception/planning
- Optional: physical xArm-compatible robot

### Install

```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Full Backend

```bash
python main.py
```

### Run the Edge / Raspberry Pi Backend

```bash
python main_pi.py
```

The API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Environment Configuration

Create a local `.env` file and configure only the providers or services you need.

Common variables:

```bash
DEFAULT_LLM_PROVIDER=anthropic
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=
NVIDIA_API_KEY=

H100_REASON_URL=http://localhost:8100
H100_PREDICT_URL=http://localhost:8200
H100_VLA_URL=http://localhost:8500

BITNET_PRELOAD=false
BITNET_MODEL_PATH=
```

Do not commit API keys or machine-specific secrets.

---

## Example API Calls

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What can you do?"}'
```

### Camera Snapshot

```bash
curl http://localhost:8000/camera/snapshot
```

### Robotics Command

```bash
curl -X POST http://localhost:8000/arm/home
```

### Cookoff / Vision-to-Action Demo

```bash
curl -X POST http://localhost:8000/cookoff/demo \
  -H "Content-Type: application/json" \
  -d '{"task":"pick up the object and place it in the bin","execute_arm":true}'
```

---

## Deployment Profiles

### Local Development

Use `main.py` for full API exploration and desktop/server workflows.

### Raspberry Pi Edge Runtime

Use `main_pi.py` for lightweight startup, camera access, robot control, and remote GPU service integration.

### GPU-Assisted Robotics

Run perception and planning models on a remote GPU service and point the Pi or backend to those service URLs through environment variables.

---

## Engineering Principles

NIS Protocol is built around a few practical principles:

- **Real tools over simulated claims**
- **Hardware actions must be traceable**
- **Model output should be validated before execution**
- **Edge devices should delegate heavy inference when necessary**
- **Fallback states must be explicit**
- **Robotics interfaces should be auditable and safe by default**

---

## Testing

Run the available smoke and integration tests from the repository root:

```bash
pytest tests
```

For hardware-specific tests, verify that the corresponding services and devices are online before running the test suite.

---

## Security Notes

Robotics and system-control endpoints can affect real hardware. Before deploying outside a controlled environment:

- Enable authentication and authorization
- Restrict shell/system endpoints
- Use network allowlists
- Keep API keys in environment variables or a secrets manager
- Log hardware actions
- Prefer simulation mode for untrusted inputs

---

## Status

NIS Protocol is an active research and engineering project. Some modules are production-oriented, while others are experimental or hardware-specific. The most mature flows are the LLM gateway, Pi edge runtime, camera/vision integration, and xArm robotics control path.

---

## License

This project is licensed under the **Business Source License (BSL)**. It is free for research and educational use. Commercial licensing is available through Organica AI Solutions.

---

## Maintainer

**Organica AI Solutions**  
Building practical AI systems for robotics, edge infrastructure, and intelligent automation.
