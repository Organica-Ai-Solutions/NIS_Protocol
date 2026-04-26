# NIS Protocol v4.0

**Multi-LLM Gateway & Physics-Validated Robotics Control**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-BSL-blue.svg)](LICENSE)

NIS Protocol is a high-performance FastAPI orchestration shell designed to bridge Large Language Models with physical robotics. It provides a unified gateway for multiple LLM providers, real-time physics validation via Physics-Informed Neural Networks (PINNs), and direct hardware control for robotic arms (e.g., xArm/ROS2).

## Core Capabilities

- **Multi-LLM Gateway:** Unified API routing across Anthropic, OpenAI, Google, DeepSeek, NVIDIA NIM, and local offline models (BitNet).
- **Physics Validation:** Built-in PINN solvers, KAN, and Laplace transforms to validate generated trajectories and constraints against real-world physics.
- **Robotics Control:** Forward and Inverse Kinematics (FK/IK) solvers, trajectory planning, and direct hardware integration for the Hiwonder xArm 1S.
- **Multimodal Workflows:** Live vision processing (YOLOv8, Grounding DINO) and real-time voice (TTS/STT) via WebSockets.
- **Modular Architecture:** Clean dependency injection, modular FastAPI routers, and robust error handling designed for edge deployments (e.g., Raspberry Pi 5).

## Architecture Overview

The system operates as a FastAPI monolith coordinating specialized modules:

- src/physics/: Mathematical and physical validation (PINN, KAN, Laplace).
- src/robotics/: Kinematics, path planning, and hardware interfaces.
- src/llm/: Provider SDK wrappers and routing logic.
- outes/: Modular endpoints for REST and WebSocket clients.
- main_pi.py / main.py: Entry points tailored for edge (Raspberry Pi) and cloud/desktop environments.

## Quick Start

### Prerequisites
- Python 3.8+
- Optional: NVIDIA GPU for local Cosmos/YOLO acceleration

### Installation

`ash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### Running the Server

For edge deployment (Raspberry Pi):
`ash
python main_pi.py
`

For full desktop/server deployment:
`ash
python main.py
`

Access the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

## License

This project is licensed under the **Business Source License (BSL)**. It is free for research and educational purposes. Commercial licensing is available upon request.
