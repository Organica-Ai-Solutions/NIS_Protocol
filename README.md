<div align="center">
  <img src="https://via.placeholder.com/150x150.png?text=NIS+Protocol" alt="NIS Protocol Logo" width="120" height="120" style="border-radius: 20px;">
  <h1>Neuro-Inspired System (NIS) Protocol v4.0.4</h1>
  <p><strong>Multi-LLM Gateway & Physics-Validated Robotics Control</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Robotics](https://img.shields.io/badge/Hardware-Hiwonder_xArm-FF6C37?logo=robotics)](https://github.com/Organica-Ai-Solutions/NIS_Protocol)
  [![License](https://img.shields.io/badge/License-BSL-blue.svg)](LICENSE)
</div>

<br/>

NIS Protocol is a high-performance FastAPI orchestration shell designed to bridge **Large Language Models (LLMs)** with **physical edge robotics**. It provides a unified gateway for multiple AI providers, real-time physics validation via Physics-Informed Neural Networks (PINNs), and deterministic hardware control for robotic arms on embedded Linux (NeuroLinux).

---

## 🌟 Core Capabilities

- 🧠 **Multi-LLM Gateway**: Unified, load-balanced API routing across Anthropic, OpenAI, DeepSeek, Google, NVIDIA NIM, and fallback 1-bit local models (BitNet).
- 🦾 **Robotics Control (OpenClaw)**: Complete Forward and Inverse Kinematics (FK/IK) solvers, trajectory planning, and direct hardware integration for the Hiwonder xArm 1S over Raspberry Pi 5.
- ⚛️ **Physics Validation**: Built-in PINN solvers, Kolmogorov-Arnold Networks (KAN), and Laplace transforms to validate AI-generated physical trajectories against real-world constraints before execution.
- 👁️ **Multimodal Workflows**: Live vision processing (YOLOv8, Grounding DINO) and real-time wake-word voice communication (VibeVoice TTS/Whisper STT) via high-frequency WebSockets.
- 🛡️ **Autonomous Safeties (NeuroKernel v2)**: Tamper-proof audit logging (AuditChain), loop circuit breakers (LoopGuard), and sandboxed execution environments.

---

## 🏗️ Architecture Overview

The system operates as a FastAPI monolith coordinating specialized modules, highly optimized for deployment across H100 clusters and Raspberry Pi edge devices:

| Directory | Subsystem | Description |
| :--- | :--- | :--- |
| 🔀 **src/core/** | **NeuroKernel v2** | Central orchestration pipeline. Manages audit logs, loop safeties, and dynamic skill injection (SKILL.md). |
| 🔌 **src/neurolinux/** | **Edge OS Bridge** | Direct hardware control and I2C/CAN bus routing for the Raspberry Pi and Hiwonder xArm. |
| 📐 **src/physics/** | **Modulus Engine** | Mathematical validation (PINN, KAN, Laplace) ensuring generated trajectories obey real-world physics. |
| 💬 **src/llm/** | **Provider Gateway** | Multi-provider SDK wrappers and routing logic supporting cloud endpoints and local BitNet. |
| 🌐 **outes/** | **API Endpoints** | Modular, load-balanced FastAPI endpoints for REST, WebSockets, and high-frequency edge telemetry. |

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the hardened production dependencies:

`ash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (Includes recent CVE security fixes)
pip install -r requirements.txt
`

### 2. Configuration
Copy the environment template and add your API keys:
`ash
cp .env.example .env
# Edit .env with your keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
`

### 3. Running the Server

**For Edge Deployment (Raspberry Pi / NeuroLinux):**
`ash
python main_pi.py
`

**For Cloud/Desktop Deployment (H100 / DGX / Local PC):**
`ash
python main.py
`

Access the interactive Swagger API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Hardware & Edge Deployment

The dev/cookoff-pcbridge toolset provides extensive scripts for deploying NIS Protocol to distributed edge networks:
- **	ools/deploy/deploy_to_pi.py**: Automated payload sync to Raspberry Pi.
- **scripts/h100_heavy/**: Multi-agent benchmarking, VLA model fine-tuning, and heavy physics simulations.
- **	ools/ops/pi_status.py**: Real-time SSH and status telemetry bridging the PC, Pi, and H100 GPU cluster.

---

## 📝 Version 4.0.4 - The Integration Release

The latest release introduces **NeuroKernel v2**, the **OpenClaw** robotic integration, and extensive edge deployment tools for controlling Raspberry Pi devices and H100 GPU clusters natively. The underlying system has been completely restructured to remove undocumented dependencies and outdated "AGI" terminology in favor of hardened, deterministic orchestration pipelines.

---

## ⚖️ License

This project is licensed under the **Business Source License (BSL)**. 
It is entirely free for research, academic, and educational purposes. Commercial licensing and enterprise support are available upon request.
