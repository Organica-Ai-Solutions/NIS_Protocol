# NIS Protocol Source Directory

This directory contains the core architecture and reference implementation of the Neural Intelligence System (NIS) Protocol v4.0.4.

## Core Architecture

The \src/\ directory is built around the **NeuroKernel v2**, an autonomous orchestration engine designed to coordinate multiple LLMs, physical robotics (OpenClaw), edge deployments (NeuroLinux), and simulation physics.

### Directory Structure

- **\/core\**: The central orchestration pipeline (NeuroKernel v2). Includes \AuditChain\ (tamper-proof logs), \LoopGuard\ (autonomous circuit breakers), \DriveScheduler\, and \SkillLoader\.
- **\/agents\**: Specialized agent implementations (e.g., VisionAgent, DocumentAnalysisAgent, ResearchAgent) and training scaffolding (BitNet).
- **\/neurolinux\**: Edge AI operating system bridges. Contains \pc_bridge.py\, edge deployment routines, and Hiwonder xArm drivers for Raspberry Pi.
- **\/llm\**: Multi-provider LLM gateway supporting Anthropic, OpenAI, DeepSeek, Google, NVIDIA NIM, and local offline models (BitNet).
- **\/calibration\**: Advanced camera and robotics calibration systems (video calibrators, intrinsic matrices).
- **\/skills\**: Dynamically loaded \SKILL.md\ behaviors injected into agents at runtime.
- **\/kinematics\**: Inverse and Forward kinematics engines for precise robotics control.
- **\/memory\**: Persistent conversation and context storage systems.

## Key Subsystems

### NeuroKernel v2
The heart of the autonomous loop. It provides an execution cycle of \scan\ -> \skills\ -> \loop\ -> \xecute\ -> \udit\. It ensures agents operate safely within hardware limits.

### OpenClaw & NeuroLinux
The translation layer that converts abstract LLM intents into deterministic, physics-validated hardware signals for edge devices (like the Raspberry Pi controlling a robotic arm).

### Local BitNet Fallback
A high-efficiency 1-bit LLM integration ensuring the orchestration layer remains operational even during network outages.

## Getting Started
See the main repository \README.md\ and \docs/\ directory for deployment instructions and API contracts.
