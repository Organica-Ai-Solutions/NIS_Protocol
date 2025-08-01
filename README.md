# NIS Protocol v3.1

The NIS Protocol (Neural Integration System Protocol) is a structured framework for developing, testing, and deploying agent-based systems for generative simulation.

## 📋 Important Repository Note
This repository uses Git LFS (Large File Storage) for managing large model files. If you're cloning this repository, please make sure you have Git LFS installed:

```bash
# Install Git LFS
git lfs install

# Clone with LFS support
git lfs clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
```

The actual large model files (located in `models/bitnet/`) are not included directly in the repository. See `models/bitnet/README.md` for instructions on downloading these files.

![NIS Protocol Banner](assets/images_organized/mathematical_visuals/v3map.png)

**The NIS Protocol is a generative AI system that builds and runs physically-informed simulations. Based on a descriptive prompt, the system can generate a 3D model, simulate its performance under specified conditions, and produce a technical report on its viability (see `benchmarks/performance_validation.py` for validation).**

This is a closed-loop design and validation system that connects generative AI to a "world model" grounded in physical principles, allowing it to create and test systematic designs.

## Core Features

| Feature                       | Description                                                                                                                              | Visual                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Generative Simulation**     | Generate physically-informed 3D models and simulations of complex systems from natural language prompts. | ![Simulation Example](assets/images_organized/system_screenshots/usesExamples.png)                           |
| **Verifiable AI Pipeline**    | The Laplace → KAN → PINN pipeline, supported by NVIDIA Modulus, ensures that generated content is grounded in scientific principles (validated in `benchmarks/performance_validation.py`). | ![Verifiable AI Pipeline](assets/images_organized/mathematical_visuals/laplace+kan.png)                      |
| **Offline-First Capabilities** | The integrated BitNet model allows for fast, offline inference, enabling the system to operate without constant cloud connectivity.     | ![BitNet Integration](assets/images_organized/mathematical_visuals/mlp.png)                                |
| **Multi-Agent Architecture**  | A hierarchy of specialized agents work in coordination to design, simulate, analyze, and learn (see `system/docs/architecture/Agent_Architecture.md`).                                   | ![Agent Architecture](system/docs/diagrams/agent_hierarchy/communication_hierarchy.md)                      |

## System Architecture

The NIS Protocol is built on a hierarchical system of specialized agents that work together to achieve complex goals. This architecture promotes a clear separation of concerns, efficient communication, and a scalable system.

![Agent Hierarchy Diagram](system/docs/diagrams/agent_hierarchy/communication_hierarchy.md#nis-protocol-agent-communication-hierarchy)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- An environment file (`.env`) with the necessary API keys (see `environment-template.txt`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/NIS_Protocol.git
    cd NIS_Protocol
    ```
2.  **Set up the environment:**
    ```bash
    cp environment-template.txt .env
    # Add your API keys to the .env file
    ```
3.  **Start the system:**
    ```bash
    ./start.sh
    ```

The system is now running! You can access the API at `http://localhost:8000`.

## API Highlights

### Run a Generative Simulation
**Endpoint**: `POST /simulation/run`

Run the full design-simulation-analysis loop for a given concept.

**Request Body**:
```json
{
  "concept": "a drone wing that mimics a falcon's"
}
```

### Chat with the System
**Endpoint**: `POST /chat`

Engage in a dialogue with the NIS Protocol's reasoning agents.

**Request Body**:
```json
{
  "message": "Explain the significance of the Laplace transform in your pipeline.",
  "user_id": "test_user"
}
```

## The NIS Protocol v3.1: A New Paradigm

Version 3.1 represents a significant step forward, moving from a theoretical framework to a functional, production-ready system with well-defined capabilities.

- **From Theory to Implementation**: Where v3 laid the groundwork, v3.1 provides a working implementation. The Generative Simulation Engine is an operational feature.
- **Offline and Autonomous Operation**: The integration of the BitNet model provides a new level of autonomy, allowing the system to learn and operate without constant cloud connectivity.
- **Verifiable and Trustworthy by Design**: The PINN-based validation ensures that the system's outputs are scientifically sound and traceable to first principles.

The NIS Protocol v3.1 is an evolution in what is possible with generative AI, with a strong focus on verifiable and trustworthy results.
