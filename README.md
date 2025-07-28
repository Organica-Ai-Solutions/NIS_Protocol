# NIS Protocol v3
## A Framework for Verifiable AI Systems

<div align="center">
  <img src="assets/images_organized/nis-protocol-logov1.png" alt="NIS Protocol v3 Logo" width="400"/>
</div>

---

## Overview

The NIS Protocol is a framework for building verifiable AI systems that are grounded in scientific principles. It provides a structured approach to building complex, multi-agent systems that are transparent, interpretable, and reliable.

The core of the NIS Protocol is a four-stage scientific processing pipeline:

**Laplace → KAN → PINN → LLM**

1.  **Laplace Transform:** Signal processing and frequency domain analysis.
2.  **Kolmogorov-Arnold Networks (KAN):** Interpretable, spline-based function approximation.
3.  **Physics-Informed Neural Networks (PINN):** Scientific validation and constraint enforcement.
4.  **Large Language Model (LLM):** Natural language generation and enhancement.

This pipeline ensures that all outputs are not only intelligent but also scientifically sound and mathematically verifiable.

---

## Getting Started

### Prerequisites

*   **Docker** and **Docker Compose**
*   **Git** for cloning the repository
*   API keys for your preferred LLM providers (OpenAI, Anthropic, etc.)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
    cd NIS_Protocol
    ```

2.  **Set up your environment:**
    *   Copy the `.env~` file to `.env`.
    *   Add your LLM provider API keys to the `.env` file.

3.  **Start the system:**
    ```bash
    ./start.sh
    ```

That's it! The entire NIS Protocol system is now running in Docker.

### Accessing the System

*   **API:** `http://localhost:8000`
*   **API Docs:** `http://localhost:8000/docs`
*   **Health Check:** `http://localhost:8000/health`

---

## Project Structure

The project is organized into the following directories:

*   `src/`: The core source code for the NIS Protocol.
*   `docs/`: All project documentation.
*   `benchmarks/`: Performance benchmarks and testing scripts.
*   `dev/`: Development-related files, such as old code and test scripts.
*   `scripts/`: Helper scripts for managing the project.
*   `config/`: Configuration files for the various services.
*   `assets/`: Images and other static assets.

---

## Documentation

*   **[Technical Whitepaper](docs/NIS_Protocol_V3_Whitepaper.md):** A detailed explanation of the NIS Protocol and its scientific foundations.
*   **[Drone Project](docs/drone/):** Documentation for the drone-based implementation of the NIS Protocol.
*   **[API Reference](docs/API_Reference.md):** A complete reference for the NIS Protocol API.

---

## Contributing

We welcome contributions from the community. To get started, please see our [contribution guidelines](CONTRIBUTING.md).

---

## License

The NIS Protocol is licensed under the [Business Source License 1.1](LICENSE_BSL). It is free for research, education, and other non-commercial uses. A commercial license is required for production environments.
