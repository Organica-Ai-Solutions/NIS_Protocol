# Getting Started with the NIS Protocol v3.1

This guide provides a streamlined path to deploying and interacting with the NIS Protocol. The recommended method uses Docker for a simple, one-command setup.

## 1. Prerequisites

- **Docker and Docker Compose:** Ensure Docker is installed and running on your system. [Install Docker](https://docs.docker.com/get-docker/)
- **Git:** Required to clone the repository.
- **API Keys:** At least one LLM provider key (OpenAI, Anthropic, DeepSeek, or Google) is required for the system to function.

## 2. Quick Start: Docker Deployment (Recommended)

This is the fastest and most reliable way to get the entire NIS Protocol system running.

### Step 1: Clone the Repository

Open your terminal and clone the project:
```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
```

### Step 2: Configure API Keys

The system requires API keys to connect to Large Language Models. A template file `.env~` is provided.

**Copy the template to create your own `.env` file:**
```bash
cp .env~ .env
```

Now, open the `.env` file with your preferred text editor and add your API key(s).

```dotenv
# .env - LLM Provider API Keys (at least one is required)
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

### Step 3: Run the System

Execute the startup script. This single command will build the Docker images, start all necessary services (including Redis, Kafka, and the main application), and run the system.

```bash
./scripts/start.sh
```

The first time you run this, it may take a few minutes to download and build the Docker images. Subsequent starts will be much faster.

### Step 4: Verify the System is Running

Once the script finishes, you can verify that all services are running correctly by checking the health endpoint:

```bash
curl http://localhost:8000/health
```

You should see a `"status": "healthy"` response, along with a list of registered agents and available tools.

## 3. Interacting with the Protocol

With the system running, you can now interact with it through its API endpoints.

### Example: Chat Endpoint

Send a message to the chat endpoint to get a response from the protocol's integrated LLM.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the core principles of the NIS Protocol."}'
```

### Example: Asynchronous Chat (for streaming)

```bash
curl -X POST http://localhost:8000/chat/async \
  -H "Content-Type: application/json" \
  -d '{"message": "Describe the Laplace to KAN pipeline."}'
```

## 4. System Management

A few simple scripts are provided in the `scripts/` directory to manage the system.

- **To Stop the System:**
  ```bash
  ./scripts/stop.sh
  ```

- **To Reset the System (stops services and removes containers):**
  ```bash
  ./scripts/reset.sh
  ```

## 5. Next Steps

Congratulations! The NIS Protocol is now running on your machine.

- **Explore the API:** View the full, interactive API documentation (provided by Swagger UI) by opening [http://localhost:8000/docs](http://localhost:8000/docs) in your web browser.
- **Read the Architecture Guide:** For a deeper understanding of the system's design, see the `ARCHITECTURE.md` file.
- **Review the Whitepaper:** For a high-level overview of the protocol's vision and capabilities, see the `NIS_Protocol_V3_Whitepaper.md`. 