# 📚 NIS Protocol Examples

This directory contains practical examples showing how to use the NIS Protocol in different scenarios.

## 🎯 Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)
**Difficulty**: Beginner  
**Topics**: Initialization, Simple LLM calls, Provider selection

Learn the basics of NIS Protocol:
- Initialize NISCore
- Get LLM responses
- Use specific providers

```bash
python examples/01_basic_usage.py
```

---

### 2. Autonomous Mode (`02_autonomous_mode.py`)
**Difficulty**: Intermediate  
**Topics**: Intent detection, Tool selection, Autonomous execution

See the autonomous AI in action:
- Automatic intent recognition
- Smart tool selection
- Multi-tool orchestration
- Reasoning transparency

```bash
python examples/02_autonomous_mode.py
```

---

### 3. FastAPI Integration (`03_fastapi_integration.py`)
**Difficulty**: Intermediate  
**Topics**: Web API, FastAPI, RESTful endpoints

Integrate NIS Protocol into a FastAPI application:
- RESTful API endpoints
- Chat and autonomous modes
- Health checks
- API documentation

```bash
# Run the server
uvicorn examples.03_fastapi_integration:app --reload

# Then visit http://localhost:8000/docs
```

---

## 🚀 Running Examples

### Prerequisites

```bash
# Install NIS Protocol
pip install nis-protocol[full]

# Or if you cloned the repo
pip install -e .[full]
```

### Set up environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Run an example

```bash
# Basic usage
python examples/01_basic_usage.py

# Autonomous mode
python examples/02_autonomous_mode.py

# FastAPI integration
cd examples
uvicorn 03_fastapi_integration:app
```

---

## 📖 Coming Soon

More examples will be added covering:

- [ ] **Flask Integration** - Using NIS Protocol with Flask
- [ ] **Django Integration** - Django REST framework example
- [ ] **Streaming Responses** - Real-time streaming chat
- [ ] **Multi-Agent Systems** - Coordinating multiple agents
- [ ] **Custom Plugins** - Creating domain-specific adapters
- [ ] **NIS-DRONE Template** - Drone-specific implementation
- [ ] **NIS-AUTO Template** - Vehicle diagnostics example
- [ ] **NIS-CITY Template** - Smart city IoT integration
- [ ] **Edge Deployment** - BitNet low-power example
- [ ] **Production Deployment** - Docker, Kubernetes, monitoring

---

## 💡 Tips

1. **Start with Example 1** to understand basics
2. **Try Example 2** to see autonomous AI in action
3. **Use Example 3** as template for your own API

---

## 🐛 Troubleshooting

### ImportError
```bash
pip install nis-protocol[full]
```

### Missing API Keys
Make sure you have a `.env` file with your API keys.

### Port Already in Use
```bash
# Find what's using the port
lsof -i :8000

# Or use a different port
uvicorn examples.03_fastapi_integration:app --port 8080
```

---

## 🔗 Learn More

- [Quick Start Guide](../QUICKSTART.md)
- [Full Documentation](../system/docs/)
- [GitHub Repository](https://github.com/Organica-Ai-Solutions/NIS_Protocol)

---

**Built with ❤️ by Organica AI Solutions**

