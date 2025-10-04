# 🚀 NIS Protocol Quick Start Guide

Get started with the NIS Protocol in 5 minutes!

## 📦 Installation

### Basic Installation
```bash
pip install nis-protocol
```

### Full Installation (recommended)
```bash
pip install nis-protocol[full]
```

### Development Installation
```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
pip install -e .[dev]
```

## 🔑 Configuration

Create a `.env` file with your API keys:

```bash
# LLM Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 🎯 Usage Examples

### 1. Simple LLM Chat

```python
from nis_protocol import NISCore

# Initialize
nis = NISCore()

# Get response
response = nis.get_llm_response("Hello, NIS!")
print(response['content'])
```

### 2. Autonomous Mode (Recommended!)

```python
import asyncio
from nis_protocol import NISCore

async def main():
    nis = NISCore()
    
    # Process autonomously - system decides what to do!
    result = await nis.process_autonomously(
        "Calculate fibonacci(10)"
    )
    
    print(f"Intent: {result['intent']}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Response: {result['response']}")

asyncio.run(main())
```

### 3. Start Web Server

```bash
# Using CLI
nis-server

# Or using Python
from nis_protocol.server import run
run(host='0.0.0.0', port=8000)
```

Then visit:
- http://localhost:8000/console (Classic Chat UI)
- http://localhost:8000/modern-chat (Modern Chat UI)
- http://localhost:8000/docs (API Documentation)

### 4. Command Line Interface

```bash
# Display info
nis-protocol info

# Quick start guide
nis-protocol quickstart

# Send chat message
nis-protocol chat "Hello, world!"

# Process autonomously
nis-protocol autonomous "Calculate 255 * 387"

# Run system test
nis-protocol test

# Verify installation
nis-protocol verify
```

### 5. Interactive Agent Mode

```bash
nis-agent
```

This starts an interactive chat session where you can talk to the autonomous AI agent.

## 🤖 Autonomous AI Features

The NIS Protocol automatically detects intent and selects tools:

### Supported Intents

| Intent | Example | Tools Used |
|--------|---------|------------|
| Code Execution | "Run fibonacci code" | Runner, LLM |
| Physics Validation | "Validate bouncing ball" | Physics PINN, LLM |
| Deep Research | "Research quantum computing" | Research Engine, LLM |
| Web Search | "Search latest AI news" | Web Search, LLM |
| Math Calculation | "Calculate 255 * 387" | Calculator, LLM |
| File Operations | "Save this to file" | File System, LLM |
| Visualization | "Plot a graph" | Visualization, LLM |

### Example: Autonomous Processing

```python
import asyncio
from nis_protocol import NISCore

async def demo():
    nis = NISCore()
    
    # The system automatically:
    # 1. Detects intent (CODE_EXECUTION)
    # 2. Selects tools (Runner + LLM)
    # 3. Executes code
    # 4. Returns results
    
    result = await nis.process_autonomously(
        "Run python code to calculate the 15th fibonacci number"
    )
    
    print(f"🎯 Intent: {result['intent']}")
    print(f"🔧 Tools: {', '.join(result['tools_used'])}")
    print(f"💬 Response: {result['response']}")
    print(f"💭 Reasoning: {result['reasoning']}")

asyncio.run(demo())
```

## 🏗️ Integration Examples

### Flask Integration

```python
from flask import Flask, jsonify, request
from nis_protocol import NISCore
import asyncio

app = Flask(__name__)
nis = NISCore()

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    result = asyncio.run(nis.process_autonomously(message))
    return jsonify(result)

if __name__ == '__main__':
    app.run()
```

### FastAPI Integration

```python
from fastapi import FastAPI
from nis_protocol import NISCore

app = FastAPI()
nis = NISCore()

@app.post("/chat")
async def chat(message: str):
    result = await nis.process_autonomously(message)
    return result

# Run with: uvicorn main:app
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
from nis_protocol import NISCore
import asyncio

nis = NISCore()

def chat_view(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        result = asyncio.run(nis.process_autonomously(message))
        return JsonResponse(result)
```

## 🔧 Advanced Usage

### Custom Configuration

```python
from nis_protocol import NISCore

config = {
    'llm_provider': 'openai',
    'temperature': 0.7,
    'max_tokens': 2000,
}

nis = NISCore(config=config)
```

### Using Specific LLM Provider

```python
from nis_protocol import LLMProvider

llm = LLMProvider()

# Use specific provider
response = await llm.generate_response(
    messages=[{"role": "user", "content": "Hello"}],
    requested_provider='anthropic'  # or 'openai', 'google', etc.
)
```

### Smart Consensus (Multi-LLM)

```python
# Use Smart Consensus to query multiple LLMs
response = await llm.generate_response(
    messages=[{"role": "user", "content": "Complex question"}],
    requested_provider='smart'  # Uses consensus of multiple models
)
```

## 🎓 Domain-Specific Templates

### NIS-DRONE (Coming Soon)
```python
from nis_protocol import NISCore
from nis_protocol.plugins import DronePlugin

nis = NISCore()
nis.register_plugin(DronePlugin(
    sensors=['camera', 'lidar', 'gps'],
    actuators=['motors', 'servos']
))
```

### NIS-AUTO (Coming Soon)
```python
from nis_protocol import NISCore
from nis_protocol.plugins import VehiclePlugin

nis = NISCore()
nis.register_plugin(VehiclePlugin(
    obd_port='/dev/ttyUSB0',
    sensors=['engine', 'transmission', 'brakes']
))
```

### NIS-CITY (Coming Soon)
```python
from nis_protocol import NISCore
from nis_protocol.plugins import SmartCityPlugin

nis = NISCore()
nis.register_plugin(SmartCityPlugin(
    iot_devices=['traffic', 'lighting', 'sensors']
))
```

## 🧪 Testing

```bash
# Run built-in system test
nis-protocol test

# Verify installation
nis-protocol verify

# Run unit tests (if you cloned the repo)
pytest tests/
```

## 📚 Documentation

- **Full Documentation**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/system/docs
- **Whitepaper**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/blob/main/system/docs/NIS_Protocol_V3_Whitepaper.md
- **API Reference**: http://localhost:8000/docs (after starting server)
- **Examples**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/examples

## 🐛 Troubleshooting

### Import Error
```bash
pip install nis-protocol[full]
```

### Missing API Keys
Create a `.env` file with your API keys (see Configuration section above).

### LLM Provider Not Working
```bash
# Verify installation
nis-protocol verify

# Check which providers are available
python -c "from nis_protocol import NISCore; nis = NISCore(); print(nis.llm_provider)"
```

### Server Won't Start
```bash
# Check if port is available
lsof -i :8000

# Try different port
nis-protocol server --port 8080
```

## 💡 Tips & Best Practices

1. **Always use autonomous mode** when possible - it's smarter!
2. **Use Smart Consensus** for important decisions
3. **Enable logging** for debugging:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```
4. **Set environment variables** for production:
   ```bash
   export OPENAI_API_KEY=sk-...
   export LOG_LEVEL=INFO
   ```

## 🔗 Next Steps

- Read the [Full Documentation](https://github.com/Organica-Ai-Solutions/NIS_Protocol)
- Join our [Community](https://github.com/Organica-Ai-Solutions/NIS_Protocol/discussions)
- Check out [Examples](https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/examples)
- Explore domain-specific templates (NIS-DRONE, NIS-AUTO, NIS-CITY)

## 🆘 Support

- **GitHub Issues**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues
- **Email**: contact@organicaai.com
- **Website**: https://www.organicaai.com

---

**Happy Building with NIS Protocol!** 🚀

*Built with ❤️ by Organica AI Solutions*

