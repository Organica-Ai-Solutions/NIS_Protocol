# NIS Protocol - Getting Started Guide

## What is the NIS Protocol?

The **NIS Protocol** is a production-ready framework for building adaptive, distributed AI systems with:

- **Signal Processing Pipeline** - Mathematical signal processing to natural language generation
- **Multi-Agent Coordination** - Specialized agents working together through comprehensive coordination
- **Physics-Informed Validation** - Scientific constraint checking and validation
- **Performance Monitoring** - Real-time system performance and quality monitoring

## Quick Start Guide

### Docker Deployment (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol

# 2. Configure LLM API keys
cat > .env << EOF
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Infrastructure Configuration
DATABASE_URL=postgresql://nis_user:nis_password_2025@postgres:5432/nis_protocol_v3
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
REDIS_HOST=redis
REDIS_PORT=6379
NIS_ENV=development
LOG_LEVEL=INFO
EOF

# Edit configuration file with your API keys
nano .env

# API key sources:
# OpenAI: https://platform.openai.com/api-keys
# Anthropic: https://console.anthropic.com/
# DeepSeek: https://platform.deepseek.com/
# Google: https://makersuite.google.com/app/apikey

# 3. Deploy infrastructure
./start.sh

# 4. Verify deployment
curl http://localhost/health
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Analyze this data for patterns"}'

# 5. Expected URLs:
# 🌐 Main API: http://localhost/
# 📊 Dashboard: http://localhost/dashboard/
# 📖 Docs: http://localhost/docs
```

> **⚠️ Important**: The system requires at least one LLM provider API key to function. We recommend starting with OpenAI or Anthropic.

### **📋 Manual Setup (Developers)**
<details>
<summary>Click to expand manual installation</summary>

```bash
# 1. Clone and setup
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
pip install -r requirements.txt

# 2. Configure external services (PostgreSQL, Kafka, Redis)
# See DOCKER_README.md for service configuration details

# 3. Run the demo
python examples/complete_agi_foundation_demo.py
```

</details>

### **What You'll See**

**🐳 Docker Deployment:**
```
[NIS-V3] Starting NIS Protocol v3 Complete System...
[SUCCESS] Docker and Docker Compose are available
[SUCCESS] All required directories are ready
[NIS-V3] Starting core infrastructure services...
[SUCCESS] PostgreSQL is ready
[SUCCESS] Redis is ready
[SUCCESS] Core infrastructure is ready
[NIS-V3] Starting NIS Protocol v3 application...
[SUCCESS] NIS Protocol v3 application started successfully
[SUCCESS] Reverse proxy started successfully

🌐 Service URLs:
  • Main API:          http://localhost/
  • API Documentation: http://localhost/docs
  • Health Check:      http://localhost/health
  • Monitoring:        http://localhost/dashboard/
```

**🧪 API Testing:**
```bash
$ curl http://localhost/health
{
  "status": "healthy",
  "uptime": 45.2,
  "components": {
    "cognitive_system": "healthy",
    "infrastructure": "healthy", 
    "consciousness": "healthy",
    "dashboard": "healthy"
  }
}

$ curl -X POST http://localhost/process -H "Content-Type: application/json" \
  -d '{"text": "Analyze this data for patterns"}'
{
  "response_text": "Pattern detected: Linear growth with 95% confidence.",
  "confidence": 0.89,
  "processing_time": 0.34,
  "agent_insights": {
    "laplace_analysis": "Frequency domain analysis complete",
    "kan_reasoning": "Symbolic function extracted", 
    "pinn_validation": "Physics constraints satisfied"
  },
  "consciousness_state": {
    "awareness_level": 0.85,
    "meta_cognitive_state": "active"
  }
}
```

**🎉 Complete AGI Infrastructure Ready!**

## 🏗️ **Architecture Overview**

### **Core Pipeline: Laplace→KAN→PINN→LLM**
```
📊 Raw Input → 🌊 Laplace Transform → 🧠 KAN Reasoning → ⚛️ Physics Validation → 💬 LLM Output
     ↓              ↓                    ↓                    ↓                   ↓
Signal Data    Frequency Domain    Symbolic Functions    Physics Check    Natural Language
```

### **agent coordination framework**
```
🎯 Input Agent ←→ 🧠 Reasoning Agent ←→ 💭 Consciousness Agent
     ↓                    ↓                     ↓
📊 Vision Agent ←→ 🔄 Coordination Agent ←→ 💾 Memory Agent
     ↓                    ↓                     ↓  
🎬 Action Agent ←→ 📡 Communication Agent ←→ 📈 Learning Agent
```

## 🔧 **Installation Options**

### **Option 1: Quick Setup (Recommended)**
```bash
# Minimal installation for core functionality
pip install -r requirements-minimal.txt

# Test core system
python utilities/final_100_test.py
# Expected: ✅ All tests pass
```

### **Option 2: Full Features**
```bash
# Complete installation with all features
pip install -r requirements.txt

# Optional: Enhanced infrastructure
pip install -r requirements_enhanced_infrastructure.txt
```

### **Option 3: Development Setup**
```bash
# For developers and researchers
pip install -r requirements_tech_stack.txt

# Install development tools
pip install pytest black isort flake8
```

## 🎮 **Usage Examples**

### **Basic Usage: Cognitive processing (implemented) (implemented)**
```python
from src.cognitive_agents.cognitive_system import CognitiveSystem

# Initialize the system
cognitive_system = CognitiveSystem()

# Process input through neural intelligence pipeline
response = cognitive_system.process_input(
    text="Analyze this data for patterns",
    generate_speech=False
)

print(f"Response: {response.response_text}")
print(f"Confidence: {response.confidence}")
```

### **advanced Usage: Multi-Agent Coordination**
```python
from src.agents.coordination.coordinator_agent import CoordinatorAgent
from src.agents.enhanced_agent_base import EnhancedAgentBase

async def solve_complex_problem(problem_description):
    # Setup specialized agents
    coordinator = CoordinatorAgent()
    agents = [
        EnhancedAgentBase("analysis_agent"),
        EnhancedAgentBase("reasoning_agent"), 
        EnhancedAgentBase("validation_agent")
    ]
    
    # Coordinate solution
    solution = await coordinator.coordinate_agents(
        task=problem_description,
        agents=agents,
        require_consensus=True
    )
    
    return solution

# Example usage
solution = asyncio.run(solve_complex_problem("Design a sustainable energy system"))
```

### **Physics-Informed processing (implemented) (implemented)**
```python
from src.agents.signal_processing.laplace_processor import LaplaceSignalProcessor
from src.agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent

def analyze_scientific_data(time_series_data):
    # Signal processing (implemented) (implemented)
    processor = LaplaceSignalProcessor()
    frequency_features = processor.compute_laplace_transform(time_series_data)
    
    # Symbolic reasoning
    kan_agent = EnhancedKANReasoningAgent()
    symbolic_function = kan_agent.extract_symbolic_function(frequency_features)
    
    return {
        "frequency_analysis": frequency_features,
        "symbolic_function": symbolic_function,
        "mathematical_form": symbolic_function.get_mathematical_expression()
    }

# Example with experimental data
import numpy as np
time_data = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))  # 5Hz sine wave
result = analyze_scientific_data(time_data)
print(f"Detected function: {result['mathematical_form']}")
```

## 📊 **Real-World Applications**

### **1. Scientific Research**
```python
# Example: Analyze experimental data
result = cognitive_system.process_input(
    text="Analyze temperature sensor data for anomalies",
    data=sensor_readings
)
# ✅ Physics validation ensures scientific accuracy
```

### **2. Autonomous Systems**
```python
# Example: Robot navigation with physics constraints
navigation_plan = coordinator.plan_with_physics_constraints(
    start_position=(0, 0),
    target_position=(10, 10),
    obstacles=detected_obstacles
)
# ✅ PINN validation ensures safe movement
```

### **3. Decision Support**
```python
# Example: Multi-criteria decision making
decision = cognitive_system.make_decision(
    options=["Option A", "Option B", "Option C"],
    criteria=["cost", "safety", "efficiency"],
    weights=[0.3, 0.4, 0.3]
)
# ✅ Consciousness Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) ([health tracking](src/infrastructure/integration_coordinator.py)) ensures confident decisions
```

## 🎯 **Choose Your Path**

### **🔬 Researcher/Scientist**
```bash
# Focus on scientific applications
cd examples/
python test_agi_v2_implementation.py  # Core AI capabilities
python data_flow_analysis.py          # System analysis
```

### **🤖 Developer/Engineer**  
```bash
# Focus on integration and development
cd examples/
python enhanced_llm_config_demo.py    # LLM integration
python tech_stack_integration_demo.py # Full stack demo
```

### **🚀 Production User**
```bash
# Focus on deployment and scaling
cd examples/
python complete_agi_foundation_demo.py    # Production demo
./setup_repo.sh                           # Production setup
```

## 🔍 **System Validation**

### **Test Your Installation**
```bash
# 1. Core functionality test
python utilities/final_100_test.py
# Expected: ✅ All tests pass

# 2. Agent integration test  
python tests/integration/practical_integration_test.py
# Expected: ✅ Agent coordination working

# 3. Performance validation
python benchmarks/performance_validation.py
# Expected: ✅ Performance metrics within expected ranges
```

### **Troubleshooting**
```bash
# Common issues and solutions

# Issue: Import errors
pip install --upgrade -r requirements.txt

# Issue: Memory errors  
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Issue: Agent coordination fails
python utilities/test_self_audit_agents.py  # Check agent health
```

## 🏆 **What Makes NIS Protocol Unique?**

### **🧠 Mathematical Interpretability**
- **KAN Networks**: Understand WHY decisions are made (validated in tests/kan_validation.py)
- **Symbolic Reasoning**: Extract mathematical functions from data
- **Transparency**: No black-box processing (implemented) (implemented)

### **⚛️ Physics-Informed Intelligence**
- **Scientific Validation**: All outputs checked against physics laws
- **Constraint Satisfaction**: Ensures realistic and safe results
- **Conservation Laws**: Respects fundamental principles

### **💭 Consciousness Integration**
- **Self-Awareness**: System knows its own confidence levels
- **Meta-Cognition**: Thinks about its own thinking
- **Uncertainty Quantification**: Honest about what it doesn't know

## 📚 **Next Steps**

### **Beginner Path**
1. ✅ Run the 5-minute demo
2. 📖 Read [Getting Started](GETTING_STARTED.md)
3. 🎮 Try [Basic Examples](../examples/README.md)
4. 🔧 Explore [API Reference](API_Reference.md)

### **advanced Path**
1. 🏗️ Study [Architecture Guide](docs/README.md)
2. 🔬 Implement custom agents
3. 🚀 Deploy to production
4. 📈 Monitor and optimize

### **Researcher Path**
1. 📊 Explore [Mathematical Foundation](../assets/images_organized/mathematical_visuals/)
2. ⚛️ Study physics validation
3. 🧠 Experiment with consciousness
4. 📝 Acknowledge current limitations

---

**💡 Need help? Check the [FAQ](faq.html) or open a [GitHub Issue](https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues)** 