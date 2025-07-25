# 🎮 How to Use the NIS Protocol

## 🎯 **Choose Your Use Case**

### **🔬 I'm a Researcher/Scientist**
**Goal**: Use NIS for scientific analysis with physics validation

```python
from src.agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent
from src.agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer

# Setup scientific pipeline
laplace = EnhancedLaplaceTransformer()
physics = EnhancedPINNPhysicsAgent()

# Analyze experimental data
frequency_data = laplace.transform(your_signal_data)
validated_result = physics.validate_physics_constraints(frequency_data)

print(f"Physics compliance: {validated_result.compliance_score}")
```

### **🤖 I'm Building an AI Application**
**Goal**: Integrate intelligent agents into your app

```python
from src.cognitive_agents.cognitive_system import CognitiveSystem

# Initialize the system
ai_system = CognitiveSystem()

# Process user input with intelligence
def handle_user_query(user_input):
    response = ai_system.process_input(
        text=user_input,
        generate_speech=False
    )
    
    return {
        "answer": response.response_text,
        "confidence": response.confidence,
        "reasoning": response.reasoning_chain
    }

# Example usage
result = handle_user_query("What's the weather like?")
```

### **🏭 I'm Deploying in Production**
**Goal**: Scalable, reliable AI system

```python
from src.infrastructure.integration_coordinator import InfrastructureCoordinator
import asyncio

async def production_setup():
    # Setup infrastructure
    coordinator = InfrastructureCoordinator()
    await coordinator.initialize()
    
    # Deploy with monitoring
    await coordinator.deploy_agent_cluster(
        agent_types=["reasoning", "memory", "coordination"],
        replicas=3,
        monitoring=True
    )
    
    return coordinator

# Run in production
coordinator = asyncio.run(production_setup())
```

## 📊 **Common Usage Patterns**

### **Pattern 1: Question Answering with Confidence**
```python
from src.cognitive_agents.cognitive_system import CognitiveSystem

cognitive_system = CognitiveSystem()

def intelligent_qa(question):
    response = cognitive_system.process_input(question)
    
    if response.confidence > 0.8:
        return f"✅ {response.response_text}"
    elif response.confidence > 0.5:
        return f"⚠️ {response.response_text} (uncertain)"
    else:
        return f"❌ I'm not confident about this answer"

# Examples
print(intelligent_qa("What is 2+2?"))          # ✅ Mathematical answer
print(intelligent_qa("What will happen tomorrow?"))  # ❌ Too uncertain
```

### **Pattern 2: Multi-Agent Problem Solving**
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

### **Pattern 3: Scientific Data Processing**
```python
from src.agents.signal_processing.laplace_processor import LaplaceSignalProcessor
from src.agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent

def analyze_scientific_data(time_series_data):
    # Signal processing
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

### **Pattern 4: Consciousness-Monitored Decision Making**
```python
from src.agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent

def make_confident_decision(options, criteria):
    conscious_agent = EnhancedConsciousAgent()
    
    # Evaluate options with self-awareness
    decision_analysis = conscious_agent.evaluate_decision(
        options=options,
        criteria=criteria,
        require_confidence_threshold=0.7
    )
    
    if decision_analysis.confidence >= 0.7:
        return {
            "decision": decision_analysis.recommended_option,
            "confidence": decision_analysis.confidence,
            "reasoning": decision_analysis.reasoning_chain,
            "status": "confident"
        }
    else:
        return {
            "decision": None,
            "confidence": decision_analysis.confidence,
            "reasoning": "Need more information",
            "status": "uncertain",
            "suggestions": decision_analysis.information_needed
        }

# Example usage
options = ["Option A", "Option B", "Option C"]
criteria = ["cost", "safety", "efficiency"]
decision = make_confident_decision(options, criteria)
```

## 🔧 **Configuration Examples**

### **Basic Configuration**
```python
# config/basic_setup.py
NIS_CONFIG = {
    "agents": {
        "reasoning": {"enabled": True, "model": "enhanced"},
        "memory": {"enabled": True, "persistence": True},
        "consciousness": {"enabled": True, "confidence_threshold": 0.7}
    },
    "infrastructure": {
        "kafka": {"enabled": False},  # Disable for basic usage
        "redis": {"enabled": False},  # Disable for basic usage  
        "monitoring": {"enabled": True, "level": "info"}
    }
}
```

### **Production Configuration**
```python
# config/production_setup.py
PRODUCTION_CONFIG = {
    "agents": {
        "reasoning": {"enabled": True, "model": "enhanced", "replicas": 3},
        "memory": {"enabled": True, "persistence": True, "backup": True},
        "consciousness": {"enabled": True, "confidence_threshold": 0.8},
        "coordination": {"enabled": True, "load_balancing": True}
    },
    "infrastructure": {
        "kafka": {"enabled": True, "cluster_size": 3},
        "redis": {"enabled": True, "cluster_mode": True},
        "monitoring": {"enabled": True, "level": "debug", "alerting": True}
    },
    "performance": {
        "caching": {"enabled": True, "ttl": 3600},
        "batching": {"enabled": True, "batch_size": 32},
        "async_processing": {"enabled": True, "workers": 8}
    }
}
```

## 🎯 **Integration Examples**

### **FastAPI Integration**
```python
from fastapi import FastAPI
from src.cognitive_agents.cognitive_system import CognitiveSystem

app = FastAPI()
cognitive_system = CognitiveSystem()

@app.post("/intelligence/process")
async def process_intelligence(request: dict):
    response = cognitive_system.process_input(
        text=request["input"],
        context=request.get("context", {})
    )
    
    return {
        "response": response.response_text,
        "confidence": response.confidence,
        "processing_time": response.processing_time,
        "agents_involved": response.agents_used
    }

@app.get("/intelligence/status")
async def get_system_status():
    return {
        "agents": cognitive_system.get_agent_status(),
        "memory_usage": cognitive_system.get_memory_stats(),
        "performance": cognitive_system.get_performance_metrics()
    }
```

### **Jupyter Notebook Integration**
```python
# Install in notebook
# !pip install -r requirements.txt

from src.cognitive_agents.cognitive_system import CognitiveSystem
import matplotlib.pyplot as plt

# Initialize system
cognitive_system = CognitiveSystem()

# Interactive analysis
def analyze_and_plot(data, question):
    # Process with NIS
    response = cognitive_system.process_input(
        text=question,
        data=data
    )
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"Analysis: {response.response_text}")
    plt.xlabel(f"Confidence: {response.confidence:.2f}")
    plt.show()
    
    return response

# Example usage in notebook
import numpy as np
data = np.random.randn(100).cumsum()
result = analyze_and_plot(data, "What pattern do you see in this data?")
```

## 🏆 **Success Metrics**

### **How to Know It's Working**
```python
from src.agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent

def system_health_check():
    conscious_agent = EnhancedConsciousAgent()
    
    # Run self-diagnostic
    health_report = conscious_agent.perform_system_diagnostic()
    
    print("🔍 NIS Protocol Health Check")
    print("=" * 40)
    print(f"✅ Agent Coordination: {health_report.agent_coordination}")
    print(f"✅ Memory System: {health_report.memory_system}")
    print(f"✅ Physics Validation: {health_report.physics_validation}")
    print(f"✅ Consciousness Level: {health_report.consciousness_level}")
    print(f"🎯 Overall Health: {health_report.overall_score}/100")
    
    return health_report.overall_score > 80

# Run health check
if system_health_check():
    print("🎉 System is healthy and ready!")
else:
    print("⚠️ System needs attention")
```

## 🚀 **Next Steps**

### **Beginner Path**
1. ✅ Run the 5-minute demo
2. 📖 Read [Getting Started](GETTING_STARTED.md)
3. 🎮 Try [Basic Examples](../examples/README.md)
4. 🔧 Explore [API Reference](API_Reference.md)

### **Advanced Path**
1. 🏗️ Study [Architecture Guide](docs/README.md)
2. 🔬 Implement custom agents
3. 🚀 Deploy to production
4. 📈 Monitor and optimize

### **Researcher Path**
1. 📊 Explore [Mathematical Foundation](../assets/images_organized/mathematical_visuals/)
2. ⚛️ Study physics validation
3. 🧠 Experiment with consciousness
4. 📝 Contribute to research

---

**💡 Need help? Check the [FAQ](faq.html) or open a [GitHub Issue](https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues)** 