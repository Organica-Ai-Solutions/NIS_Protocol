# 🔗 NIS Protocol v3 - Complete API Reference

> **Comprehensive API documentation for Docker-deployed NIS Protocol v3 with full agent and model integration**

---

## 📋 **Table of Contents**

1. [🐳 **REST API (Primary Interface)**](#-rest-api-primary-interface)
2. [🧠 **Agent Connection APIs**](#-agent-connection-apis)
3. [🎼 **Model Orchestration APIs**](#-model-orchestration-apis)
4. [🔬 **Core Scientific Pipeline APIs**](#-core-scientific-pipeline-apis)
5. [💭 **Consciousness & Meta-Cognition APIs**](#-consciousness--meta-cognition-apis)
6. [🛡️ **Monitoring & Health APIs**](#️-monitoring--health-apis)
7. [🐍 **Python Integration APIs**](#-python-integration-apis)
8. [📝 **Examples & Integration**](#-examples--integration)

---

## 🐳 **REST API (Primary Interface)**

**Base URL**: `http://localhost/` (after `./start.sh`)

> **⚠️ Prerequisites**: Configure your LLM provider API keys in `.env` file before starting. See [Getting Started Guide](GETTING_STARTED.md#quick-start-with-docker-recommended) for setup instructions.

### **🔍 Health & Status**

#### **GET `/health`**
System health check and component status.

```bash
curl http://localhost/health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime": 123.45,
  "components": {
    "cognitive_system": "healthy",
    "infrastructure": "healthy",
    "consciousness": "healthy",
    "dashboard": "healthy"
  },
  "metrics": {
    "uptime_seconds": 123.45,
    "memory_usage_mb": 2048.0,
    "active_agents": 8,
    "response_time_ms": 12.5
  }
}
```

#### **GET `/consciousness/status`**
Consciousness agent status and awareness level.

```bash
curl http://localhost/consciousness/status
```

**Response:**
```json
{
  "agent_status": "active",
  "awareness_level": 0.85,
  "meta_cognitive_processes": [
    "introspection",
    "self_reflection", 
    "performance_monitoring"
  ],
  "last_update": 1705123456,
  "processing_queue_size": 3
}
```

#### **GET `/infrastructure/status`**
Infrastructure component health and performance.

```bash
curl http://localhost/infrastructure/status
```

**Response:**
```json
{
  "kafka": "connected",
  "redis": "connected",
  "postgresql": "connected",
  "message_queue_size": 12,
  "cache_hit_ratio": 0.95,
  "active_topics": ["nis-consciousness", "nis-goals", "nis-coordination"],
  "performance_metrics": {
    "latency_ms": 12.5,
    "throughput_ops_per_sec": 150,
    "error_rate": 0.001
  }
}
```

#### **GET `/metrics`**
Comprehensive system metrics.

```bash
curl http://localhost/metrics
```

### **🧠 Intelligence Processing**

#### **POST `/process`**
Primary intelligence processing endpoint.

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Analyze quantum consciousness in AI systems",
    "generate_speech": false,
    "context": {
      "operation": "comprehensive_analysis",
      "depth": "deep",
      "include_consciousness": true,
      "enable_physics_validation": true
    }
  }'
```

**Request Schema:**
```json
{
  "text": "string (required)",
  "generate_speech": "boolean (optional, default: false)",
  "context": {
    "operation": "string (optional)",
    "depth": "string (optional: shallow|medium|deep)",
    "include_consciousness": "boolean (optional)",
    "enable_physics_validation": "boolean (optional)",
    "agent_preferences": "array (optional)",
    "model_preferences": "object (optional)"
  }
}
```

**Response Schema:**
```json
{
  "response_text": "string",
  "confidence": "number (0-1)",
  "processing_time": "number (seconds)",
  "agent_insights": {
    "goal_adaptation": "object",
    "domain_generalization": "object", 
    "autonomous_planning": "object",
    "kan_reasoning": "object",
    "pinn_validation": "object",
    "laplace_analysis": "object"
  },
  "consciousness_state": {
    "awareness_level": "number (0-1)",
    "meta_cognitive_state": "string",
    "introspection_score": "number (0-1)",
    "ethical_evaluation": "object"
  },
  "model_usage": {
    "primary_model": "string",
    "fallback_used": "boolean",
    "total_tokens": "number",
    "cost_estimate": "number"
  }
}
```

### **🎯 Specialized Intelligence Operations**

#### **POST `/process` - Goal Generation**
Generate autonomous goals and objectives.

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Generate research goals for quantum AI",
    "context": {
      "operation": "goal_generation",
      "domain": "research", 
      "time_horizon": "6_months",
      "priority": "high"
    }
  }'
```

#### **POST `/process` - Domain Transfer**
Cross-domain knowledge transfer and analogical reasoning.

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apply physics principles to biological systems",
    "context": {
      "operation": "domain_transfer",
      "source_domain": "physics",
      "target_domain": "biology",
      "concepts": ["energy_conservation", "force_dynamics"]
    }
  }'
```

#### **POST `/process` - Strategic Planning**
Multi-step strategic planning with resource optimization.

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Create comprehensive AI safety research plan",
    "context": {
      "operation": "strategic_planning",
      "goal": "ai_safety_research",
      "resources": ["team_of_5", "12_month_timeline"],
      "constraints": ["budget_limited", "ethical_compliance"]
    }
  }'
```

### **🔧 Administration**

#### **POST `/admin/restart`**
Restart system services (requires admin access).

```bash
curl -X POST http://localhost/admin/restart \
  -H "Content-Type: application/json" \
  -d '{
    "services": ["infrastructure", "agents"],
    "force": false
  }'
```

---

## 🧠 **Agent Connection APIs**

### **Agent Coordination Patterns**

#### **Direct Agent Communication**
Use the `/process` endpoint with specific agent targeting:

```bash
# Target specific agents in sequence
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Research quantum consciousness",
    "context": {
      "operation": "agent_coordination",
      "coordination_mode": "sequential",
      "agent_chain": [
        "consciousness",
        "domain_generalization", 
        "kan_reasoning",
        "planning"
      ]
    }
  }'
```

#### **Parallel Agent Processing**
Execute multiple agents in parallel:

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Analyze AI consciousness from multiple perspectives",
    "context": {
      "operation": "parallel_analysis",
      "coordination_mode": "parallel",
      "agents": {
        "consciousness": {"depth": "deep"},
        "physics": {"validation_level": "strict"},
        "reasoning": {"analysis_type": "structural"}
      }
    }
  }'
```

---

## 🎼 **Model Orchestration APIs**

### **Multi-Model Processing**

#### **Cognitive Function Assignment**
Route different cognitive functions to specialized models:

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Analyze ethical implications of quantum AI",
    "context": {
      "operation": "multi_model_analysis",
      "cognitive_functions": {
        "reasoning": {"model": "gpt-4", "temperature": 0.3},
        "creativity": {"model": "claude-3-opus", "temperature": 0.8},
        "consciousness": {"model": "claude-3-opus", "temperature": 0.7},
        "ethics": {"model": "gpt-4", "temperature": 0.4}
      }
    }
  }'
```

#### **Model Performance Optimization**
Request optimal model selection based on criteria:

```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Complex mathematical proof",
    "context": {
      "operation": "optimized_processing",
      "optimization_criteria": {
        "priority": "quality",
        "max_latency": 10.0,
        "budget_constraint": 0.25
      }
    }
  }'
```

---

## 🔬 **Core Scientific Pipeline APIs**

### **🌊 Enhanced Laplace Transformer**

#### **Class: `EnhancedLaplaceTransformer`**
**Location**: `src/agents/signal_processing/enhanced_laplace_transformer.py`

```python
from src.agents.signal_processing.enhanced_laplace_transformer import (
    EnhancedLaplaceTransformer, TransformType, SignalQuality
)

# Initialize
transformer = EnhancedLaplaceTransformer(
    agent_id="laplace_001",
    max_frequency=1000.0,      # Maximum frequency to analyze (Hz)
    num_points=2048,           # Number of s-plane grid points
    enable_self_audit=True     # Enable integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
)
```

#### **Primary Methods**

##### **`compute_laplace_transform()`**
```python
def compute_laplace_transform(
    self,
    signal_data: np.ndarray,           # Input signal
    time_vector: np.ndarray,           # Time vector (must be uniform)
    transform_type: TransformType = TransformType.LAPLACE_NUMERICAL,
    validate_result: bool = True        # Perform validation
) -> LaplaceTransformResult:
```

**Parameters:**
- `signal_data`: Input time-domain signal (numpy array)
- `time_vector`: Corresponding time vector (uniform spacing required)
- `transform_type`: Type of transform (`LAPLACE_NUMERICAL`, `LAPLACE_UNILATERAL`, `LAPLACE_BILATERAL`)
- `validate_result`: Whether to validate results via inverse transform

**Returns:** `LaplaceTransformResult` with analysis

 with implemented coverage**Example:**
```python
import numpy as np

# Create test signal
t = np.linspace(0, 2, 1000)
signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-t)

# Transform
result = transformer.compute_laplace_transform(signal, t)

print(f"Processing time: {result.metrics.processing_time:.4f}s")
print(f"Reconstruction error: {result.reconstruction_error:.6f}")
print(f"Signal quality: {result.quality_assessment.value}")
print(f"Poles found: {len(result.poles)}")
```

##### **`compress_signal()`**
```python
def compress_signal(
    self,
    transform_result: LaplaceTransformResult,
    compression_target: float = 0.1,    # Target compression ratio
    quality_threshold: float = 0.95     # Minimum quality to maintain
) -> CompressionResult:
```

**Parameters:**
- `transform_result`: Result from `compute_laplace_transform()`
- `compression_target`: Target compression ratio (0-1, smaller = more compression)
- `quality_threshold`: Minimum reconstruction quality to maintain

**Returns:** `CompressionResult` with compression metrics

##### **`get_performance_summary()`**
```python
def get_performance_summary(self) -> Dict[str, Any]:
```

**Returns:** performance statistics and metrics

 with implemented coverage---

### **🧮 Enhanced Reasoning Agent**

#### **Class: `EnhancedKANReasoningAgent`**
**Location**: `src/agents/reasoning/enhanced_kan_reasoning_agent.py`

```python
from src.agents.reasoning.enhanced_kan_reasoning_agent import (
    EnhancedKANReasoningAgent, ReasoningType, FunctionComplexity
)

# Initialize
kan_agent = EnhancedKANReasoningAgent(
    agent_id="kan_reasoning_001",
    input_dim=8,                       # Input feature dimension
    hidden_dims=[16, 12, 8],          # Hidden layer dimensions
    output_dim=4,                     # Output dimension
    enable_self_audit=True            # Enable integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
)
```

#### **Primary Methods**

##### **`process_laplace_input()`**
```python
def process_laplace_input(
    self,
    laplace_result: Dict[str, Any]     # Results from Laplace transformer
) -> SymbolicResult:
```

**Parameters:**
- `laplace_result`: Dictionary containing Laplace transform results

**Returns:** `SymbolicResult` with symbolic function extraction

**Example:**
```python
# Process Laplace results
symbolic_result = kan_agent.process_laplace_input(laplace_result)

print(f"Symbolic expression: {symbolic_result.symbolic_expression}")
print(f"Confidence: {symbolic_result.confidence_score:.3f}")
print(f"Complexity: {symbolic_result.mathematical_complexity.value}")
print(f"Approximation error: {symbolic_result.approximation_error:.6f}")
```

##### **`get_performance_summary()`**
```python
def get_performance_summary(self) -> Dict[str, Any]:
```

**Returns:** Network performance and symbolic extraction statistics

---

### **⚖️ Enhanced PINN Physics Agent**

#### **Class: `EnhancedPINNPhysicsAgent`**
**Location**: `src/agents/physics/enhanced_pinn_physics_agent.py`

```python
from src.agents.physics.enhanced_pinn_physics_agent import (
    EnhancedPINNPhysicsAgent, PhysicsLaw, ViolationSeverity
)

# Initialize
pinn_agent = EnhancedPINNPhysicsAgent(
    agent_id="pinn_physics_001",
    enable_self_audit=True,           # Enable integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
    strict_mode=False                 # Strict physics enforcement
)
```

#### **Primary Methods**

##### **`validate_kan_output()`**
```python
def validate_kan_output(
    self,
    reasoning_result: Dict[str, Any]        # Results from reasoning agent
) -> PINNValidationResult:
```

**Parameters:**
- `reasoning_result`: Dictionary containing symbolic results

**Returns:** `PINNValidationResult` with physics compliance assessment

**Example:**
```python
# Validate reasoning output against physics
physics_result = pinn_agent.validate_kan_output(kan_result)

print(f"Physics compliance: {physics_result.physics_compliance_score:.3f}")
print(f"Violations detected: {len(physics_result.violations)}")
print(f"Conservation scores: {physics_result.conservation_scores}")
```

##### **`get_performance_summary()`**
```python
def get_performance_summary(self) -> Dict[str, Any]:
```

**Returns:** Physics validation performance and compliance statistics

---

### **🎼 Enhanced Scientific Coordinator**

#### **Class: `EnhancedScientificCoordinator`**
**Location**: `src/meta/enhanced_scientific_coordinator.py`

```python
from src.meta.enhanced_scientific_coordinator import (
    EnhancedScientificCoordinator, PipelineStage, ProcessingPriority
)

# Initialize
coordinator = EnhancedScientificCoordinator(
    coordinator_id="scientific_coordinator",
    enable_self_audit=True,           # Enable integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
    enable_auto_correction=True       # Enable auto-correction
)
```

#### **Setup Methods**

##### **`register_pipeline_agent()`**
```python
def register_pipeline_agent(
    self,
    stage: PipelineStage,             # Pipeline stage
    agent: Any                        # Agent instance
):
```

**Parameters:**
- `stage`: Pipeline stage (`LAPLACE_TRANSFORM`, `KAN_REASONING`, `PINN_VALIDATION`, etc.)
- `agent`: Initialized agent instance

#### **Primary Methods**

##### **`execute_scientific_pipeline()`**
```python
async def execute_scientific_pipeline(
    self,
    input_data: Dict[str, Any],       # Input data for pipeline
    execution_id: Optional[str] = None,
    priority: ProcessingPriority = ProcessingPriority.NORMAL
) -> PipelineExecutionResult:
```

**Parameters:**
- `input_data`: Dictionary containing signal data and metadata
- `execution_id`: Optional execution identifier
- `priority`: Processing priority level

**Returns:** `PipelineExecutionResult` with pipeline results

 with implemented coverage**Example:**
```python
# implemented pipeline setup and execution
coordinator.register_pipeline_agent(PipelineStage.LAPLACE_TRANSFORM, transformer)
coordinator.register_pipeline_agent(PipelineStage.KAN_REASONING, kan_agent)
coordinator.register_pipeline_agent(PipelineStage.PINN_VALIDATION, pinn_agent)

# Execute pipeline
input_data = {
    'signal_data': signal,
    'time_vector': t,
    'description': 'Test signal processing'
}

result = await coordinator.execute_scientific_pipeline(input_data)

print(f"Overall accuracy: {result.overall_accuracy:.3f}")
print(f"Physics compliance: {result.physics_compliance:.3f}")
print(f"Processing time: {result.total_processing_time:.4f}s")
```

---

## 🧠 **Consciousness & Meta-Cognition APIs**

### **🧠 Enhanced Conscious Agent**

#### **Class: `EnhancedConsciousAgent`**
**Location**: `src/agents/consciousness/enhanced_conscious_agent.py`

```python
from src.agents.consciousness.enhanced_conscious_agent import (
    EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
)

# Initialize
conscious_agent = EnhancedConsciousAgent(
    agent_id="conscious_001",
    reflection_interval=60.0,         # Reflection interval in seconds
    enable_self_audit=True,           # Enable integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
    consciousness_level=ConsciousnessLevel.ENHANCED
)
```

#### **Primary Methods**

##### **`perform_introspection()`**
```python
def perform_introspection(
    self,
    reflection_type: ReflectionType,   # Type of reflection
    target_agent_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> IntrospectionResult:
```

**Parameters:**
- `reflection_type`: Type of reflection (`PERFORMANCE_REVIEW`, `ERROR_ANALYSIS`, `INTEGRITY_ASSESSMENT`, etc.)
- `target_agent_id`: Specific agent to reflect on (None for self-reflection)
- `context`: Additional context for reflection

**Returns:** `IntrospectionResult` with analysis

 with implemented coverage**Example:**
```python
# Perform system health check
result = conscious_agent.perform_introspection(
    ReflectionType.SYSTEM_HEALTH_CHECK
)

print(f"Confidence: {result.confidence:.3f}")
print(f"integrity score 100/100 ([audit results](nis-integrity-toolkit/audit-scripts/))")
print(f"Findings: {result.findings}")
print(f"Recommendations: {result.recommendations}")
```

##### **`register_agent_for_monitoring()`**
```python
def register_agent_for_monitoring(
    self,
    agent_id: str,                    # Agent identifier
    agent_metadata: Dict[str, Any]    # Agent metadata
):
```

##### **`start_continuous_reflection()`** / **`stop_continuous_reflection()`**
```python
def start_continuous_reflection(self):
def stop_continuous_reflection(self):
```

##### **`get_consciousness_summary()`**
```python
def get_consciousness_summary(self) -> Dict[str, Any]:
```

**Returns:** consciousness status and metrics

 with implemented coverage---

### **🔍 Meta-Cognitive Processor**

#### **Class: `MetaCognitiveProcessor`**
**Location**: `src/agents/consciousness/meta_cognitive_processor.py`

```python
from src.agents.consciousness.meta_cognitive_processor import MetaCognitiveProcessor

# Initialize (this agent is already operational)
processor = MetaCognitiveProcessor()
```

#### **Key Methods (Existing)**

##### **`analyze_cognitive_process()`**
```python
def analyze_cognitive_process(
    self,
    process_data: Dict[str, Any],     # Process data to analyze
    context: Dict[str, Any]           # Context information
) -> CognitiveAnalysis:
```

##### **`audit_self_output()`**
```python
def audit_self_output(
    self,
    output_text: str,                 # Text output to audit
    context: str = ""                 # Additional context
) -> Dict[str, Any]:
```

##### **`auto_correct_self_output()`**
```python
def auto_correct_self_output(
    self,
    output_text: str                  # Text to correct
) -> Dict[str, Any]:
```

---

### **👁️ Introspection Manager**

#### **Class: `IntrospectionManager`**
**Location**: `src/agents/consciousness/introspection_manager.py`

```python
from src.agents.consciousness.introspection_manager import IntrospectionManager

# Initialize (this agent is already operational)
manager = IntrospectionManager()
```

#### **Key Methods (Existing)**

##### **`audit_introspection_output()`**
```python
def audit_introspection_output(
    self,
    introspection_result: Dict[str, Any],
    context: str = ""
) -> Dict[str, Any]:
```

##### **`generate_system_integrity_report()`**
```python
def generate_system_integrity_report(self) -> Dict[str, Any]:
```

---

## 🛡️ **Integrity & monitoring ([health tracking](src/infrastructure/integration_coordinator.py)) APIs**

### **🔍 Self-Audit Engine**

#### **Module: `self_audit`**
**Location**: `src/utils/self_audit.py`

```python
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

# The self_audit_engine is a singleton instance ready to use
```

#### **Primary Functions**

##### **`audit_text()`**
```python
def audit_text(
    text: str,                        # Text to audit
    context: str = ""                 # Additional context
) -> List[IntegrityViolation]:
```

**Example:**
```python
# Audit text for integrity violations
text = "AI system delivers measured results with measured performance"
violations = self_audit_engine.audit_text(text)

for violation in violations:
    print(f"{violation.severity}: {violation.text}")
    print(f"Suggested: {violation.suggested_replacement}")
```

##### **`auto_correct_text()`**
```python
def auto_correct_text(
    text: str                         # Text to correct
) -> Tuple[str, List[IntegrityViolation]]:
```

##### **`get_integrity_score()`**
```python
def get_integrity_score(
    text: str                         # Text to score
) -> float:
```

**Returns:** integrity score 100/100 ([audit results](nis-integrity-toolkit/audit-scripts/))

##### **`generate_integrity_report()`**
```python
def generate_integrity_report() -> Dict[str, Any]:
```

**Returns:** integrity system report

 with implemented coverage---

### **📊 Integrity Metrics**

#### **Module: `integrity_metrics`**
**Location**: `src/utils/integrity_metrics.py`

```python
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)
```

#### **Primary Functions**

##### **`calculate_confidence()`**
```python
def calculate_confidence(
    data_quality: float,              # Data quality score (0-1)
    model_complexity: float,          # Model complexity factor (0-1)
    validation_score: float,          # Validation accuracy (0-1)
    confidence_factors: ConfidenceFactors
) -> float:
```

**Returns:** Calculated confidence score (0-1)

##### **`create_default_confidence_factors()`**
```python
def create_default_confidence_factors() -> ConfidenceFactors:
```

**Returns:** Default confidence calculation factors

---

## 🔧 **Utility & Helper APIs**

### **📊 Data Structures**

#### **Common Result Types**

##### **`LaplaceTransformResult`**
```python
@dataclass
class LaplaceTransformResult:
    s_values: np.ndarray              # Complex frequency grid
    transform_values: np.ndarray      # F(s) transform values
    original_signal: np.ndarray       # Input signal
    time_vector: np.ndarray           # Time vector
    transform_type: TransformType     # Transform type used
    poles: np.ndarray                 # System poles
    zeros: np.ndarray                 # System zeros
    residues: np.ndarray              # Partial fraction residues
    convergence_region: Tuple[float, float]
    metrics: SignalMetrics            # Performance metrics
    quality_assessment: SignalQuality
    reconstruction_error: float       # Validation error
    frequency_accuracy: float         # Frequency domain accuracy
    phase_accuracy: float             # Phase preservation
```

##### **`SymbolicResult`**
```python
@dataclass
class SymbolicResult:
    symbolic_expression: sp.Expr     # Mathematical expression
    confidence_score: float          # Extraction confidence
    mathematical_complexity: FunctionComplexity
    approximation_error: float       # L2 error vs original
    function_domain: Tuple[float, float]
    function_range: Tuple[float, float]
    continuity_verified: bool
    differentiability_verified: bool
    spline_coefficients: np.ndarray
    grid_points: np.ndarray
    basis_functions_used: int
    processing_time: float
    memory_usage: int
    validation_score: float
    reasoning_steps: List[str]
    intermediate_expressions: List[sp.Expr]
```

##### **`PINNValidationResult`**
```python
@dataclass
class PINNValidationResult:
    physics_compliance_score: float  # Overall compliance (0-1)
    conservation_scores: Dict[str, float]
    violations: List[PhysicsViolation]
    processing_time: float
    memory_usage: int
    validation_confidence: float
    auto_correction_applied: bool
    corrected_function: Optional[sp.Expr]
    correction_enhancement: float
    constraint_evaluations: Dict[str, float]
    physics_law_scores: Dict[PhysicsLaw, float]
    numerical_stability: float
    physics_recommendations: List[str]
    enhancement_suggestions: List[str]
```

---

## 🧪 **Integration Examples**

### **implemented Pipeline Example**
```python
import asyncio
import numpy as np
from src.meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator, PipelineStage
from src.agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
from src.agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent
from src.agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent

async def complete_pipeline_example():
    # Initialize coordinator
    coordinator = EnhancedScientificCoordinator(
        coordinator_id="example_pipeline",
        enable_self_audit=True
    )
    
    # Initialize agents
    laplace_agent = EnhancedLaplaceTransformer(
        agent_id="laplace_example",
        max_frequency=50.0,
        enable_self_audit=True
    )
    
    kan_agent = EnhancedKANReasoningAgent(
        agent_id="kan_example",
        input_dim=8,
        hidden_dims=[16, 8],
        output_dim=1,
        enable_self_audit=True
    )
    
    pinn_agent = EnhancedPINNPhysicsAgent(
        agent_id="pinn_example",
        enable_self_audit=True
    )
    
    # Register agents
    coordinator.register_pipeline_agent(PipelineStage.LAPLACE_TRANSFORM, laplace_agent)
    coordinator.register_pipeline_agent(PipelineStage.KAN_REASONING, kan_agent)
    coordinator.register_pipeline_agent(PipelineStage.PINN_VALIDATION, pinn_agent)
    
    # Create test signal
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t) + 0.1*np.random.randn(len(t))
    
    # Execute pipeline
    input_data = {
        'signal_data': signal,
        'time_vector': t,
        'description': 'Multi-frequency test signal with noise'
    }
    
    result = await coordinator.execute_scientific_pipeline(input_data)
    
    # Process results
    print(f"Pipeline Status: {result.status.value}")
    print(f"Overall Accuracy: {result.overall_accuracy:.3f}")
    print(f"Physics Compliance: {result.physics_compliance:.3f}")
    print(f"Processing Time: {result.total_processing_time:.4f}s")
    print(f"integrity score 100/100 ([audit results](nis-integrity-toolkit/audit-scripts/))")
    
    return result

# Run example
if __name__ == "__main__":
    result = asyncio.run(complete_pipeline_example())
```

### **Consciousness Integration Example**
```python
from src.agents.consciousness.enhanced_conscious_agent import (
    EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
)

def consciousness_example():
    # Initialize consciousness agent
    agent = EnhancedConsciousAgent(
        agent_id="consciousness_example",
        consciousness_level=ConsciousnessLevel.ENHANCED,
        enable_self_audit=True
    )
    
    # Register other agents for monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
    agent.register_agent_for_monitoring("laplace_001", {"type": "signal_processing"})
    agent.register_agent_for_monitoring("kan_001", {"type": "reasoning"})
    
    # Perform different types of introspection
    reflection_types = [
        ReflectionType.SYSTEM_HEALTH_CHECK,
        ReflectionType.INTEGRITY_ASSESSMENT,
        ReflectionType.PERFORMANCE_REVIEW
    ]
    
    results = {}
    for reflection_type in reflection_types:
        result = agent.perform_introspection(reflection_type)
        results[reflection_type.value] = result
        
        print(f"\n{reflection_type.value.replace('_', ' ').title()}:")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  integrity score 100/100 ([audit results](nis-integrity-toolkit/audit-scripts/))")
        print(f"  Recommendations: {len(result.recommendations)}")
    
    # Start continuous monitoring ([health tracking](src/infrastructure/integration_coordinator.py))
    agent.start_continuous_reflection()
    
    # Get summary
    summary  with implemented coverage= agent.get_consciousness_summary()
    print(f"\nConsciousness Summary:")
    print(f"  Consciousness Level: {summary['consciousness_level']}")
    print(f"  Total Reflections: {summary['consciousness_metrics']['total_reflections']}")
    print(f"  Success Rate: {summary['consciousness_metrics']['success_rate']:.1%}")
    
    return agent, results

# Run example
if __name__ == "__main__":
    agent, results = consciousness_example()
```

---

## ⚠️ **Error Handling**

### **Common Exceptions**

#### **scientific pipeline ([integration tests](test_week3_complete_pipeline.py)) Exceptions**
```python
try:
    result = transformer.compute_laplace_transform(signal, time_vector)
except ValueError as e:
    # Invalid input parameters
    print(f"Parameter error: {e}")
except np.linalg.LinAlgError as e:
    # Numerical computation error
    print(f"Numerical error: {e}")
except Exception as e:
    # General processing error
    print(f"Processing error: {e}")
```

#### **Consciousness Exceptions**
```python
try:
    result = agent.perform_introspection(ReflectionType.SYSTEM_HEALTH_CHECK)
except RuntimeError as e:
    # Introspection system error
    print(f"Introspection error: {e}")
except ValueError as e:
    # Invalid reflection parameters
    print(f"Parameter error: {e}")
```

#### **Integrity monitoring ([health tracking](src/infrastructure/integration_coordinator.py)) Exceptions**
```python
try:
    violations = self_audit_engine.audit_text(text)
except Exception as e:
    # Audit system error - should be rare
    print(f"Audit error: {e}")
    # Fallback to basic validation
```

### **Error Recovery Patterns**

#### **Graceful Degradation**
```python
def robust_pipeline_execution(input_data):
    try:
        # Attempt full pipeline
        result = await coordinator.execute_scientific_pipeline(input_data)
        return result
    except Exception as e:
        # Log error and attempt partial processing
        logger.error(f"Full pipeline failed: {e}")
        
        try:
            # Attempt Laplace-only processing
            laplace_result = laplace_agent.compute_laplace_transform(
                input_data['signal_data'], 
                input_data['time_vector']
            )
            return create_partial_result(laplace_result)
        except Exception as e2:
            # Final fallback
            logger.error(f"Partial processing failed: {e2}")
            return create_error_result(e2)
```

---

## 📈 **Performance Guidelines**

### **Memory Management**
```python
# Good: Process signals in chunks for large datasets
def process_large_signal(signal_data, chunk_size=10000):
    results = []
    for i in range(0, len(signal_data), chunk_size):
        chunk = signal_data[i:i+chunk_size]
        result = transformer.compute_laplace_transform(chunk, time_vector)
        results.append(result)
    return results

# Good: Clean up agent resources
def cleanup_agents():
    if conscious_agent.continuous_reflection_enabled:
        conscious_agent.stop_continuous_reflection()
```

### **Performance Optimization**
```python
# Optimize for repeated operations
transformer = EnhancedLaplaceTransformer(
    num_points=1024,  # Reduce for faster processing
    enable_self_audit=False  # Disable for performance-critical operations
)

# Batch processing for multiple signals
results = []
for signal in signal_batch:
    result = transformer.compute_laplace_transform(signal, time_vector)
    results.append(result)
```

### **monitoring ([health tracking](src/infrastructure/integration_coordinator.py)) Performance**
```python
import time

def timed_operation():
    start_time = time.time()
    result = transformer.compute_laplace_transform(signal, time_vector)
    processing_time = time.time() - start_time
    
    # Log performance metrics
    logger.info(f"Processing time: {processing_time:.4f}s")
    logger.info(f"Signal quality: {result.quality_assessment.value}")
    logger.info(f"Memory usage: {result.metrics.memory_usage} bytes")
    
    return result
```

---

## 📝 **API Documentation Standards**

### **Docstring Format**
All API methods follow this documentation standard:

```python
def method_name(self, param1: Type1, param2: Type2 = default) -> ReturnType:
    """
    Brief description of method functionality.
    
    Longer description with implementation details, mathematical background,
    and performance characteristics.
    
    Args:
        param1: Description of parameter 1 with valid range/constraints
        param2: Description of parameter 2 with default behavior
        
    Returns:
        Description of return value with type and content details
        
    Raises:
        SpecificException: When this exception occurs
        
    Example:
        >>> result = obj.method_name(value1, value2)
        >>> print(f"Result: {result.property}")
        
    Performance:
        Time complexity: O(n log n)
        Memory usage: O(n) where n is signal length
        
    Mathematical Foundation:
        Mathematical description if applicable
    """
```

### **Type Annotations**
All methods include type annotations with implemented coverage:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def process_data(
    input_data: np.ndarray,
    parameters: Dict[str, Union[int, float]],
    options: Optional[List[str]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
```

---

<div align="center">
  <h3>📚 implemented API Reference for NIS Protocol v3</h3>
  <p><em>Mathematical rigor meets practical implementation</em></p>
  
  <p>
    <a href="../README.md">🏠 Main Documentation</a> •
    <a href="INTEGRATION_GUIDE.md">🔗 Integration Guide</a> •
    <a href="../examples/">🧪 Examples</a> •
    <a href="GETTING_STARTED.md">🚀 Getting Started</a>
  </p>
</div> 