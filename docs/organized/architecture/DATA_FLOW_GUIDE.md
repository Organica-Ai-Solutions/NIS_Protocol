# 🔄 NIS Protocol v3.1 - Data Flow Guide
## From API Request to AI Response

### 🚀 **Overview**

This guide traces the path of a single user request as it travels through the NIS Protocol v3.1. The flow is designed to be sequential, ensuring that every output is processed through a validation pipeline before being delivered.

---

## 📊 **The Data Flow Architecture**

The data flow is orchestrated by the **`EnhancedScientificCoordinator`** and managed by the **`EnhancedConsciousAgent`**. It follows a clear, linear path through the Scientific Pipeline.

```mermaid
graph TD
    subgraph "Start: User Interaction"
        A[API Request: POST /chat]
    end

    subgraph "Step 1: Scientific Coordination"
        B(EnhancedScientificCoordinator)
    end
    
    subgraph "Step 2: The Scientific Pipeline"
        C[1. Laplace Transform<br/>(EnhancedLaplaceTransformer)]
        D[2. KAN Reasoning<br/>(EnhancedKANReasoningAgent)]
        E[3. PINN Validation<br/>(EnhancedPINNPhysicsAgent)]
    end

    subgraph "Step 3: Language Synthesis"
        F[4. LLM Response Generation<br/>(CognitiveOrchestra)]
    end

    subgraph "End: Grounded Response"
        G[API Response]
    end
    
    subgraph "System Oversight (Continuous)"
        H((EnhancedConsciousAgent))
        I((DRLResourceManager))
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    H <--> B
    I <--> B
```

---

## 🔍 **Step-by-Step Data Flow Walkthrough**

Let's trace a chat request: `curl -X POST -d '{"message": "Is time travel possible?"}' http://localhost:8000/chat`

### **Phase 1: Request Reception & Coordination** 🚪

#### **1.1. API Request Received**
The FastAPI application in `main.py` receives the incoming POST request at the `/chat` endpoint. The request body contains the user's message.

```python
# main.py
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # The user's message is in request.message
    # "Is time travel possible?"
    ...
```

#### **1.2. Handoff to the Scientific Coordinator**
The endpoint's main responsibility is to pass the request to the central orchestrator, the `EnhancedScientificCoordinator`. The coordinator is responsible for managing the entire validation workflow.

```python
# main.py
# The coordinator is initialized at startup
scientific_coordinator = EnhancedScientificCoordinator()

# The request is handed off
pipeline_result = await scientific_coordinator.process_request(request.message)
```

### **Phase 2: The Scientific Pipeline Journey** 🔬

The `EnhancedScientificCoordinator` now pushes the data through the four-stage pipeline.

#### **2.1. Stage 1: Laplace Transform**
The coordinator first sends the raw text to the `EnhancedLaplaceTransformer`.

-   **Input:** `"Is time travel possible?"` (String)
-   **Process:** The agent converts the text into a numerical signal and applies the Laplace Transform, converting it into a representation in the frequency domain.
-   **Output:** A complex mathematical object representing the signal's core components.

```python
# EnhancedScientificCoordinator.py
# from src.agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
laplace_agent = EnhancedLaplaceTransformer()
laplace_output = laplace_agent.compute_laplace_transform(text_as_signal, time_vector)
```

#### **2.2. Stage 2: KAN Reasoning**
The output from the Laplace agent is then passed to the `EnhancedKANReasoningAgent`.

-   **Input:** The frequency-domain data from the Laplace agent.
-   **Process:** The KAN agent analyzes the mathematical object to discover an underlying symbolic function. It finds a mathematical equation that describes the relationships in the data.
-   **Output:** A symbolic expression, like `f(t) = -kt^2 + c`. This is the system's "interpretation" of the query.

```python
# EnhancedScientificCoordinator.py
# from src.agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent
kan_agent = EnhancedKANReasoningAgent()
kan_output = kan_agent.process_laplace_input(laplace_output) # Returns a SymbolicResult object
```

#### **2.3. Stage 3: PINN Physics Validation**
The symbolic function from the KAN agent is now sent to the `EnhancedPINNPhysicsAgent` for the critical validation step.

-   **Input:** The symbolic formula, `f(t) = -kt^2 + c`.
-   **Process:** The PINN agent checks this function against its internal knowledge base of physical laws.
-   **Output:** A `PINNValidationResult` object, which contains a boolean `is_valid` flag and a list of the laws it was checked against. For this query, it would likely find a violation of causality.

```python
# EnhancedScientificCoordinator.py
# from src.agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent
pinn_agent = EnhancedPINNPhysicsAgent()
pinn_output = pinn_agent.validate_kan_output(kan_output)
```

### **Phase 3: Language Synthesis & Final Response** 🗣️

#### **3.1. LLM Response Generation**
The `EnhancedScientificCoordinator` now has a validated (or invalidated) result from the pipeline. It passes this structured result to the `CognitiveOrchestra`.

-   **Input:** The `PINNValidationResult` object.
-   **Process:** The `CognitiveOrchestra` selects the configured LLM provider (e.g., OpenAI). It constructs a prompt that includes the final, validated information. For example: *"The user asked about time travel. Our physics validation pipeline concluded that it violates the principle of causality. Explain this conclusion in natural language."*
-   **Output:** A fluent, natural language string that reflects the findings of the scientific pipeline.

```python
# CognitiveOrchestra.py
# from src.llm.llm_manager import LLMManager
llm_manager = LLMManager()
llm_response = await llm_manager.process_request(
    prompt=synthesis_prompt,
    provider=self.active_provider
)
```

#### **3.2. Final API Response**
The coordinator packages the LLM's response along with the pipeline's structured output and returns it to the `main.py` endpoint, which then sends it back to the user.

```json
{
  "response": "According to our physics-based validation, time travel to the past is not considered possible as it would violate the principle of causality.",
  "nis_pipeline_output": {
    "pipeline": {
      "is_valid": false,
      "confidence": "calculated_value",
      "symbolic_equation": "f(t) = ...",
      "violated_laws": ["causality"]
    }
  },
  "llm_provider": "openai"
}
```

### **Continuous Oversight: The Meta-Control Loop** 🧠

While the data flows through the pipeline, the meta-control agents are active:

-   **`EnhancedConsciousAgent`:** Receives status updates from the coordinator at each stage. It monitors for integrity violations, performance bottlenecks, or repeated failures, updating its internal "awareness" metrics.
-   **`DRLResourceManager`:** Monitors the resource consumption (CPU, memory) of each agent in the pipeline. It can learn to proactively allocate more resources to agents that are under heavy load, ensuring the system remains responsive.

---

## 📜 **Key Design Principles**

1.  **Mandatory Validation:** There is no path from user input to LLM response that bypasses the scientific validation pipeline.
2.  **Interpretability by Design:** The flow through the KAN agent ensures that the system's reasoning is available as a symbolic formula.
3.  **Grounded in Physics:** The PINN agent acts as a gatekeeper, preventing physically impossible or nonsensical ideas from ever reaching the language generation stage.
4.  **Separation of Concerns:** The pipeline agents are responsible for **validation and reasoning**. The LLM is responsible only for **communication**. This clear separation is key to the system's design.

This verifiable data flow is a core component of the NIS Protocol's commitment to building trustworthy AI.
