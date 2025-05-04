# Basic Agent Communication Example

This example demonstrates how agents in the NIS Protocol communicate with each other and how the emotional state system modulates their behavior.

## Overview

In this example, we create a simple scenario with four types of agents:

1. **Vision Agent** - Processes visual inputs (simulated images)
2. **Memory Agent** - Stores and retrieves information
3. **Cortex Agent** - Makes decisions based on detected objects and emotional state
4. **Action Agent** - Executes actions based on decisions

The example simulates two scenarios:

1. A normal image containing everyday objects
2. A suspicious image that triggers an elevated emotional state

## What This Example Demonstrates

- How agents process and pass messages to each other
- How the emotional state system influences agent decisions
- How memory is used to store and retrieve information
- How the system responds differently to normal vs. suspicious inputs

## Running the Example

To run this example:

```bash
# From the project root directory
python examples/basic_agent_communication/run.py
```

## Expected Output

The example will print information about:

- Objects detected in each scenario
- Decisions made by the cortex agent
- Actions taken by the action agent
- Changes in the emotional state
- Memory retrieval results

## Key Concepts

### Agent Communication

Agents communicate by passing messages to each other. Each message contains:

- A payload with the main information
- Metadata with additional context
- The current emotional state

### Emotional Modulation

The emotional state influences decision-making. When suspicious objects are detected:

1. The suspicion dimension increases
2. This affects the decision threshold in the cortex agent
3. The action agent takes more cautious actions

### Memory Usage

The memory system stores information that can be retrieved later:

- Scene information is stored with unique keys
- Other agents can retrieve this information when needed
- This provides historical context for decisions

## Code Structure

- `toll_booth_simulation.py` - Main simulation script
- `toll_utils.py` - Helper utilities for the toll booth example
- `config.json` - Configuration for the simulation

## Key Implementation Details

### Agent Registration

```python
# Create and register agents with the global registry
vision_agent = VisionAgent(agent_id="toll_vision", description="Toll booth camera")
input_agent = InputAgent(agent_id="toll_payment", description="Payment processor")
memory_agent = MemoryAgent(agent_id="toll_memory", description="Vehicle history", 
                           storage_path="./data")
cortex_agent = CortexAgent(agent_id="toll_decision", description="Passage decision maker")
action_agent = ActionAgent(agent_id="toll_action", description="Gate and alert controller")
```

### Information Flow

```python
# Process visual input
vision_result = vision_agent.process({
    "image_data": vehicle.image_data
})

# Process payment info
input_result = input_agent.process({
    "text": vehicle.payment_info
})

# Query memory for vehicle history
memory_result = memory_agent.process({
    "operation": "query",
    "query": {
        "tags": ["vehicle", vehicle.plate_number],
        "max_results": 10
    }
})

# Make decision based on all inputs
cortex_result = cortex_agent.process({
    "vision_data": vision_result,
    "input_data": input_result,
    "memory_data": memory_result,
    "vehicle": vehicle
})

# Execute the decided action
action_result = action_agent.process({
    "decision": cortex_result["decision"],
    "vehicle": vehicle
})

# Store the result in memory
memory_agent.process({
    "operation": "store",
    "data": {
        "vehicle": vehicle.plate_number,
        "decision": cortex_result["decision"],
        "action": action_result["action"],
        "timestamp": time.time()
    },
    "tags": ["vehicle", vehicle.plate_number, cortex_result["decision"]],
    "importance": cortex_result.get("importance", 0.5)
})
```

### Emotional Influence

The example demonstrates how emotional state influences decision-making:

1. When a vehicle with no payment is detected, the **suspicion** dimension increases
2. During high traffic periods, the **urgency** dimension increases
3. With higher suspicion, the decision threshold for flagging vehicles lowers
4. With higher urgency, processing speed is prioritized over thorough checks

## Extending the Example

You can modify this example to explore different aspects of the NIS Protocol:

1. Add more agent types from different cognitive layers
2. Implement more sophisticated decision-making in the Cortex Agent
3. Add learning capabilities to improve decision accuracy over time
4. Integrate with actual hardware (cameras, payment systems, gates)

## Further Reading

For more detailed information on the NIS Protocol components used in this example, see:

- [Agent Registry Documentation](../../docs/agent_registry.md)
- [Emotional State System](../../docs/emotional_state.md)
- [Memory System](../../docs/memory_system.md)
- [Message Flow](../../docs/message_flow.md) 