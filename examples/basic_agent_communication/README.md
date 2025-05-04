# Basic Agent Communication Example

This example demonstrates how to create and connect multiple agents using the NIS Protocol. It showcases the flow of information through the cognitive layers and how the emotional state system influences decision-making.

## Overview

In this example, we create a simple system with the following agents:

1. **Vision Agent** - Processes visual input (simulated camera feed)
2. **Input Agent** - Processes text commands
3. **Memory Agent** - Stores and retrieves information
4. **Cortex Agent** - Makes decisions based on inputs and memory
5. **Action Agent** - Executes actions based on decisions

The example simulates a toll booth system where the agents need to process vehicles passing through, determine if they have valid payment methods, and decide whether to allow them to pass or flag them for inspection.

## Key Concepts Demonstrated

- Agent registration and communication
- Information flow through cognitive layers
- Emotional state influence on decision thresholds
- Memory storage and retrieval
- Feedback loops for continuous learning

## Running the Example

```bash
# Navigate to the example directory
cd examples/basic_agent_communication

# Run the example
python toll_booth_simulation.py
```

## Expected Output

The example will output the processing of several vehicles through the toll system:

```
[INFO] System initialized with 5 agents
[INFO] Processing vehicle: ABC123
[INFO] Vision Agent detected vehicle type: sedan
[INFO] Input Agent processed payment info: valid_ezpass
[INFO] Memory Agent found previous entries: 5 (all valid passages)
[INFO] Cortex Agent decision: allow_passage
[INFO] Action Agent executed: opened_gate
[INFO] Memory updated with successful passage

[INFO] Processing vehicle: XYZ789
[INFO] Vision Agent detected vehicle type: truck
[INFO] Input Agent processed payment info: missing_payment
[INFO] Memory Agent found previous entries: 0
[INFO] Cortex Agent decision: deny_passage (suspicion level: 0.72)
[INFO] Action Agent executed: alert_operator
[INFO] Memory updated with flagged passage
```

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