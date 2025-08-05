"""
Example demonstrating neuroplasticity and learning in the NIS Protocol.

This example shows how the NeuroplasticityAgent works with the EnhancedMemoryAgent
to implement adaptive learning and memory modification.
"""

import time
import random
from typing import Dict, Any

from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType
from src.agents.learning.neuroplasticity_agent import NeuroplasticityAgent
from src.agents.learning.optimizer_agent import OptimizerAgent
from src.emotion.emotional_state import EmotionalState

def create_test_memory(memory_agent: EnhancedMemoryAgent, content: str, memory_type: str) -> str:
    """Create a test memory and return its ID."""
    response = memory_agent.process({
        "operation": "store",
        "content": content,
        "memory_type": memory_type,
        "metadata": {
            "source": "test",
            "timestamp": time.time()
        }
    })
    
    if response["status"] == "success":
        return response["memory_id"]
    return None

def main():
    # Initialize agents
    memory_agent = EnhancedMemoryAgent(
        agent_id="memory",
        storage_path="data/memory",
        vector_store_path="data/vectors"
    )
    
    plasticity_agent = NeuroplasticityAgent(
        agent_id="plasticity",
        memory_agent=memory_agent,
        storage_path="data/plasticity"
    )
    
    optimizer_agent = OptimizerAgent(
        agent_id="optimizer",
        learning_rate=0.1
    )
    
    # Create some test memories
    memories = {
        "python": create_test_memory(
            memory_agent,
            "Python is a high-level programming language",
            MemoryType.SEMANTIC
        ),
        "coding": create_test_memory(
            memory_agent,
            "Writing code is a fundamental skill in software development",
            MemoryType.SEMANTIC
        ),
        "debugging": create_test_memory(
            memory_agent,
            "Found and fixed a bug in the authentication system",
            MemoryType.EPISODIC
        ),
        "testing": create_test_memory(
            memory_agent,
            "Wrote unit tests for the new feature",
            MemoryType.EPISODIC
        )
    }
    
    print("\n1. Initial State")
    print("---------------")
    stats = plasticity_agent.process({"operation": "stats"})
    print(f"Initial plasticity stats: {stats['stats']}")
    
    # Simulate memory activations and learning
    print("\n2. Simulating Memory Usage")
    print("------------------------")
    
    # Activate related memories together
    print("Activating related memories...")
    plasticity_agent.process({
        "operation": "record_activation",
        "memory_id": memories["python"],
        "activation_strength": 0.8
    })
    
    time.sleep(1)  # Small delay to simulate temporal relationship
    
    plasticity_agent.process({
        "operation": "record_activation",
        "memory_id": memories["coding"],
        "activation_strength": 0.9
    })
    
    # Explicitly strengthen some connections
    print("Strengthening connections...")
    plasticity_agent.process({
        "operation": "strengthen",
        "memory_id1": memories["debugging"],
        "memory_id2": memories["testing"],
        "strength_increase": 0.5
    })
    
    # Check stats after learning
    print("\n3. State After Learning")
    print("---------------------")
    stats = plasticity_agent.process({"operation": "stats"})
    print(f"Updated plasticity stats: {stats['stats']}")
    
    # Demonstrate optimization
    print("\n4. Optimizing Learning Parameters")
    print("-------------------------------")
    
    # Simulate some loss history
    loss_history = [0.5, 0.45, 0.4, 0.38, 0.35]
    
    new_rate = optimizer_agent.suggest_learning_rate(loss_history)
    print(f"Suggested learning rate: {new_rate}")
    
    # Update plasticity phase based on performance
    print("\n5. Adjusting Plasticity")
    print("---------------------")
    if new_rate > optimizer_agent.learning_rate:
        print("Performance improving - increasing plasticity")
        plasticity_agent.set_plasticity_phase("high")
    else:
        print("Performance stabilizing - reducing plasticity")
        plasticity_agent.set_plasticity_phase("low")
    
    # Final consolidation
    print("\n6. Memory Consolidation")
    print("---------------------")
    result = plasticity_agent.process({"operation": "consolidate"})
    print(f"Consolidation result: {result}")
    
    # Final stats
    print("\n7. Final State")
    print("-------------")
    stats = plasticity_agent.process({"operation": "stats"})
    print(f"Final plasticity stats: {stats['stats']}")

if __name__ == "__main__":
    main() 