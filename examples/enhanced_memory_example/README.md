# NIS Protocol Enhanced Memory Example

This example demonstrates the capabilities of the EnhancedMemoryAgent, which provides advanced memory management features including semantic search, memory organization, and forgetting mechanisms inspired by cognitive architectures.

## Features Demonstrated

- **Semantic Search**: Find memories based on meaning, not just keywords
- **Memory Organization**: Work with different memory types (episodic, semantic, procedural)
- **Working Memory**: Maintain a small set of currently relevant memories
- **Memory Consolidation**: Automatically move memories from short-term to long-term storage
- **Forgetting Mechanisms**: Importance-based memory decay over time
- **Metadata Queries**: Search by themes, importance, time, and other metadata

## Prerequisites

Before running this example, make sure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

This example requires additional packages for the vector database and embedding capabilities:
- `hnswlib`: For efficient vector search
- `numpy` and `scipy`: For scientific computing
- `sentence-transformers`: For generating semantic embeddings (optional, falls back to a simpler method if not available)

## Running the Example

Run the example with default settings:

```bash
python run.py
```

Clear existing memories and create new ones:

```bash
python run.py --clear
```

Use a custom storage location:

```bash
python run.py --storage /path/to/memory/storage
```

## Memory Types

The enhanced memory system divides memories into different types:

1. **Episodic Memories**: Records of specific events or experiences
2. **Semantic Memories**: General facts and knowledge
3. **Procedural Memories**: Knowledge about how to do things
4. **Working Memory**: Currently active memories regardless of type

## Key API Operations

The EnhancedMemoryAgent supports the following operations:

- **store**: Add a new memory
- **retrieve**: Get a specific memory by ID
- **query**: Find memories by metadata criteria
- **search**: Find memories by semantic similarity
- **forget**: Remove or mark memories as forgotten
- **consolidate**: Manually trigger memory consolidation
- **stats**: Get statistics about the memory system

## Example Integration

Here's a basic example of how to integrate the enhanced memory agent into your own NIS Protocol application:

```python
from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType
from src.emotion.emotional_state import EmotionalState

# Initialize the memory agent
memory_agent = EnhancedMemoryAgent(
    agent_id="my_memory",
    description="Memory system for my application",
    emotional_state=EmotionalState(),
    storage_path="./memory_data",
    short_term_capacity=500,
    working_memory_limit=10
)

# Store a memory
response = memory_agent.process({
    "operation": "store",
    "content": "The user asked about how to use the memory system.",
    "memory_type": MemoryType.EPISODIC,
    "themes": ["user_interaction", "help"],
    "importance": 0.7
})

# Search for related memories
search_response = memory_agent.process({
    "operation": "search",
    "query": {
        "text": "How do I help users with memory questions?",
        "top_k": 5
    }
})

# Process the search results
if search_response["status"] == "success":
    for memory in search_response["results"]:
        print(f"Found: {memory['content']} (similarity: {memory['similarity']:.2f})")
```

## How It Works

The EnhancedMemoryAgent uses vector embeddings to represent the semantic meaning of text content. These embeddings are stored in a vector database (HNSW) that allows for efficient similarity search. This enables finding memories based on their meaning rather than exact text matches.

Memory is organized in multiple layers:
- Working memory (small, active set)
- Short-term memory (recently accessed or created)
- Long-term memory (persistent storage)

Memory importance decays over time based on a combination of factors:
- Initial importance when stored
- Age of the memory
- How frequently it's accessed
- Emotional salience

Memories that decay below a threshold are automatically marked as forgotten.

## Neuroplasticity Implementation

The following code block demonstrates how to implement a NeuroplasticityAgent that strengthens connections between memories that activate together.

```python
from typing import List
import random

class NeuroplasticityAgent(NISAgent):
    def __init__(self, memory_agent: EnhancedMemoryAgent):
        super().__init__("neuroplasticity", NISLayer.LEARNING, "Implements neuroplasticity mechanisms")
        self.memory_agent = memory_agent
        self.connection_strengths = {}  # Tracks strength between memory pairs
        
    def strengthen_connections(self, activated_memories: List[str], strength_increase: float = 0.1):
        """Strengthen connections between simultaneously activated memories."""
        for i, mem_id1 in enumerate(activated_memories):
            for mem_id2 in activated_memories[i+1:]:
                pair_key = f"{mem_id1}:{mem_id2}"
                self.connection_strengths[pair_key] = self.connection_strengths.get(pair_key, 0.5) + strength_increase
                
    def apply_hebbian_learning(self, recent_activations: List[str]):
        """Apply Hebbian learning to recent memory activations."""
        self.strengthen_connections(recent_activations)
        
        # Periodically update vector store based on connection strengths
        if random.random() < 0.1:  # 10% chance each call
            self._update_vector_embeddings()
            
    def _update_vector_embeddings(self):
        """Update vector embeddings based on learned connection strengths."""
        # Complex implementation that would adjust vector positions
        # to reflect learned connection strengths
        pass 