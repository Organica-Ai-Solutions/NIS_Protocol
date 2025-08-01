#!/usr/bin/env python3
"""
NIS Protocol Enhanced Memory Agent Example

This example demonstrates how to use the EnhancedMemoryAgent with semantic search
and comprehensive memory features.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType
from src.emotion.emotional_state import EmotionalState
from src.core.registry import NISRegistry
from src.memory.embedding_utils import get_embedding_provider

# Sample data for demonstration
SAMPLE_FACTS = [
    {
        "content": "The NIS Protocol is a biologically-inspired framework for multi-agent systems.",
        "memory_type": MemoryType.SEMANTIC,
        "themes": ["protocol", "architecture"],
        "importance": 0.9
    },
    {
        "content": "The emotional state system in NIS Protocol modulates agent behavior based on context.",
        "memory_type": MemoryType.SEMANTIC,
        "themes": ["emotion", "architecture"],
        "importance": 0.8
    },
    {
        "content": "Memory in the NIS Protocol is organized into episodic, semantic, and procedural types.",
        "memory_type": MemoryType.SEMANTIC,
        "themes": ["memory", "architecture"],
        "importance": 0.85
    },
    {
        "content": "The vision agent using YOLO can detect objects in images and video streams.",
        "memory_type": MemoryType.SEMANTIC,
        "themes": ["vision", "perception"],
        "importance": 0.75
    },
    {
        "content": "Agent communication in NIS Protocol happens through standardized message formats.",
        "memory_type": MemoryType.SEMANTIC,
        "themes": ["communication", "architecture"],
        "importance": 0.7
    }
]

SAMPLE_EPISODES = [
    {
        "content": "The system detected a car in the driveway at 2:30 PM yesterday.",
        "memory_type": MemoryType.EPISODIC,
        "themes": ["observation", "security"],
        "importance": 0.6
    },
    {
        "content": "A user asked about the emotional state system and how it affects decision making.",
        "memory_type": MemoryType.EPISODIC,
        "themes": ["user_interaction", "emotion"],
        "importance": 0.65
    },
    {
        "content": "The system experienced a temporary connection loss to the database at 3:15 AM.",
        "memory_type": MemoryType.EPISODIC,
        "themes": ["error", "maintenance"],
        "importance": 0.85
    },
    {
        "content": "A new agent was registered in the system with ID 'weather_monitor'.",
        "memory_type": MemoryType.EPISODIC,
        "themes": ["system_event", "registration"],
        "importance": 0.5
    }
]

SAMPLE_PROCEDURES = [
    {
        "content": "To register a new agent, instantiate the agent class and the registry will systematically add it.",
        "memory_type": MemoryType.PROCEDURAL,
        "themes": ["agent", "registration"],
        "importance": 0.75
    },
    {
        "content": "Image processing in the vision agent starts with loading the image then applying YOLO detection.",
        "memory_type": MemoryType.PROCEDURAL,
        "themes": ["vision", "procedure"],
        "importance": 0.7
    },
    {
        "content": "Memory consolidation happens systematically at regular intervals to move short-term memories to long-term storage.",
        "memory_type": MemoryType.PROCEDURAL,
        "themes": ["memory", "maintenance"],
        "importance": 0.8
    }
]

def print_memory(memory: Dict[str, Any], with_similarity: bool = False) -> None:
    """
    Print a memory in a human-readable format.
    
    Args:
        memory: The memory to print
        with_similarity: Whether to include similarity score
    """
    print(f"Memory ID: {memory.get('memory_id')}")
    print(f"Type: {memory.get('memory_type')}")
    print(f"Content: {memory.get('content')}")
    print(f"Themes: {', '.join(memory.get('themes', []))}")
    print(f"Importance: {memory.get('importance', 0.0):.2f}")
    
    if with_similarity and "similarity" in memory:
        print(f"Similarity: {memory.get('similarity', 0.0):.4f}")
    
    print(f"Created: {memory.get('created')}")
    print(f"Access Count: {memory.get('access_count', 0)}")
    print()

def initialize_memory_agent(storage_path: str, clear_existing: bool = False) -> EnhancedMemoryAgent:
    """
    Initialize and populate the memory agent.
    
    Args:
        storage_path: Path to store memory data
        clear_existing: Whether to clear existing memories
        
    Returns:
        Initialized memory agent
    """
    # Create storage directory if it doesn't exist
    if not os.path.exists(storage_path):
        os.makedirs(storage_path, exist_ok=True)
    
    # Clear existing memories if requested
    if clear_existing:
        for root, dirs, files in os.walk(storage_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".npy") or file.endswith(".bin"):
                    os.remove(os.path.join(root, file))
    
    # Initialize the memory agent
    memory_agent = EnhancedMemoryAgent(
        agent_id="demo_memory",
        description="Enhanced memory agent for demonstration",
        emotional_state=EmotionalState(),
        storage_path=storage_path,
        short_term_capacity=100,
        working_memory_limit=5,
        consolidation_interval=3600,  # 1 hour
        forgetting_factor=0.01
    )
    
    # Populate with sample memories if cleared
    if clear_existing:
        # Add semantic facts
        for fact in SAMPLE_FACTS:
            memory_agent.process({
                "operation": "store",
                "content": fact["content"],
                "memory_type": fact["memory_type"],
                "themes": fact["themes"],
                "importance": fact["importance"],
                "source_agent": "example"
            })
        
        # Add episodic memories
        for episode in SAMPLE_EPISODES:
            memory_agent.process({
                "operation": "store",
                "content": episode["content"],
                "memory_type": episode["memory_type"],
                "themes": episode["themes"],
                "importance": episode["importance"],
                "source_agent": "example"
            })
        
        # Add procedural memories
        for procedure in SAMPLE_PROCEDURES:
            memory_agent.process({
                "operation": "store",
                "content": procedure["content"],
                "memory_type": procedure["memory_type"],
                "themes": procedure["themes"],
                "importance": procedure["importance"],
                "source_agent": "example"
            })
    
    return memory_agent

def demonstrate_semantic_search(memory_agent: EnhancedMemoryAgent) -> None:
    """
    Demonstrate semantic search functionality.
    
    Args:
        memory_agent: The memory agent to use
    """
    print("\n=== Semantic Search Demonstration ===\n")
    
    # Example search queries
    queries = [
        "How does emotional state work in the system?",
        "What happens when detecting objects in images?",
        "Tell me about memory organization",
        "What errors or issues have occurred?",
        "How do agents communicate with each other?"
    ]
    
    for query in queries:
        print(f"Query: \"{query}\"\n")
        
        # Perform semantic search
        response = memory_agent.process({
            "operation": "search",
            "query": {
                "text": query,
                "top_k": 2,
                "min_similarity": 0.0
            }
        })
        
        if response["status"] == "success" and "results" in response:
            print(f"Found {len(response['results'])} results:\n")
            
            for memory in response["results"]:
                print_memory(memory, with_similarity=True)
        else:
            print(f"Search failed: {response.get('error', 'Unknown error')}\n")
        
        print("-" * 50)

def demonstrate_metadata_query(memory_agent: EnhancedMemoryAgent) -> None:
    """
    Demonstrate metadata-based query functionality.
    
    Args:
        memory_agent: The memory agent to use
    """
    print("\n=== Metadata Query Demonstration ===\n")
    
    # Example queries
    query_examples = [
        {
            "description": "Memories with 'architecture' theme",
            "query": {
                "themes": ["architecture"],
                "max_results": 3
            }
        },
        {
            "description": "High importance memories (>=0.8)",
            "query": {
                "min_importance": 0.8,
                "max_results": 3
            }
        },
        {
            "description": "Episodic memories only",
            "query": {
                "memory_type": MemoryType.EPISODIC,
                "max_results": 3
            }
        },
        {
            "description": "Memories about vision or perception",
            "query": {
                "themes": ["vision", "perception"],
                "max_results": 3
            }
        }
    ]
    
    for example in query_examples:
        print(f"Query: {example['description']}\n")
        
        # Perform metadata query
        response = memory_agent.process({
            "operation": "query",
            "query": example["query"]
        })
        
        if response["status"] == "success" and "results" in response:
            print(f"Found {len(response['results'])} results:\n")
            
            for memory in response["results"]:
                print_memory(memory)
        else:
            print(f"Query failed: {response.get('error', 'Unknown error')}\n")
        
        print("-" * 50)

def demonstrate_working_memory(memory_agent: EnhancedMemoryAgent) -> None:
    """
    Demonstrate working memory functionality.
    
    Args:
        memory_agent: The memory agent to use
    """
    print("\n=== Working Memory Demonstration ===\n")
    
    # Add some memories to working memory
    working_memories = [
        {
            "content": "The current user is asking about memory systems.",
            "memory_type": MemoryType.EPISODIC,
            "themes": ["user_context", "current_session"],
            "importance": 0.9,
            "add_to_working": True
        },
        {
            "content": "The system is currently in a high alert state due to multiple failed login attempts.",
            "memory_type": MemoryType.EPISODIC,
            "themes": ["system_state", "security"],
            "importance": 0.95,
            "add_to_working": True
        },
        {
            "content": "The current task is to explain how the memory system works to the user.",
            "memory_type": MemoryType.EPISODIC,
            "themes": ["current_task", "user_interaction"],
            "importance": 0.9,
            "add_to_working": True
        }
    ]
    
    print("Adding items to working memory...\n")
    
    for item in working_memories:
        response = memory_agent.process({
            "operation": "store",
            "content": item["content"],
            "memory_type": item["memory_type"],
            "themes": item["themes"],
            "importance": item["importance"],
            "source_agent": "example",
            "add_to_working": item["add_to_working"]
        })
        
        if response["status"] == "success":
            print(f"Added to working memory: \"{item['content']}\"")
    
    print("\nQuering working memory items:\n")
    
    # Query working memory
    response = memory_agent.process({
        "operation": "query",
        "query": {
            "memory_type": MemoryType.WORKING,
            "max_results": 5
        }
    })
    
    if response["status"] == "success" and "results" in response:
        print(f"Found {len(response['results'])} items in working memory:\n")
        
        for memory in response["results"]:
            print_memory(memory)
    else:
        print(f"Query failed: {response.get('error', 'Unknown error')}\n")

def demonstrate_memory_operations(memory_agent: EnhancedMemoryAgent) -> None:
    """
    Demonstrate various memory operations.
    
    Args:
        memory_agent: The memory agent to use
    """
    print("\n=== Memory Operations Demonstration ===\n")
    
    # Store a new memory
    print("Storing a new memory...\n")
    store_response = memory_agent.process({
        "operation": "store",
        "content": "The enhanced memory agent demonstration was run at " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_type": MemoryType.EPISODIC,
        "themes": ["demonstration", "example"],
        "importance": 0.7,
        "source_agent": "example"
    })
    
    if store_response["status"] == "success":
        memory_id = store_response["memory_id"]
        print(f"Successfully stored memory with ID: {memory_id}\n")
        
        # Retrieve the memory
        print("Retrieving the memory...\n")
        retrieve_response = memory_agent.process({
            "operation": "retrieve",
            "query": {
                "memory_id": memory_id
            }
        })
        
        if retrieve_response["status"] == "success" and "memory" in retrieve_response:
            print("Retrieved memory:")
            print_memory(retrieve_response["memory"])
            
            # Update access count by retrieving again
            memory_agent.process({
                "operation": "retrieve",
                "query": {
                    "memory_id": memory_id
                }
            })
            
            memory_agent.process({
                "operation": "retrieve",
                "query": {
                    "memory_id": memory_id
                }
            })
            
            print("After multiple retrievals:")
            retrieve_response = memory_agent.process({
                "operation": "retrieve",
                "query": {
                    "memory_id": memory_id
                }
            })
            print_memory(retrieve_response["memory"])
            
            # Forget the memory
            print("Forgetting the memory...\n")
            forget_response = memory_agent.process({
                "operation": "forget",
                "memory_id": memory_id,
                "forget_type": "soft"  # Mark as forgotten but don't delete
            })
            
            if forget_response["status"] == "success":
                print(f"Successfully forgot memory: {forget_response['message']}")
                
                # Try to retrieve it again
                print("\nTrying to retrieve forgotten memory...\n")
                retrieve_response = memory_agent.process({
                    "operation": "retrieve",
                    "query": {
                        "memory_id": memory_id
                    }
                })
                
                if retrieve_response["status"] == "error":
                    print(f"As expected, couldn't retrieve forgotten memory: {retrieve_response['error']}")
                else:
                    print("Unexpectedly able to retrieve forgotten memory!")
            else:
                print(f"Failed to forget memory: {forget_response.get('error', 'Unknown error')}")
        else:
            print(f"Failed to retrieve memory: {retrieve_response.get('error', 'Unknown error')}")
    else:
        print(f"Failed to store memory: {store_response.get('error', 'Unknown error')}")

def demonstrate_memory_stats(memory_agent: EnhancedMemoryAgent) -> None:
    """
    Demonstrate memory statistics functionality.
    
    Args:
        memory_agent: The memory agent to use
    """
    print("\n=== Memory Statistics Demonstration ===\n")
    
    stats_response = memory_agent.process({
        "operation": "stats"
    })
    
    if stats_response["status"] == "success" and "stats" in stats_response:
        stats = stats_response["stats"]
        
        print("Memory System Statistics:")
        print(f"  Short-term memory: {stats['short_term_count']} / {stats['short_term_capacity']} items")
        print(f"  Working memory: {stats['working_memory_count']} / {stats['working_memory_capacity']} items")
        print(f"  Theme count: {stats['theme_count']} themes")
        print(f"  Available themes: {', '.join(stats['themes'])}")
        print(f"  Time since last consolidation: {stats['time_since_consolidation']:.2f} seconds")
        
        print("\nVector stores:")
        for memory_type, store_stats in stats['vector_stores'].items():
            print(f"  {memory_type}: {store_stats['vector_count']} vectors")
    else:
        print(f"Failed to get statistics: {stats_response.get('error', 'Unknown error')}")

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="NIS Enhanced Memory Agent Example")
    parser.add_argument("--storage", type=str, default="./memory_storage",
                        help="Path to store memory data")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing memories")
    args = parser.parse_args()
    
    # Initialize the NIS Registry
    registry = NISRegistry()
    
    print("Initializing Enhanced Memory Agent...")
    memory_agent = initialize_memory_agent(args.storage, args.clear)
    
    # Run demonstrations
    demonstrate_semantic_search(memory_agent)
    demonstrate_metadata_query(memory_agent)
    demonstrate_working_memory(memory_agent)
    demonstrate_memory_operations(memory_agent)
    demonstrate_memory_stats(memory_agent)
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 