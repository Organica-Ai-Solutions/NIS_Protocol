"""
Memory Agent

Stores and retrieves information for use by other agents in the system.
Analogous to the hippocampus in the brain, responsible for memory formation and recall.
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
import os
import datetime
from collections import deque

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension


class MemoryAgent(NISAgent):
    """
    Agent that manages storage and retrieval of information.
    
    The Memory Agent is responsible for:
    - Storing information from other agents
    - Retrieving relevant information based on queries
    - Maintaining both short-term and long-term memory
    - Providing context for decision-making
    """
    
    def __init__(
        self,
        agent_id: str = "memory",
        description: str = "Stores and retrieves information for the system",
        emotional_state: Optional[EmotionalState] = None,
        storage_path: Optional[str] = None,
        short_term_capacity: int = 1000
    ):
        """
        Initialize a new Memory Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            storage_path: Path to store persistent memory data
            short_term_capacity: Maximum number of items in short-term memory
        """
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.emotional_state = emotional_state or EmotionalState()
        
        # Short-term memory (in-memory cache)
        self.short_term = deque(maxlen=short_term_capacity)
        
        # Long-term memory (persistent storage)
        self.storage_path = storage_path
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request.
        
        Args:
            message: Message containing memory operation
                'operation': 'store', 'retrieve', 'query', 'forget'
                'data': Data to store (for 'store' operation)
                'query': Query parameters (for 'retrieve' or 'query' operations)
                'memory_id': ID of memory to forget (for 'forget' operation)
        
        Returns:
            Result of the memory operation
        """
        if not self._validate_message(message):
            return {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        operation = message.get("operation", "").lower()
        
        if operation == "store":
            return self._store_memory(message)
        elif operation == "retrieve":
            return self._retrieve_memory(message)
        elif operation == "query":
            return self._query_memory(message)
        elif operation == "forget":
            return self._forget_memory(message)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message format.
        
        Args:
            message: The message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
        
        if "operation" not in message:
            return False
        
        operation = message.get("operation", "").lower()
        
        if operation == "store" and "data" not in message:
            return False
        
        if operation in ["retrieve", "query"] and "query" not in message:
            return False
        
        if operation == "forget" and "memory_id" not in message:
            return False
        
        return True
    
    def _store_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            message: Message with data to store
            
        Returns:
            Storage result
        """
        data = message.get("data", {})
        
        # Generate memory ID if not provided
        memory_id = message.get("memory_id", f"mem_{time.time()}")
        
        # Prepare memory object
        memory = {
            "memory_id": memory_id,
            "timestamp": time.time(),
            "created": datetime.datetime.now().isoformat(),
            "data": data,
            "tags": message.get("tags", []),
            "importance": message.get("importance", 0.5),
            "source_agent": message.get("source_agent", "unknown")
        }
        
        # Store in short-term memory
        self.short_term.append(memory)
        
        # Store in long-term memory if path is configured
        if self.storage_path:
            self._persist_memory(memory)
        
        # Update emotional state based on importance
        if memory.get("importance", 0.5) > 0.7:
            self.emotional_state.update(EmotionalDimension.INTEREST.value, 0.7)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _retrieve_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            message: Message with memory ID to retrieve
            
        Returns:
            Retrieved memory or error
        """
        memory_id = message.get("query", {}).get("memory_id")
        
        if not memory_id:
            return {
                "status": "error",
                "error": "No memory_id provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # First check short-term memory
        for memory in self.short_term:
            if memory.get("memory_id") == memory_id:
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "short_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        # Then check long-term memory
        if self.storage_path:
            memory = self._load_memory(memory_id)
            if memory:
                # Add to short-term memory for faster future access
                self.short_term.append(memory)
                
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "long_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        return {
            "status": "error",
            "error": f"Memory not found: {memory_id}",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _query_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query memories based on criteria.
        
        Args:
            message: Message with query parameters
            
        Returns:
            List of matching memories
        """
        query = message.get("query", {})
        
        # Extract query parameters with defaults
        max_results = query.get("max_results", 10)
        start_time = query.get("start_time", 0)
        end_time = query.get("end_time", time.time())
        tags = query.get("tags", [])
        min_importance = query.get("min_importance", 0.0)
        source_agent = query.get("source_agent", None)
        
        results = []
        
        # Search short-term memory
        for memory in self.short_term:
            if self._memory_matches_query(memory, start_time, end_time, tags, min_importance, source_agent):
                results.append(memory)
                
                if len(results) >= max_results:
                    break
        
        # If not enough results and we have long-term storage, search there
        if len(results) < max_results and self.storage_path:
            # This is a simplified version; a real implementation would use 
            # a database or vector store for efficient querying
            long_term_results = self._query_long_term(
                start_time, end_time, tags, min_importance, source_agent, max_results - len(results)
            )
            results.extend(long_term_results)
        
        return {
            "status": "success",
            "results": results,
            "result_count": len(results),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _forget_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove a memory from storage.
        
        Args:
            message: Message with memory ID to forget
            
        Returns:
            Result of the operation
        """
        memory_id = message.get("memory_id")
        
        # Remove from short-term memory
        self.short_term = deque([m for m in self.short_term if m.get("memory_id") != memory_id], 
                               maxlen=self.short_term.maxlen)
        
        # Remove from long-term memory
        if self.storage_path:
            memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
            if os.path.exists(memory_path):
                os.remove(memory_path)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _persist_memory(self, memory: Dict[str, Any]) -> None:
        """
        Save memory to persistent storage.
        
        Args:
            memory: Memory object to persist
        """
        if not self.storage_path:
            return
        
        memory_id = memory.get("memory_id")
        memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
        
        with open(memory_path, 'w') as f:
            json.dump(memory, f)
    
    def _load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory from persistent storage.
        
        Args:
            memory_id: ID of memory to load
            
        Returns:
            Memory object or None if not found
        """
        if not self.storage_path:
            return None
        
        memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
        
        if not os.path.exists(memory_path):
            return None
        
        try:
            with open(memory_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _memory_matches_query(self, 
                             memory: Dict[str, Any], 
                             start_time: float, 
                             end_time: float, 
                             tags: List[str], 
                             min_importance: float, 
                             source_agent: Optional[str]) -> bool:
        """
        Check if a memory matches query criteria.
        
        Args:
            memory: Memory to check
            start_time: Minimum timestamp
            end_time: Maximum timestamp
            tags: Required tags (any match)
            min_importance: Minimum importance value
            source_agent: Source agent filter
            
        Returns:
            True if memory matches criteria
        """
        # Check timestamp
        timestamp = memory.get("timestamp", 0)
        if timestamp < start_time or timestamp > end_time:
            return False
        
        # Check importance
        importance = memory.get("importance", 0.0)
        if importance < min_importance:
            return False
        
        # Check source agent
        if source_agent and memory.get("source_agent") != source_agent:
            return False
        
        # Check tags (any match)
        if tags:
            memory_tags = memory.get("tags", [])
            if not any(tag in memory_tags for tag in tags):
                return False
        
        return True
    
    def _query_long_term(self, 
                        start_time: float, 
                        end_time: float, 
                        tags: List[str], 
                        min_importance: float, 
                        source_agent: Optional[str], 
                        max_results: int) -> List[Dict[str, Any]]:
        """
        Query long-term memory storage.
        
        Args:
            start_time: Minimum timestamp
            end_time: Maximum timestamp
            tags: Required tags (any match)
            min_importance: Minimum importance value
            source_agent: Source agent filter
            max_results: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if not self.storage_path:
            return []
        
        results = []
        
        # This is a simple implementation that scans all files
        # A real implementation would use a database or index
        try:
            memory_files = os.listdir(self.storage_path)
            for filename in memory_files:
                if not filename.endswith('.json'):
                    continue
                
                memory_path = os.path.join(self.storage_path, filename)
                
                try:
                    with open(memory_path, 'r') as f:
                        memory = json.load(f)
                        
                        if self._memory_matches_query(
                            memory, start_time, end_time, tags, min_importance, source_agent
                        ):
                            results.append(memory)
                            
                            if len(results) >= max_results:
                                break
                except (json.JSONDecodeError, IOError):
                    continue
        except OSError:
            pass
        
        return results 