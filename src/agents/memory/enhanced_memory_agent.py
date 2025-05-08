"""
Enhanced Memory Agent

Advanced memory management including semantic search, memory organization, and forgetting.
Inspired by memory systems in cognitive architectures and neuroscience.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import time
import json
import os
import datetime
import logging
from collections import deque, defaultdict
import numpy as np
import math

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension
from src.memory.vector_store import VectorStore
from src.memory.embedding_utils import get_embedding_provider, EmbeddingProvider


class MemoryType:
    """Memory types for different kinds of information."""
    EPISODIC = "episodic"  # Memories of specific events/experiences
    SEMANTIC = "semantic"  # General knowledge/facts
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Currently active information


class EnhancedMemoryAgent(NISAgent):
    """
    Enhanced agent for advanced memory management with semantic search.
    
    Features:
    - Semantic search using vector embeddings
    - Memory organization by themes and types
    - Memory consolidation and forgetting
    - Importance-based retention
    - Time-based decay for relevance
    - Query by similarity, time, or metadata
    """
    
    def __init__(
        self,
        agent_id: str = "memory",
        description: str = "Enhanced memory system with semantic search capabilities",
        emotional_state: Optional[EmotionalState] = None,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_dimensions: int = 768,
        max_vectors: int = 100000,
        short_term_capacity: int = 1000,
        working_memory_limit: int = 10,
        consolidation_interval: int = 3600,  # 1 hour
        forgetting_factor: float = 0.05,
        enable_logging: bool = True
    ):
        """
        Initialize the enhanced memory agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            storage_path: Path to store persistent memory data
            embedding_provider: Provider for creating text embeddings
            vector_dimensions: Dimensions for embedding vectors
            max_vectors: Maximum number of vectors to store
            short_term_capacity: Maximum items in short-term memory
            working_memory_limit: Maximum items in working memory
            consolidation_interval: Seconds between memory consolidation
            forgetting_factor: Rate of memory importance decay (0-1)
            enable_logging: Whether to log memory operations
        """
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.emotional_state = emotional_state or EmotionalState()
        
        # Set up storage paths
        self.storage_path = storage_path
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            
            # Create subdirectories for different memory types
            for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
                os.makedirs(os.path.join(storage_path, memory_type), exist_ok=True)
        
        # Set up in-memory storage
        self.short_term = deque(maxlen=short_term_capacity)
        self.working_memory = deque(maxlen=working_memory_limit)
        
        # Set up collection of memories by theme/category
        self.themes = defaultdict(list)
        
        # Set up embedding provider
        self.embedding_provider = embedding_provider or get_embedding_provider(
            cache_dir=os.path.join(storage_path, "embeddings_cache") if storage_path else ".embeddings_cache",
            dimensions=vector_dimensions
        )
        
        # Set up vector stores for different memory types
        self.vector_stores = {}
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            self.vector_stores[memory_type] = VectorStore(
                dim=vector_dimensions,
                max_elements=max_vectors
            )
            
            # Load existing vector store if available
            if storage_path:
                vector_path = os.path.join(storage_path, memory_type, "vectors")
                if os.path.exists(vector_path):
                    self.vector_stores[memory_type].load(vector_path)
        
        # Set up memory settings
        self.consolidation_interval = consolidation_interval
        self.forgetting_factor = forgetting_factor
        self.last_consolidation = time.time()
        
        # Set up logging
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger(f"nis_memory_agent_{agent_id}")
            self.logger.setLevel(logging.INFO)
            
            if storage_path:
                log_file = os.path.join(storage_path, f"memory_{agent_id}.log")
                handler = logging.FileHandler(log_file)
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request.
        
        Args:
            message: Message containing memory operation
                'operation': Operation to perform 
                    ('store', 'retrieve', 'query', 'search', 'forget', 'consolidate', 'stats')
                + Additional parameters based on operation
        
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
        
        # Check if it's time to consolidate memories
        current_time = time.time()
        if current_time - self.last_consolidation > self.consolidation_interval:
            self._consolidate_memories()
            self.last_consolidation = current_time
        
        operation = message.get("operation", "").lower()
        
        # Route to appropriate handler based on operation
        if operation == "store":
            return self._store_memory(message)
        elif operation == "retrieve":
            return self._retrieve_memory(message)
        elif operation == "query":
            return self._query_memory(message)
        elif operation == "search":
            return self._semantic_search(message)
        elif operation == "forget":
            return self._forget_memory(message)
        elif operation == "consolidate":
            return self._manual_consolidate()
        elif operation == "stats":
            return self._get_stats()
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
        
        if operation == "store":
            return "content" in message or "data" in message
        
        if operation in ["retrieve", "query", "search"]:
            return "query" in message
        
        if operation == "forget":
            return "memory_id" in message
        
        # No special requirements for "consolidate" or "stats"
        return operation in ["consolidate", "stats"]
    
    def _store_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            message: Message with memory content to store
                'content': Text content of the memory (optional)
                'data': Structured data for the memory (optional)
                'memory_type': Type of memory (episodic, semantic, procedural)
                'memory_id': Optional ID (generated if not provided)
                'themes': List of themes/categories for this memory
                'importance': Importance score (0.0-1.0)
                'source_agent': ID of the source agent
                'add_to_working': Whether to add to working memory
                
        Returns:
            Storage result with memory ID
        """
        # Extract memory data
        content = message.get("content", "")
        data = message.get("data", {})
        memory_type = message.get("memory_type", MemoryType.EPISODIC)
        themes = message.get("themes", [])
        importance = message.get("importance", 0.5)
        source_agent = message.get("source_agent", "unknown")
        add_to_working = message.get("add_to_working", False)
        
        # Generate memory ID if not provided
        memory_id = message.get("memory_id", f"mem_{time.time()}_{memory_type}")
        
        # Create timestamp and metadata
        timestamp = time.time()
        created_time = datetime.datetime.now().isoformat()
        
        # Ensure memory_type is valid
        if memory_type not in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            memory_type = MemoryType.EPISODIC
        
        # Prepare memory object
        memory = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "created": created_time,
            "content": content,
            "data": data,
            "themes": themes,
            "importance": importance,
            "source_agent": source_agent,
            "last_accessed": timestamp,
            "access_count": 0,
            "embedding_available": False
        }
        
        # Calculate embedding if there's content
        if content:
            try:
                embedding = self.embedding_provider.get_embedding(content)
                
                # Store in vector database for this memory type
                self.vector_stores[memory_type].add(
                    memory_id=memory_id,
                    vector=embedding,
                    metadata={
                        "memory_id": memory_id,
                        "timestamp": timestamp,
                        "importance": importance,
                        "themes": themes,
                        "memory_type": memory_type
                    }
                )
                
                memory["embedding_available"] = True
                
                if self.enable_logging:
                    self.logger.info(f"Generated embedding for memory {memory_id}")
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Failed to generate embedding for memory {memory_id}: {e}")
        
        # Store in short-term memory
        self.short_term.append(memory)
        
        # Add to working memory if requested
        if add_to_working:
            # Remove oldest if at capacity
            if len(self.working_memory) >= self.working_memory.maxlen:
                self.working_memory.popleft()
            self.working_memory.append(memory)
        
        # Add to theme collections
        for theme in themes:
            self.themes[theme].append(memory_id)
        
        # Persist to long-term storage if path configured
        if self.storage_path:
            self._persist_memory(memory)
        
        # Update emotional state based on importance
        if importance > 0.7:
            self.emotional_state.update(EmotionalDimension.INTEREST.value, 0.7)
        
        if self.enable_logging:
            self.logger.info(f"Stored memory {memory_id} of type {memory_type}")
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "memory_type": memory_type,
            "embedding_stored": memory["embedding_available"],
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _retrieve_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            message: Message with query parameters
                'query': Dictionary with 'memory_id' 
                'update_access': Whether to update access time (default: True)
            
        Returns:
            Retrieved memory or error
        """
        memory_id = message.get("query", {}).get("memory_id")
        update_access = message.get("update_access", True)
        
        if not memory_id:
            return {
                "status": "error",
                "error": "No memory_id provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # First check working memory
        for memory in self.working_memory:
            if memory.get("memory_id") == memory_id:
                if update_access:
                    memory["last_accessed"] = time.time()
                    memory["access_count"] += 1
                
                if self.enable_logging:
                    self.logger.info(f"Retrieved memory {memory_id} from working memory")
                
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "working_memory",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        # Then check short-term memory
        for memory in self.short_term:
            if memory.get("memory_id") == memory_id:
                if update_access:
                    memory["last_accessed"] = time.time()
                    memory["access_count"] += 1
                
                if self.enable_logging:
                    self.logger.info(f"Retrieved memory {memory_id} from short-term memory")
                
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "short_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        # Finally check long-term memory
        if self.storage_path:
            memory = self._load_memory(memory_id)
            if memory:
                # Update access information if requested
                if update_access:
                    memory["last_accessed"] = time.time()
                    memory["access_count"] += 1
                    self._persist_memory(memory)
                
                # Add to short-term memory for faster future access
                self.short_term.append(memory)
                
                if self.enable_logging:
                    self.logger.info(f"Retrieved memory {memory_id} from long-term memory")
                
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "long_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        if self.enable_logging:
            self.logger.warning(f"Memory not found: {memory_id}")
        
        return {
            "status": "error",
            "error": f"Memory not found: {memory_id}",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _query_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query memories based on metadata criteria.
        
        Args:
            message: Message with query parameters
                'query': Dictionary with query parameters:
                    'memory_type': Type of memory to query
                    'themes': List of themes to filter by
                    'start_time': Start timestamp
                    'end_time': End timestamp
                    'min_importance': Minimum importance
                    'source_agent': Source agent ID
                    'max_results': Maximum number of results
                    
        Returns:
            List of matching memories
        """
        query = message.get("query", {})
        
        # Extract query parameters with defaults
        memory_type = query.get("memory_type")
        themes = query.get("themes", [])
        max_results = query.get("max_results", 10)
        start_time = query.get("start_time", 0)
        end_time = query.get("end_time", time.time())
        min_importance = query.get("min_importance", 0.0)
        source_agent = query.get("source_agent")
        
        results = []
        
        # Search working memory
        if not memory_type or memory_type == MemoryType.WORKING:
            for memory in self.working_memory:
                if self._memory_matches_metadata(
                    memory, memory_type, themes, start_time, end_time, 
                    min_importance, source_agent
                ):
                    results.append(memory)
        
        # Search short-term memory
        for memory in self.short_term:
            if self._memory_matches_metadata(
                memory, memory_type, themes, start_time, end_time, 
                min_importance, source_agent
            ):
                # Skip duplicates from working memory
                if not any(r["memory_id"] == memory["memory_id"] for r in results):
                    results.append(memory)
                    
                    # Stop if we have enough results
                    if len(results) >= max_results:
                        break
        
        # If we don't have enough results and have a storage path, search long-term
        if len(results) < max_results and self.storage_path:
            long_term_results = self._query_long_term(
                memory_type, themes, start_time, end_time, 
                min_importance, source_agent, max_results - len(results)
            )
            
            # Add long-term results, avoiding duplicates
            for memory in long_term_results:
                if not any(r["memory_id"] == memory["memory_id"] for r in results):
                    results.append(memory)
        
        # Sort results by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Limit to max_results
        results = results[:max_results]
        
        # Update access times for retrieved memories
        for memory in results:
            memory["last_accessed"] = time.time()
            memory["access_count"] = memory.get("access_count", 0) + 1
        
        if self.enable_logging:
            self.logger.info(f"Query returned {len(results)} results")
        
        return {
            "status": "success",
            "results": results,
            "result_count": len(results),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _semantic_search(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform semantic search on memories using text similarity.
        
        Args:
            message: Message with search parameters
                'query': Dictionary with search parameters:
                    'text': Text to search for
                    'memory_type': Type of memory to search (optional)
                    'top_k': Number of results to return (default: 5)
                    'min_similarity': Minimum similarity score (0-1)
                    'filter_themes': List of themes to filter by
                    
        Returns:
            List of semantically similar memories with similarity scores
        """
        query_params = message.get("query", {})
        
        # Extract query parameters
        query_text = query_params.get("text", "")
        memory_type = query_params.get("memory_type")
        top_k = query_params.get("top_k", 5)
        min_similarity = query_params.get("min_similarity", 0.0)
        filter_themes = query_params.get("filter_themes", [])
        
        if not query_text:
            return {
                "status": "error",
                "error": "No search text provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Generate embedding for query text
        try:
            query_embedding = self.embedding_provider.get_embedding(query_text)
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to generate embedding for query: {e}")
            
            return {
                "status": "error",
                "error": f"Failed to generate embedding: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        results = []
        
        # Determine which memory types to search
        memory_types = [memory_type] if memory_type else [
            MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL
        ]
        
        # Prepare filter categories
        filter_categories = {}
        if filter_themes:
            filter_categories["themes"] = filter_themes[0]  # Simple implementation: just use first theme
        
        # Search each requested memory type
        for mem_type in memory_types:
            if mem_type in self.vector_stores:
                vector_results = self.vector_stores[mem_type].search(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter_categories=filter_categories if filter_categories else None
                )
                
                # Process search results
                for memory_id, similarity, metadata in vector_results:
                    if similarity >= min_similarity:
                        # Retrieve the full memory
                        memory = None
                        
                        # Check short-term memory first
                        for mem in self.short_term:
                            if mem.get("memory_id") == memory_id:
                                memory = mem
                                break
                        
                        # If not in short-term, check long-term
                        if memory is None and self.storage_path:
                            memory = self._load_memory(memory_id)
                        
                        # If found, add to results
                        if memory:
                            # Update access information
                            memory["last_accessed"] = time.time()
                            memory["access_count"] = memory.get("access_count", 0) + 1
                            
                            # Add to results with similarity score
                            result = memory.copy()
                            result["similarity"] = similarity
                            results.append(result)
                            
                            # Update persisted memory if in long-term storage
                            if self.storage_path:
                                self._persist_memory(memory)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Limit to top_k
        results = results[:top_k]
        
        if self.enable_logging:
            self.logger.info(f"Semantic search returned {len(results)} results")
        
        return {
            "status": "success",
            "results": results,
            "result_count": len(results),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _forget_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forget (remove) a specific memory or memories.
        
        Args:
            message: Message with memory to forget
                'memory_id': ID of memory to forget
                'memory_type': Type of memory (optional, for optimization)
                'forget_type': 'soft' (mark as forgotten) or 'hard' (delete)
                
        Returns:
            Forgetting operation result
        """
        memory_id = message.get("memory_id")
        memory_type = message.get("memory_type")
        forget_type = message.get("forget_type", "soft")
        
        if not memory_id:
            return {
                "status": "error",
                "error": "No memory_id provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Remove from working memory if present
        for i, memory in enumerate(list(self.working_memory)):
            if memory.get("memory_id") == memory_id:
                self.working_memory.remove(memory)
                break
        
        # Remove from short-term memory if present
        for i, memory in enumerate(list(self.short_term)):
            if memory.get("memory_id") == memory_id:
                # Save memory type for vector store removal
                if not memory_type:
                    memory_type = memory.get("memory_type", MemoryType.EPISODIC)
                
                # Remove from themes
                for theme in memory.get("themes", []):
                    if theme in self.themes and memory_id in self.themes[theme]:
                        self.themes[theme].remove(memory_id)
                
                # Remove from short-term memory
                self.short_term.remove(memory)
                break
        
        # Remove from vector store if available
        if memory_type:
            if memory_type in self.vector_stores:
                self.vector_stores[memory_type].delete(memory_id)
        else:
            # If memory_type not provided, try all vector stores
            for store in self.vector_stores.values():
                store.delete(memory_id)
        
        # Handle long-term storage if available
        if self.storage_path:
            if forget_type == "hard":
                # Hard delete - completely remove
                success = self._delete_memory(memory_id)
            else:
                # Soft delete - mark as forgotten but keep
                memory = self._load_memory(memory_id)
                if memory:
                    memory["forgotten"] = True
                    memory["forget_time"] = time.time()
                    self._persist_memory(memory)
                    success = True
                else:
                    success = False
        else:
            # No long-term storage, consider successful if we reached here
            success = True
        
        if self.enable_logging:
            self.logger.info(f"Forgot memory {memory_id} (type: {forget_type})")
        
        return {
            "status": "success" if success else "error",
            "message": f"Memory {memory_id} {'forgotten' if success else 'not found'}",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _manual_consolidate(self) -> Dict[str, Any]:
        """
        Manually trigger memory consolidation.
        
        Returns:
            Consolidation operation result
        """
        consolidated = self._consolidate_memories()
        
        return {
            "status": "success",
            "consolidated_count": consolidated,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "short_term_count": len(self.short_term),
            "short_term_capacity": self.short_term.maxlen,
            "working_memory_count": len(self.working_memory),
            "working_memory_capacity": self.working_memory.maxlen,
            "theme_count": len(self.themes),
            "themes": list(self.themes.keys()),
            "consolidation_interval": self.consolidation_interval,
            "time_since_consolidation": time.time() - self.last_consolidation,
            "vector_stores": {}
        }
        
        # Add vector store stats
        for memory_type, store in self.vector_stores.items():
            stats["vector_stores"][memory_type] = store.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _persist_memory(self, memory: Dict[str, Any]) -> None:
        """
        Persist memory to long-term storage.
        
        Args:
            memory: Memory object to persist
        """
        if not self.storage_path:
            return
        
        try:
            memory_type = memory.get("memory_type", MemoryType.EPISODIC)
            memory_id = memory.get("memory_id")
            
            # Ensure the directory exists
            type_dir = os.path.join(self.storage_path, memory_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # Save the memory
            memory_path = os.path.join(type_dir, f"{memory_id}.json")
            with open(memory_path, "w") as f:
                json.dump(memory, f)
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to persist memory {memory.get('memory_id')}: {e}")
    
    def _load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory from long-term storage.
        
        Args:
            memory_id: ID of memory to load
            
        Returns:
            Memory object or None if not found
        """
        if not self.storage_path:
            return None
        
        # Try each memory type directory
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            memory_path = os.path.join(self.storage_path, memory_type, f"{memory_id}.json")
            
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Failed to load memory {memory_id}: {e}")
                    return None
        
        return None
    
    def _delete_memory(self, memory_id: str) -> bool:
        """
        Delete memory from long-term storage.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self.storage_path:
            return False
        
        # Try each memory type directory
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            memory_path = os.path.join(self.storage_path, memory_type, f"{memory_id}.json")
            
            if os.path.exists(memory_path):
                try:
                    os.remove(memory_path)
                    return True
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Failed to delete memory {memory_id}: {e}")
                    return False
        
        return False
    
    def _memory_matches_metadata(self, 
                                memory: Dict[str, Any], 
                                memory_type: Optional[str], 
                                themes: List[str], 
                                start_time: float, 
                                end_time: float, 
                                min_importance: float, 
                                source_agent: Optional[str]) -> bool:
        """
        Check if memory matches metadata criteria.
        
        Args:
            memory: Memory to check
            memory_type: Type of memory to match
            themes: List of themes to match
            start_time: Start timestamp
            end_time: End timestamp
            min_importance: Minimum importance
            source_agent: Source agent ID
            
        Returns:
            True if memory matches all criteria
        """
        # Check memory type if specified
        if memory_type and memory.get("memory_type") != memory_type:
            return False
        
        # Check timestamp range
        timestamp = memory.get("timestamp", 0)
        if timestamp < start_time or timestamp > end_time:
            return False
        
        # Check importance
        importance = memory.get("importance", 0.0)
        if importance < min_importance:
            return False
        
        # Check source agent if specified
        if source_agent and memory.get("source_agent") != source_agent:
            return False
        
        # Check if memory has all required themes
        if themes:
            memory_themes = set(memory.get("themes", []))
            if not all(theme in memory_themes for theme in themes):
                return False
        
        # Check if memory is marked as forgotten
        if memory.get("forgotten", False):
            return False
        
        return True
    
    def _query_long_term(self, 
                       memory_type: Optional[str], 
                       themes: List[str], 
                       start_time: float, 
                       end_time: float, 
                       min_importance: float, 
                       source_agent: Optional[str], 
                       max_results: int) -> List[Dict[str, Any]]:
        """
        Query long-term memory for matching memories.
        
        Args:
            memory_type: Type of memory to query
            themes: List of themes to filter by
            start_time: Start timestamp
            end_time: End timestamp
            min_importance: Minimum importance
            source_agent: Source agent ID
            max_results: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if not self.storage_path:
            return []
        
        results = []
        count = 0
        
        # Determine memory types to search
        memory_types = [memory_type] if memory_type else [
            MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL
        ]
        
        # Search each memory type directory
        for mem_type in memory_types:
            type_dir = os.path.join(self.storage_path, mem_type)
            if not os.path.exists(type_dir):
                continue
            
            # List all memory files
            memory_files = [f for f in os.listdir(type_dir) if f.endswith(".json")]
            
            # Check each memory file
            for file_name in memory_files:
                if count >= max_results:
                    break
                
                memory_path = os.path.join(type_dir, file_name)
                
                try:
                    with open(memory_path, "r") as f:
                        memory = json.load(f)
                        
                        if self._memory_matches_metadata(
                            memory, mem_type, themes, start_time, end_time, 
                            min_importance, source_agent
                        ):
                            results.append(memory)
                            count += 1
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Failed to read memory file {memory_path}: {e}")
        
        return results
    
    def _consolidate_memories(self) -> int:
        """
        Consolidate short-term memories to long-term storage with importance decay.
        
        Returns:
            Number of memories consolidated
        """
        if not self.storage_path:
            return 0
        
        consolidated_count = 0
        current_time = time.time()
        
        # First, save vector stores
        for memory_type, store in self.vector_stores.items():
            vector_path = os.path.join(self.storage_path, memory_type, "vectors")
            os.makedirs(vector_path, exist_ok=True)
            store.save(vector_path)
        
        # Process all short-term memories
        for memory in list(self.short_term):
            # Skip already forgotten memories
            if memory.get("forgotten", False):
                continue
            
            # Calculate memory decay based on importance, recency, and access count
            importance = memory.get("importance", 0.5)
            time_factor = min(1.0, (current_time - memory.get("timestamp", 0)) / (86400 * 30))  # 30 days
            access_count = memory.get("access_count", 0)
            access_bonus = min(0.5, access_count * 0.1)  # More accessed memories decay slower
            
            # Apply decay
            decay_amount = self.forgetting_factor * time_factor * (1.0 - access_bonus)
            new_importance = max(0.0, importance - decay_amount)
            
            # Update importance
            memory["importance"] = new_importance
            memory["last_consolidated"] = current_time
            
            # If importance drops too low, mark for forgetting
            if new_importance < 0.1:
                memory["forgotten"] = True
                memory["forget_time"] = current_time
            
            # Persist to long-term storage
            self._persist_memory(memory)
            consolidated_count += 1
            
            # Remove from short-term if older than a certain threshold
            # but not if in working memory
            if current_time - memory.get("timestamp", 0) > 86400:  # 1 day
                if not any(m.get("memory_id") == memory.get("memory_id") for m in self.working_memory):
                    self.short_term.remove(memory)
        
        if self.enable_logging:
            self.logger.info(f"Consolidated {consolidated_count} memories")
        
        return consolidated_count 