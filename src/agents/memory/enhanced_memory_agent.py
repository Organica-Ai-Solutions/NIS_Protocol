"""
Enhanced Memory Agent with LSTM Temporal Modeling

Advanced memory management including semantic search, memory organization, forgetting,
and LSTM-based temporal sequence modeling for enhanced memory prediction and consolidation.
Inspired by memory systems in cognitive architectures and neuroscience.

Enhanced Features (v3 + LSTM):
- LSTM-based temporal sequence modeling for memory patterns
- Attention mechanisms for selective memory retrieval and prediction
- Dynamic working memory management with temporal context
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of memory operations with evidence-based metrics
- Comprehensive integrity oversight for all memory outputs
- Auto-correction capabilities for memory-related communications
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

# PyTorch for LSTM functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension
from src.memory.vector_store import VectorStore
from src.memory.embedding_utils import get_embedding_provider, EmbeddingProvider

# LSTM core integration
from src.agents.memory.lstm_memory_core import (
    LSTMMemoryCore, MemorySequenceType, MemorySequence, LSTMMemoryState
)

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation


class MemoryType:
    """Memory types for different kinds of information."""
    EPISODIC = "episodic"  # Memories of specific events/experiences
    SEMANTIC = "semantic"  # General knowledge/facts
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Currently active information


class EnhancedMemoryAgent(NISAgent):
    """
    Enhanced agent for advanced memory management with LSTM temporal modeling.
    
    Features:
    - LSTM-based temporal sequence modeling for memory patterns
    - Attention mechanisms for selective memory retrieval and prediction
    - Semantic search using vector embeddings
    - Memory organization by themes and types
    - Memory consolidation and forgetting with temporal context
    - Importance-based retention with sequence learning
    - Time-based decay for relevance with LSTM prediction
    - Query by similarity, time, sequence patterns, or metadata
    """
    
    def __init__(
        self,
        agent_id: str = "memory",
        description: str = "Enhanced memory system with LSTM temporal modeling and semantic search",
        emotional_state: Optional[EmotionalState] = None,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_dimensions: int = 768,
        max_vectors: int = 100000,
        short_term_capacity: int = 1000,
        working_memory_limit: int = 10,
        consolidation_interval: int = 3600,  # 1 hour
        forgetting_factor: float = 0.05,
        enable_logging: bool = True,
        enable_self_audit: bool = True,
        # LSTM-specific parameters
        enable_lstm: bool = True,
        lstm_hidden_dim: int = 512,
        lstm_num_layers: int = 2,
        lstm_learning_rate: float = 0.001,
        max_sequence_length: int = 100,
        lstm_device: str = "cpu"
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
            enable_self_audit: Whether to enable real-time integrity monitoring
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
        
        # LSTM temporal modeling integration
        self.enable_lstm = enable_lstm and TORCH_AVAILABLE
        self.lstm_core = None
        
        if self.enable_lstm:
            try:
                self.lstm_core = LSTMMemoryCore(
                    memory_dim=vector_dimensions,
                    hidden_dim=lstm_hidden_dim,
                    num_layers=lstm_num_layers,
                    max_sequence_length=max_sequence_length,
                    learning_rate=lstm_learning_rate,
                    device=lstm_device,
                    enable_self_audit=enable_self_audit
                )
                
                # LSTM-enhanced working memory
                self.temporal_working_memory = deque(maxlen=working_memory_limit * 2)  # Larger for sequences
                self.active_memory_sequences = {}
                self.sequence_predictions = {}
                
                if enable_logging:
                    self.logger.info(f"LSTM temporal modeling enabled with hidden_dim={lstm_hidden_dim}")
            except Exception as e:
                self.enable_lstm = False
                if enable_logging:
                    self.logger.warning(f"Failed to initialize LSTM core: {e}")
        
        if not self.enable_lstm and enable_logging:
            self.logger.info("LSTM temporal modeling disabled - using traditional memory management")
        
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
        
        # Set up self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_monitoring_enabled = enable_self_audit
        self.integrity_metrics = {
            'monitoring_start_time': time.time(),
            'total_outputs_monitored': 0,
            'total_violations_detected': 0,
            'auto_corrections_applied': 0,
            'average_integrity_score': 100.0
        }
        
        # Initialize confidence factors for mathematical validation
        self.confidence_factors = create_default_confidence_factors()
        
        if enable_logging:
            self.logger.info(f"Enhanced Memory Agent initialized with self-audit: {enable_self_audit}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request with integrated self-audit monitoring.
        
        Args:
            message: Message containing memory operation
                'operation': Operation to perform 
                    ('store', 'retrieve', 'query', 'search', 'forget', 'consolidate', 'stats')
                + Additional parameters based on operation
        
        Returns:
            Result of the memory operation with integrity monitoring
        """
        if not self._validate_message(message):
            error_response = {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            # Apply self-audit monitoring to error response
            if self.enable_self_audit:
                error_text = error_response.get("error", "")
                error_response["error"] = self._monitor_memory_output_integrity(error_text, "validation_error")
            
            return error_response

        # Check if it's time to consolidate memories
        current_time = time.time()
        if current_time - self.last_consolidation > self.consolidation_interval:
            self._consolidate_memories()
            self.last_consolidation = current_time

        operation = message.get("operation", "").lower()

        # Route to appropriate handler based on operation
        try:
            if operation == "store":
                result = self._store_memory(message)
            elif operation == "retrieve":
                result = self._retrieve_memory(message)
            elif operation == "query":
                result = self._query_memory(message)
            elif operation == "search":
                result = self._semantic_search(message)
            elif operation == "forget":
                result = self._forget_memory(message)
            elif operation == "consolidate":
                result = self._manual_consolidate()
            elif operation == "stats":
                result = self._get_stats()
            # LSTM-enhanced operations
            elif operation == "predict_next":
                result = self._predict_next_memory(message)
            elif operation == "predict_sequence":
                result = self._predict_memory_sequence(message)
            elif operation == "lstm_stats":
                result = self._get_lstm_stats()
            elif operation == "temporal_context":
                result = self._get_temporal_context(message)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Apply self-audit monitoring to all responses
            if self.enable_self_audit and result:
                result = self._apply_memory_integrity_monitoring(result, operation)
            
            return result
            
        except Exception as e:
            error_response = {
                "status": "error", 
                "error": f"Memory operation failed: {str(e)}",
                "operation": operation,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            # Apply self-audit monitoring to exception response
            if self.enable_self_audit:
                error_text = error_response.get("error", "")
                error_response["error"] = self._monitor_memory_output_integrity(error_text, f"{operation}_error")
            
            if self.enable_logging:
                self.logger.error(f"Memory operation {operation} failed: {str(e)}")
            
            return error_response
    
    def _apply_memory_integrity_monitoring(self, result: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """
        Apply integrity monitoring to memory operation results.
        
        Args:
            result: Original result dictionary
            operation: Memory operation that was performed
            
        Returns:
            Result with integrity monitoring applied
        """
        # Monitor text fields in the response for integrity
        monitored_fields = ["error", "message", "description"]
        
        for field in monitored_fields:
            if field in result and isinstance(result[field], str):
                result[field] = self._monitor_memory_output_integrity(result[field], operation)
        
        # Monitor memory content if present
        if "memory" in result and isinstance(result["memory"], dict):
            memory = result["memory"]
            if "content" in memory and isinstance(memory["content"], str):
                memory["content"] = self._monitor_memory_output_integrity(memory["content"], f"{operation}_content")
        
        # Monitor search results content if present
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                    item["content"] = self._monitor_memory_output_integrity(item["content"], f"{operation}_search_result")
        
        # Add integrity metadata to result
        if hasattr(self, 'integrity_metrics') and self.integrity_metrics['total_outputs_monitored'] > 0:
            result["integrity_metadata"] = {
                "monitoring_enabled": True,
                "outputs_monitored": self.integrity_metrics['total_outputs_monitored'],
                "violations_detected": self.integrity_metrics['total_violations_detected'],
                "auto_corrections_applied": self.integrity_metrics['auto_corrections_applied']
            }
        
        return result
    
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
        
        # LSTM temporal sequence learning integration
        if self.enable_lstm and self.lstm_core and memory["embedding_available"]:
            try:
                # Determine sequence type based on memory type and source
                if memory_type == MemoryType.EPISODIC:
                    sequence_type = MemorySequenceType.EPISODIC_SEQUENCE
                elif memory_type == MemoryType.SEMANTIC:
                    sequence_type = MemorySequenceType.SEMANTIC_PATTERN
                elif memory_type == MemoryType.PROCEDURAL:
                    sequence_type = MemorySequenceType.PROCEDURAL_CHAIN
                else:
                    sequence_type = MemorySequenceType.CONTEXTUAL_FLOW
                
                # Add memory to LSTM sequence for temporal learning
                lstm_memory_data = {
                    'memory_id': memory_id,
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    'importance': importance,
                    'timestamp': timestamp,
                    'memory_type': memory_type,
                    'source_agent': source_agent,
                    'themes': themes
                }
                
                sequence_id = self.lstm_core.add_memory_to_sequence(
                    memory_data=lstm_memory_data,
                    sequence_type=sequence_type
                )
                
                # Track sequence for this memory
                memory["lstm_sequence_id"] = sequence_id
                self.active_memory_sequences[memory_id] = sequence_id
                
                # Add to temporal working memory for LSTM context
                if add_to_working or importance > 0.6:
                    self.temporal_working_memory.append({
                        'memory_id': memory_id,
                        'sequence_id': sequence_id,
                        'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        'importance': importance,
                        'timestamp': timestamp
                    })
                
                if self.enable_logging:
                    self.logger.debug(f"Added memory {memory_id} to LSTM sequence {sequence_id}")
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Failed to add memory {memory_id} to LSTM sequence: {e}")
        
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
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_memory_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on memory operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Memory operation type (store, retrieve, query, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        if self.enable_logging:
            self.logger.info(f"Performing self-audit on memory output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"memory:{operation}:{context}" if context else f"memory:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for memory-specific analysis
        if violations and self.enable_logging:
            self.logger.warning(f"Detected {len(violations)} integrity violations in memory output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_memory_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_memory_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in memory outputs.
        
        Args:
            output_text: Text to correct
            operation: Memory operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        if self.enable_logging:
            self.logger.info(f"Performing self-correction on memory output for operation: {operation}")
        
        corrected_text, violations = self_audit_engine.auto_correct_text(output_text)
        
        # Calculate improvement metrics with mathematical validation
        original_score = self_audit_engine.get_integrity_score(output_text)
        corrected_score = self_audit_engine.get_integrity_score(corrected_text)
        improvement = calculate_confidence(corrected_score - original_score, self.confidence_factors)
        
        # Update integrity metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['auto_corrections_applied'] += len(violations)
        
        return {
            'original_text': output_text,
            'corrected_text': corrected_text,
            'violations_fixed': violations,
            'original_integrity_score': original_score,
            'corrected_integrity_score': corrected_score,
            'improvement': improvement,
            'operation': operation,
            'correction_timestamp': time.time()
        }
    
    def analyze_memory_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze memory operation integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Memory integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        if self.enable_logging:
            self.logger.info(f"Analyzing memory integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate memory-specific metrics
        memory_metrics = {
            'total_memory_operations': getattr(self, '_operation_count', 0),
            'memory_types_used': len(self.vector_stores),
            'short_term_utilization': len(self.short_term) / self.short_term.maxlen if self.short_term.maxlen else 0,
            'working_memory_utilization': len(self.working_memory) / self.working_memory.maxlen if self.working_memory.maxlen else 0
        }
        
        # Generate memory-specific recommendations
        recommendations = self._generate_memory_integrity_recommendations(
            integrity_report, memory_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'memory_metrics': memory_metrics,
            'integrity_trend': self._calculate_memory_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def enable_real_time_memory_monitoring(self) -> bool:
        """
        Enable continuous integrity monitoring for all memory operations.
        
        Returns:
            Success status
        """
        if self.enable_logging:
            self.logger.info("Enabling real-time memory integrity monitoring")
        
        # Set flag for monitoring
        self.integrity_monitoring_enabled = True
        
        # Initialize monitoring metrics if not already done
        if not hasattr(self, 'integrity_metrics'):
            self.integrity_metrics = {
                'monitoring_start_time': time.time(),
                'total_outputs_monitored': 0,
                'total_violations_detected': 0,
                'auto_corrections_applied': 0,
                'average_integrity_score': 100.0
            }
        
        return True
    
    def _monitor_memory_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct memory output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Memory operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_memory_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_memory_output(output_text, operation)
            
            if self.enable_logging:
                self.logger.info(f"Auto-corrected memory output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_memory_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to memory operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_memory_integrity_recommendations(self, integrity_report: Dict[str, Any], 
                                                 memory_metrics: Dict[str, Any]) -> List[str]:
        """Generate memory-specific integrity recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 0:
            recommendations.append('Review memory operation outputs for integrity compliance')
        
        if memory_metrics.get('short_term_utilization', 0) > 0.9:
            recommendations.append('Consider increasing short-term memory capacity or consolidation frequency')
        
        if memory_metrics.get('working_memory_utilization', 0) > 0.8:
            recommendations.append('Monitor working memory usage to prevent overflow')
        
        recommendations.append('Maintain evidence-based memory operation descriptions')
        
        return recommendations
    
    def _calculate_memory_integrity_trend(self) -> str:
        """Calculate memory integrity trend over time"""
        if not hasattr(self, 'integrity_metrics'):
            return 'INSUFFICIENT_DATA'
        
        # Simple trend calculation based on recent performance
        total_monitored = self.integrity_metrics.get('total_outputs_monitored', 0)
        total_violations = self.integrity_metrics.get('total_violations_detected', 0)
        
        if total_monitored == 0:
            return 'NO_DATA'
        
        violation_rate = total_violations / total_monitored
        
        if violation_rate == 0:
            return 'EXCELLENT'
        elif violation_rate < 0.1:
            return 'GOOD'
        elif violation_rate < 0.2:
            return 'NEEDS_IMPROVEMENT'
        else:
            return 'CRITICAL'
    
    def get_memory_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add memory-specific metrics
        memory_report = {
            'memory_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'memory_capacity_status': {
                'short_term': f"{len(self.short_term)}/{self.short_term.maxlen}" if self.short_term.maxlen else "unlimited",
                'working_memory': f"{len(self.working_memory)}/{self.working_memory.maxlen}" if self.working_memory.maxlen else "unlimited"
            },
            'memory_types_active': list(self.vector_stores.keys()),
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return memory_report
    
    # =====================
    # LSTM-Enhanced Methods
    # =====================
    
    def _predict_next_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the next memory in a sequence using LSTM.
        
        Args:
            message: Message with prediction parameters
                'sequence_id': Optional specific sequence ID
                'memory_id': Memory ID to predict from
                'context': Optional context for prediction
                
        Returns:
            Prediction result with attention weights and confidence
        """
        if not self.enable_lstm or not self.lstm_core:
            return {
                "status": "error",
                "error": "LSTM temporal modeling not available",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Get sequence ID
        sequence_id = message.get("sequence_id")
        memory_id = message.get("memory_id")
        context = message.get("context", {})
        
        # Find sequence if memory_id provided
        if not sequence_id and memory_id:
            sequence_id = self.active_memory_sequences.get(memory_id)
        
        # If still no sequence, use most recent active sequence
        if not sequence_id:
            if self.active_memory_sequences:
                # Get most recently accessed sequence
                latest_memory = max(self.active_memory_sequences.items(), 
                                  key=lambda x: self._get_memory_timestamp(x[0]))
                sequence_id = latest_memory[1]
            else:
                return {
                    "status": "error",
                    "error": "No active memory sequences found for prediction",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        try:
            # Get prediction from LSTM core
            prediction_result = self.lstm_core.predict_next_memory(sequence_id, context)
            
            # Store prediction for validation
            self.sequence_predictions[sequence_id] = {
                'prediction': prediction_result,
                'timestamp': time.time(),
                'context': context
            }
            
            return {
                "status": "success",
                "sequence_id": sequence_id,
                "prediction": {
                    "predicted_embedding": prediction_result.get("predicted_embedding"),
                    "confidence": prediction_result.get("confidence", 0.0),
                    "attention_weights": prediction_result.get("attention_weights", []),
                    "attention_coherence": prediction_result.get("attention_coherence", 0.0),
                    "temporal_position": prediction_result.get("temporal_position", 0),
                },
                "lstm_metadata": prediction_result.get("processing_metadata", {}),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to predict next memory for sequence {sequence_id}: {e}")
            
            return {
                "status": "error",
                "error": f"Prediction failed: {str(e)}",
                "sequence_id": sequence_id,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _predict_memory_sequence(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict multiple memories in a sequence.
        
        Args:
            message: Message with sequence prediction parameters
                'sequence_id': Sequence to predict for
                'prediction_length': Number of memories to predict (default: 3)
                'context': Optional context for prediction
                
        Returns:
            Sequence of predictions with confidence scores
        """
        if not self.enable_lstm or not self.lstm_core:
            return {
                "status": "error",
                "error": "LSTM temporal modeling not available",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        sequence_id = message.get("sequence_id")
        prediction_length = message.get("prediction_length", 3)
        context = message.get("context", {})
        
        if not sequence_id:
            return {
                "status": "error",
                "error": "sequence_id required for sequence prediction",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        try:
            predictions = []
            current_sequence_id = sequence_id
            
            # Generate sequence of predictions
            for i in range(prediction_length):
                prediction_result = self.lstm_core.predict_next_memory(current_sequence_id, context)
                
                predictions.append({
                    "step": i + 1,
                    "predicted_embedding": prediction_result.get("predicted_embedding"),
                    "confidence": prediction_result.get("confidence", 0.0),
                    "attention_weights": prediction_result.get("attention_weights", []),
                    "attention_coherence": prediction_result.get("attention_coherence", 0.0)
                })
                
                # Update context for next prediction
                context["previous_prediction"] = prediction_result
            
            # Calculate overall sequence confidence
            overall_confidence = np.mean([p["confidence"] for p in predictions])
            overall_coherence = np.mean([p["attention_coherence"] for p in predictions])
            
            return {
                "status": "success",
                "sequence_id": sequence_id,
                "prediction_length": prediction_length,
                "predictions": predictions,
                "overall_confidence": float(overall_confidence),
                "overall_coherence": float(overall_coherence),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to predict memory sequence {sequence_id}: {e}")
            
            return {
                "status": "error",
                "error": f"Sequence prediction failed: {str(e)}",
                "sequence_id": sequence_id,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _get_lstm_stats(self) -> Dict[str, Any]:
        """
        Get LSTM performance statistics and metrics.
        
        Returns:
            Comprehensive LSTM statistics
        """
        if not self.enable_lstm or not self.lstm_core:
            return {
                "status": "error",
                "error": "LSTM temporal modeling not available",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        try:
            # Get core LSTM statistics
            lstm_stats = self.lstm_core.get_sequence_statistics()
            
            # Add agent-specific LSTM metrics
            agent_lstm_stats = {
                "lstm_enabled": self.enable_lstm,
                "active_sequences": len(self.active_memory_sequences),
                "temporal_working_memory_size": len(self.temporal_working_memory) if hasattr(self, 'temporal_working_memory') else 0,
                "prediction_cache_size": len(self.sequence_predictions) if hasattr(self, 'sequence_predictions') else 0,
                "memory_types_with_sequences": self._count_sequences_by_type(),
                "lstm_integration_performance": {
                    "total_memories_in_sequences": sum(1 for _ in self.active_memory_sequences),
                    "average_sequence_age": self._calculate_average_sequence_age(),
                    "prediction_accuracy_trend": self._calculate_prediction_accuracy_trend()
                }
            }
            
            # Combine statistics
            combined_stats = {
                "status": "success",
                "lstm_core_stats": lstm_stats,
                "agent_lstm_stats": agent_lstm_stats,
                "integration_health": "good" if lstm_stats.get("total_sequences", 0) > 0 else "initializing",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            return combined_stats
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to get LSTM stats: {e}")
            
            return {
                "status": "error",
                "error": f"Failed to retrieve LSTM stats: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _get_temporal_context(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get temporal context for memory operations using LSTM.
        
        Args:
            message: Message with context parameters
                'memory_id': Memory to get context for
                'context_window': Size of context window (default: 5)
                
        Returns:
            Temporal context information
        """
        if not self.enable_lstm or not self.lstm_core:
            return {
                "status": "error",
                "error": "LSTM temporal modeling not available",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        memory_id = message.get("memory_id")
        context_window = message.get("context_window", 5)
        
        if not memory_id:
            return {
                "status": "error",
                "error": "memory_id required for temporal context",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        try:
            # Get sequence for memory
            sequence_id = self.active_memory_sequences.get(memory_id)
            if not sequence_id:
                return {
                    "status": "error",
                    "error": f"No sequence found for memory {memory_id}",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Get temporal context from working memory
            temporal_context = []
            current_time = time.time()
            
            # Get recent memories from temporal working memory
            for entry in list(self.temporal_working_memory)[-context_window:]:
                if entry.get("sequence_id") == sequence_id:
                    temporal_context.append({
                        "memory_id": entry["memory_id"],
                        "importance": entry["importance"],
                        "timestamp": entry["timestamp"],
                        "time_since": current_time - entry["timestamp"],
                        "relative_position": len(temporal_context)
                    })
            
            # Get prediction if available
            prediction = self.sequence_predictions.get(sequence_id)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "sequence_id": sequence_id,
                "temporal_context": temporal_context,
                "context_window_size": len(temporal_context),
                "recent_prediction": prediction,
                "context_coherence": self._calculate_context_coherence(temporal_context),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to get temporal context for memory {memory_id}: {e}")
            
            return {
                "status": "error",
                "error": f"Failed to get temporal context: {str(e)}",
                "memory_id": memory_id,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    # Helper methods for LSTM integration
    
    def _get_memory_timestamp(self, memory_id: str) -> float:
        """Get timestamp for a memory ID"""
        # Search in short-term memory
        for memory in self.short_term:
            if memory.get("memory_id") == memory_id:
                return memory.get("timestamp", 0.0)
        
        # Search in working memory
        for memory in self.working_memory:
            if memory.get("memory_id") == memory_id:
                return memory.get("timestamp", 0.0)
        
        return 0.0
    
    def _count_sequences_by_type(self) -> Dict[str, int]:
        """Count active sequences by memory type"""
        if not hasattr(self, 'lstm_core') or not self.lstm_core:
            return {}
        
        type_counts = defaultdict(int)
        for sequence in self.lstm_core.memory_sequences.values():
            type_counts[sequence.sequence_type.value] += 1
        
        return dict(type_counts)
    
    def _calculate_average_sequence_age(self) -> float:
        """Calculate average age of active sequences"""
        if not hasattr(self, 'lstm_core') or not self.lstm_core:
            return 0.0
        
        current_time = time.time()
        ages = []
        
        for sequence in self.lstm_core.memory_sequences.values():
            if sequence.temporal_order:
                first_timestamp = min(sequence.temporal_order)
                age = current_time - first_timestamp
                ages.append(age)
        
        return np.mean(ages) if ages else 0.0
    
    def _calculate_prediction_accuracy_trend(self) -> str:
        """Calculate trend in prediction accuracy"""
        if not hasattr(self, 'lstm_core') or not self.lstm_core:
            return "no_data"
        
        recent_accuracy = list(self.lstm_core.prediction_accuracy)[-10:] if self.lstm_core.prediction_accuracy else []
        
        if len(recent_accuracy) < 3:
            return "insufficient_data"
        
        # Simple trend calculation
        early_avg = np.mean(recent_accuracy[:len(recent_accuracy)//2])
        late_avg = np.mean(recent_accuracy[len(recent_accuracy)//2:])
        
        if late_avg > early_avg + 0.05:
            return "improving"
        elif late_avg < early_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_context_coherence(self, temporal_context: List[Dict[str, Any]]) -> float:
        """Calculate coherence of temporal context"""
        if len(temporal_context) < 2:
            return 1.0
        
        # Simple coherence based on temporal ordering and importance
        importance_values = [ctx["importance"] for ctx in temporal_context]
        time_gaps = [temporal_context[i+1]["time_since"] - temporal_context[i]["time_since"] 
                    for i in range(len(temporal_context)-1)]
        
        # Coherence is higher when importance is consistent and time gaps are regular
        importance_coherence = 1.0 - np.std(importance_values)
        temporal_coherence = 1.0 - (np.std(time_gaps) / max(np.mean(time_gaps), 1.0)) if time_gaps else 1.0
        
        return (importance_coherence + temporal_coherence) / 2.0 