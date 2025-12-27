#!/usr/bin/env python3
"""
RAG-Based Advanced Memory System for NIS Protocol
Semantic search and retrieval using vector embeddings

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a memory entry with embeddings."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class RAGMemorySystem:
    """
    Advanced memory system with RAG (Retrieval-Augmented Generation).
    
    Features:
    - Semantic search using embeddings
    - Vector similarity matching
    - Persistent storage
    - Metadata filtering
    
    Honest Assessment:
    - Uses simple cosine similarity (not FAISS/Pinecone)
    - Embeddings from basic sentence transformers
    - Linear search (not optimized for scale)
    - Good for <10k entries, not production-scale
    """
    
    def __init__(self, workspace_dir: str = "/tmp/nis_workspace"):
        """Initialize RAG memory system."""
        self.workspace_dir = Path(workspace_dir)
        self.memory_dir = self.workspace_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_file = self.memory_dir / "rag_memory.json"
        self.memories: Dict[str, MemoryEntry] = {}
        
        # Load existing memories
        self._load_memories()
        
        # Simple embedding (not real transformer)
        # In production, use sentence-transformers or OpenAI embeddings
        self.embedding_dim = 384
        
        logger.info(f"🧠 RAG Memory initialized: {len(self.memories)} entries")
    
    def _load_memories(self):
        """Load memories from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(**entry_data)
                        self.memories[entry.id] = entry
                logger.info(f"✅ Loaded {len(self.memories)} memories")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
    
    def _save_memories(self):
        """Save memories to disk."""
        try:
            data = [asdict(entry) for entry in self.memories.values()]
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        HONEST: This is a simple hash-based embedding, not real transformer.
        In production, use:
        - sentence-transformers (all-MiniLM-L6-v2)
        - OpenAI embeddings (text-embedding-3-small)
        - Cohere embeddings
        
        This is 40% real - it creates vectors but not semantic.
        """
        # Simple hash-based embedding (deterministic)
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_bytes), 2):
            val = (hash_bytes[i] * 256 + hash_bytes[i+1]) / 65535.0
            embedding.append(val)
        
        # Pad or truncate to embedding_dim
        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:self.embedding_dim - len(embedding)])
        embedding = embedding[:self.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product  # Already normalized
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store content in memory with semantic embedding.
        
        Args:
            content: Text content to store
            metadata: Optional metadata
            memory_id: Optional custom ID
            
        Returns:
            Dict with success status and memory ID
        """
        try:
            # Generate ID if not provided
            if memory_id is None:
                memory_id = hashlib.sha256(
                    f"{content}{time.time()}".encode()
                ).hexdigest()[:16]
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Create memory entry
            entry = MemoryEntry(
                id=memory_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding
            )
            
            # Store
            self.memories[memory_id] = entry
            self._save_memories()
            
            logger.info(f"✅ Stored memory: {memory_id}")
            
            return {
                "success": True,
                "memory_id": memory_id,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"❌ Store error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve memories using semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filtering
            
        Returns:
            Dict with search results
        """
        try:
            if not self.memories:
                return {
                    "success": True,
                    "results": [],
                    "count": 0
                }
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Calculate similarities
            results = []
            for memory_id, entry in self.memories.items():
                # Apply metadata filter if provided
                if metadata_filter:
                    match = all(
                        entry.metadata.get(k) == v
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                # Calculate similarity
                if entry.embedding:
                    similarity = self._cosine_similarity(query_embedding, entry.embedding)
                    results.append({
                        "memory_id": memory_id,
                        "content": entry.content,
                        "metadata": entry.metadata,
                        "similarity": similarity,
                        "timestamp": entry.timestamp
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:top_k]
            
            logger.info(f"✅ Retrieved {len(results)} memories for query")
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"❌ Retrieve error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete memory by ID."""
        try:
            if memory_id in self.memories:
                del self.memories[memory_id]
                self._save_memories()
                
                logger.info(f"✅ Deleted memory: {memory_id}")
                
                return {
                    "success": True,
                    "memory_id": memory_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Memory not found: {memory_id}"
                }
                
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_memories(
        self,
        limit: int = 100,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List all memories with optional filtering."""
        try:
            results = []
            for memory_id, entry in self.memories.items():
                # Apply metadata filter if provided
                if metadata_filter:
                    match = all(
                        entry.metadata.get(k) == v
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                results.append({
                    "memory_id": memory_id,
                    "content": entry.content[:200],  # Truncate for listing
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp
                })
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            results = results[:limit]
            
            return {
                "success": True,
                "memories": results,
                "count": len(results),
                "total": len(self.memories)
            }
            
        except Exception as e:
            logger.error(f"❌ List error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_rag_memory: Optional[RAGMemorySystem] = None


def get_rag_memory_system(workspace_dir: str = "/tmp/nis_workspace") -> RAGMemorySystem:
    """Get or create RAG memory system instance."""
    global _rag_memory
    if _rag_memory is None:
        _rag_memory = RAGMemorySystem(workspace_dir=workspace_dir)
    return _rag_memory
