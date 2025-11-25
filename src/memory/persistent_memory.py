"""
Persistent Memory System - NIS Protocol v4.0
Long-term memory with episodic and semantic storage.
Implements the "Continuum Memory" concept from Nested Learning.
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np

logger = logging.getLogger("nis.persistent_memory")

# Try to import vector DB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available - using in-memory fallback")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    timestamp: float
    importance: float  # 0-1, how important/surprising
    access_count: int = 0
    last_accessed: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class MemorySearchResult:
    """Result from memory search"""
    entry: MemoryEntry
    relevance_score: float
    recency_score: float
    importance_score: float
    combined_score: float


class PersistentMemorySystem:
    """
    Long-term memory system with multiple memory types:
    - Episodic: Specific events/conversations
    - Semantic: General knowledge/facts
    - Procedural: How to do things (patterns learned)
    
    Uses vector similarity for retrieval with importance-weighted ranking.
    """
    
    def __init__(
        self,
        storage_path: str = "data/memory",
        collection_name: str = "nis_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_memories: int = 10000,
        importance_decay: float = 0.99,  # Daily decay
        recency_weight: float = 0.3,
        importance_weight: float = 0.4,
        relevance_weight: float = 0.3
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.max_memories = max_memories
        self.importance_decay = importance_decay
        
        # Scoring weights
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight
        
        # Initialize embedding model
        self.embedder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                logger.info(f"✅ Embedding model loaded: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
        
        # Initialize vector store
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.storage_path / "chromadb"),
                    anonymized_telemetry=False
                ))
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ ChromaDB initialized: {collection_name}")
            except Exception as e:
                logger.warning(f"ChromaDB init failed: {e}")
        
        # Fallback in-memory storage
        self.memory_store: Dict[str, MemoryEntry] = {}
        self._load_fallback_store()
        
        # Statistics
        self.stats = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "procedural_count": 0,
            "total_retrievals": 0,
            "avg_retrieval_time_ms": 0
        }
        
        logger.info(f"🧠 PersistentMemorySystem initialized: {len(self.memory_store)} memories loaded")
    
    def _generate_id(self, content: str, memory_type: str) -> str:
        """Generate unique ID for memory"""
        hash_input = f"{content[:100]}:{memory_type}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        if self.embedder:
            try:
                embedding = self.embedder.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        return None
    
    def _load_fallback_store(self):
        """Load memories from JSON fallback"""
        fallback_path = self.storage_path / "memory_fallback.json"
        if fallback_path.exists():
            try:
                with open(fallback_path, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(**entry_data)
                        self.memory_store[entry.id] = entry
            except Exception as e:
                logger.warning(f"Failed to load fallback store: {e}")
    
    def _save_fallback_store(self):
        """Save memories to JSON fallback"""
        fallback_path = self.storage_path / "memory_fallback.json"
        try:
            data = []
            for entry in self.memory_store.values():
                entry_dict = {
                    "id": entry.id,
                    "content": entry.content,
                    "memory_type": entry.memory_type,
                    "timestamp": entry.timestamp,
                    "importance": entry.importance,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "metadata": entry.metadata,
                    "embedding": None  # Don't save embeddings to JSON
                }
                data.append(entry_dict)
            with open(fallback_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save fallback store: {e}")
    
    async def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new memory"""
        memory_id = self._generate_id(content, memory_type)
        timestamp = time.time()
        
        # Get embedding
        embedding = self._get_embedding(content)
        
        # Create entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            importance=importance,
            access_count=0,
            last_accessed=timestamp,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # Store in ChromaDB if available
        if self.collection and embedding:
            try:
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{
                        "memory_type": memory_type,
                        "importance": importance,
                        "timestamp": timestamp,
                        **(metadata or {})
                    }]
                )
            except Exception as e:
                logger.warning(f"ChromaDB store failed: {e}")
        
        # Always store in fallback
        self.memory_store[memory_id] = entry
        
        # Update stats
        self.stats["total_memories"] = len(self.memory_store)
        self.stats[f"{memory_type}_count"] = self.stats.get(f"{memory_type}_count", 0) + 1
        
        # Periodic save
        if len(self.memory_store) % 10 == 0:
            self._save_fallback_store()
        
        logger.debug(f"📝 Stored {memory_type} memory: {memory_id}")
        return memory_id
    
    async def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_relevance: float = 0.3
    ) -> List[MemorySearchResult]:
        """Retrieve relevant memories"""
        start_time = time.time()
        results = []
        
        query_embedding = self._get_embedding(query)
        
        # Try ChromaDB first
        if self.collection and query_embedding:
            try:
                where_filter = {"memory_type": memory_type} if memory_type else None
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,  # Get more, then filter
                    where=where_filter
                )
                
                if chroma_results and chroma_results['ids']:
                    for i, doc_id in enumerate(chroma_results['ids'][0]):
                        if doc_id in self.memory_store:
                            entry = self.memory_store[doc_id]
                            distance = chroma_results['distances'][0][i] if chroma_results.get('distances') else 0.5
                            relevance = 1.0 - min(distance, 1.0)
                            
                            if relevance >= min_relevance:
                                result = self._score_memory(entry, relevance)
                                results.append(result)
            except Exception as e:
                logger.warning(f"ChromaDB query failed: {e}")
        
        # Fallback: Simple keyword matching
        if not results:
            query_words = set(query.lower().split())
            for entry in self.memory_store.values():
                if memory_type and entry.memory_type != memory_type:
                    continue
                
                content_words = set(entry.content.lower().split())
                overlap = len(query_words & content_words)
                relevance = overlap / max(len(query_words), 1)
                
                if relevance >= min_relevance:
                    result = self._score_memory(entry, relevance)
                    results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        results = results[:top_k]
        
        # Update access counts
        for result in results:
            result.entry.access_count += 1
            result.entry.last_accessed = time.time()
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self.stats["total_retrievals"] += 1
        self.stats["avg_retrieval_time_ms"] = (
            (self.stats["avg_retrieval_time_ms"] * (self.stats["total_retrievals"] - 1) + elapsed)
            / self.stats["total_retrievals"]
        )
        
        return results
    
    def _score_memory(self, entry: MemoryEntry, relevance: float) -> MemorySearchResult:
        """Score a memory based on relevance, recency, and importance"""
        now = time.time()
        
        # Recency score (exponential decay over days)
        age_days = (now - entry.timestamp) / 86400
        recency_score = np.exp(-age_days / 30)  # 30-day half-life
        
        # Importance with decay
        decayed_importance = entry.importance * (self.importance_decay ** age_days)
        
        # Combined score
        combined = (
            self.relevance_weight * relevance +
            self.recency_weight * recency_score +
            self.importance_weight * decayed_importance
        )
        
        return MemorySearchResult(
            entry=entry,
            relevance_score=relevance,
            recency_score=recency_score,
            importance_score=decayed_importance,
            combined_score=combined
        )
    
    async def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        importance: float = 0.5,
        conversation_id: Optional[str] = None
    ) -> str:
        """Store a conversation exchange as episodic memory"""
        content = f"User: {user_message}\nAssistant: {assistant_response}"
        return await self.store(
            content=content,
            memory_type="episodic",
            importance=importance,
            metadata={
                "conversation_id": conversation_id,
                "user_message": user_message[:200],
                "response_preview": assistant_response[:200]
            }
        )
    
    async def store_knowledge(
        self,
        fact: str,
        source: str = "learned",
        confidence: float = 0.8
    ) -> str:
        """Store a fact as semantic memory"""
        return await self.store(
            content=fact,
            memory_type="semantic",
            importance=confidence,
            metadata={"source": source, "confidence": confidence}
        )
    
    async def store_pattern(
        self,
        pattern_description: str,
        success_rate: float = 0.5
    ) -> str:
        """Store a learned pattern as procedural memory"""
        return await self.store(
            content=pattern_description,
            memory_type="procedural",
            importance=success_rate,
            metadata={"success_rate": success_rate}
        )
    
    async def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 1000
    ) -> str:
        """Get relevant context from memory for a query"""
        results = await self.retrieve(query, top_k=5)
        
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate
        
        for result in results:
            entry = result.entry
            if total_chars + len(entry.content) > char_limit:
                break
            
            prefix = {
                "episodic": "📅 Past conversation",
                "semantic": "📚 Known fact",
                "procedural": "🔧 Learned pattern"
            }.get(entry.memory_type, "💭 Memory")
            
            context_parts.append(f"{prefix} (relevance: {result.relevance_score:.2f}):\n{entry.content}")
            total_chars += len(entry.content)
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            **self.stats,
            "chromadb_available": self.collection is not None,
            "embeddings_available": self.embedder is not None,
            "storage_path": str(self.storage_path)
        }
    
    async def consolidate(self):
        """Consolidate memories - merge similar, prune low-importance"""
        # Prune old, low-importance memories if over limit
        if len(self.memory_store) > self.max_memories:
            entries = list(self.memory_store.values())
            entries.sort(key=lambda e: e.importance * (0.99 ** ((time.time() - e.timestamp) / 86400)))
            
            # Remove bottom 10%
            to_remove = entries[:len(entries) // 10]
            for entry in to_remove:
                del self.memory_store[entry.id]
                if self.collection:
                    try:
                        self.collection.delete(ids=[entry.id])
                    except:
                        pass
            
            logger.info(f"🧹 Consolidated memory: removed {len(to_remove)} low-importance entries")
        
        self._save_fallback_store()


# Global instance
_memory_system: Optional[PersistentMemorySystem] = None


def get_memory_system() -> PersistentMemorySystem:
    """Get or create the global memory system"""
    global _memory_system
    if _memory_system is None:
        _memory_system = PersistentMemorySystem()
    return _memory_system


async def create_memory_system(**kwargs) -> PersistentMemorySystem:
    """Create a new memory system with custom config"""
    global _memory_system
    _memory_system = PersistentMemorySystem(**kwargs)
    return _memory_system
