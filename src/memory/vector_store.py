"""
NIS Protocol Vector Store

This module provides vector database functionality for semantic search in the memory system,
enabling more powerful and flexible retrieval based on semantic similarity.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

# Try to import hnswlib, fall back to simple implementation if not available
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError as e:
    print(f"hnswlib not available ({e}), using simple vector store implementation")
    HNSWLIB_AVAILABLE = False
    # Import the simple fallback
    from .simple_vector_store import SimpleVectorStore

# Try to import production vector databases
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

class VectorStore:
    """
    Vector database implementation for semantic search of memories.
    
    This class provides the ability to store and query memory embeddings
    for semantic similarity search, which is crucial for context-aware
    memory retrieval in the NIS Protocol.
    
    Automatically chooses between hnswlib (preferred) and simple numpy implementation
    based on availability.
    """
    
    def __init__(self, dim: int = 768, max_elements: int = 100000, space: str = "cosine", ef_search: int = 50):
        """
        Initialize the vector store with automatic backend selection.
        
        Backend Priority:
        1. Pinecone (if PINECONE_API_KEY set)
        2. Weaviate (if WEAVIATE_URL set)
        3. HNSW (if hnswlib available)
        4. Simple (fallback)
        
        Args:
            dim: Dimensionality of the embedding vectors
            max_elements: Maximum number of vectors to store
            space: Distance metric to use (cosine, l2, ip)
            ef_search: Query-time accuracy parameter
        """
        # Check for production vector DB config
        vector_backend = os.getenv("VECTOR_STORE_BACKEND", "auto").lower()
        pinecone_key = os.getenv("PINECONE_API_KEY", "")
        weaviate_url = os.getenv("WEAVIATE_URL", "")
        
        # Auto-select best available backend
        if vector_backend == "pinecone" or (vector_backend == "auto" and pinecone_key and PINECONE_AVAILABLE):
            print(f"✅ Using Pinecone vector database (production)")
            self._impl = PineconeVectorStore(dim, max_elements, space)
        elif vector_backend == "weaviate" or (vector_backend == "auto" and weaviate_url and WEAVIATE_AVAILABLE):
            print(f"✅ Using Weaviate vector database (production)")
            self._impl = WeaviateVectorStore(dim, max_elements, space)
        elif HNSWLIB_AVAILABLE:
            print(f"✅ Using HNSW vector store (local)")
            self._impl = HNSWVectorStore(dim, max_elements, space, ef_search)
        else:
            print(f"⚠️  Using simple vector store (fallback)")
            # Map ip space to cosine for simple implementation
            simple_space = "cosine" if space == "ip" else space
            self._impl = SimpleVectorStore(dim, max_elements, simple_space)
        
        # Expose implementation attributes for compatibility
        self.dim = self._impl.dim
        self.max_elements = self._impl.max_elements
        self.space = self._impl.space
    
    def add(self, memory_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> int:
        """Add a vector embedding to the store."""
        return self._impl.add(memory_id, vector, metadata)
    
    def search(self, 
               query_vector: np.ndarray, 
               top_k: int = 10, 
               filter_categories: Dict[str, Any] = None) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search for vectors similar to the query vector."""
        return self._impl.search(query_vector, top_k, filter_categories)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a vector from the store."""
        return self._impl.delete(memory_id)
    
    def get_by_id(self, memory_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get a vector by memory ID."""
        return self._impl.get_by_id(memory_id)
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        return self._impl.save(path)
    
    def load(self, path: str) -> bool:
        """Load the vector store from disk."""
        return self._impl.load(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self._impl.get_stats()


class HNSWVectorStore:
    """
    HNSW-based vector store implementation using hnswlib.
    """
    
    def __init__(self, dim: int = 768, max_elements: int = 100000, space: str = "cosine", ef_search: int = 50):
        self.dim = dim
        self.max_elements = max_elements
        self.space = space
        self.ef_search = ef_search
        
        # Initialize index with HNSW algorithm
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(ef_search)
        
        # Track mappings between vector IDs and memory IDs
        self.vector_to_memory_id = {}
        self.memory_id_to_vector = {}
        
        # Track vector count
        self.vector_count = 0
        
        # Metadata storage
        self.metadata = {}
        
        # Category to memory ID mapping for filtering
        self.categories = defaultdict(set)
    
    def add(self, memory_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> int:
        # Ensure vector is the right shape and type
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {vector.shape}")
        
        # Add to index
        vector_id = self.vector_count
        self.index.add_items(vector, vector_id)
        
        # Update mappings
        self.vector_to_memory_id[vector_id] = memory_id
        self.memory_id_to_vector[memory_id] = vector_id
        
        # Store metadata
        if metadata:
            self.metadata[vector_id] = metadata
            
            # Update category indices for filtering
            for category, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    self.categories[f"{category}:{value}"].add(vector_id)
        
        # Increment counter
        self.vector_count += 1
        
        return vector_id
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_categories: Dict[str, Any] = None) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        if self.vector_count == 0:
            return []
        
        # Ensure vector is the right shape and type
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        if query_vector.shape != (self.dim,):
            raise ValueError(f"Query vector must have shape ({self.dim},), got {query_vector.shape}")
        
        # Apply filters if specified
        if filter_categories:
            # Find candidate set that matches all filters
            candidate_set = None
            for category, value in filter_categories.items():
                filter_key = f"{category}:{value}"
                if filter_key in self.categories:
                    if candidate_set is None:
                        candidate_set = set(self.categories[filter_key])
                    else:
                        candidate_set &= self.categories[filter_key]
            
            # If no vectors match the filters, return empty list
            if candidate_set is None or len(candidate_set) == 0:
                return []
            
            # Search all and filter after
            labels, distances = self.index.knn_query(query_vector, k=min(self.vector_count, top_k * 10))
            
            # Filter to only include candidates
            filtered_results = []
            for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
                if label in candidate_set:
                    filtered_results.append((label, distance))
                    if len(filtered_results) >= top_k:
                        break
            
            labels = np.array([[r[0] for r in filtered_results]])
            distances = np.array([[r[1] for r in filtered_results]])
        else:
            # No filtering, search all vectors
            labels, distances = self.index.knn_query(query_vector, k=min(self.vector_count, top_k))
        
        # Convert from vector ID to memory ID and get metadata
        results = []
        for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
            if i >= top_k:
                break
                
            memory_id = self.vector_to_memory_id.get(label)
            if memory_id:
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - distance if self.space == "cosine" else 1 / (1 + distance)
                
                # Get metadata if available
                meta = self.metadata.get(label)
                
                results.append((memory_id, similarity, meta))
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        if memory_id not in self.memory_id_to_vector:
            return False
        
        # Get vector ID
        vector_id = self.memory_id_to_vector[memory_id]
        
        # Remove from mappings
        del self.memory_id_to_vector[memory_id]
        del self.vector_to_memory_id[vector_id]
        
        # Remove from category indices
        if vector_id in self.metadata:
            for category, value in self.metadata[vector_id].items():
                filter_key = f"{category}:{value}"
                if filter_key in self.categories and vector_id in self.categories[filter_key]:
                    self.categories[filter_key].remove(vector_id)
            
            # Remove metadata
            del self.metadata[vector_id]
        
        return True
    
    def get_by_id(self, memory_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        if memory_id not in self.memory_id_to_vector:
            return None
        
        vector_id = self.memory_id_to_vector[memory_id]
        meta = self.metadata.get(vector_id, {})
        
        # HNSW doesn't provide direct access to vectors
        return (None, meta)
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        
        # Save index
        self.index.save_index(os.path.join(path, "vector_index.bin"))
        
        # Save mappings and metadata
        with open(os.path.join(path, "vector_mappings.json"), "w") as f:
            json.dump({
                "vector_to_memory_id": {str(k): v for k, v in self.vector_to_memory_id.items()},
                "memory_id_to_vector": {k: int(v) for k, v in self.memory_id_to_vector.items()},
                "vector_count": self.vector_count,
                "metadata": {str(k): v for k, v in self.metadata.items()},
                "dim": self.dim,
                "max_elements": self.max_elements,
                "space": self.space,
                "ef_search": self.ef_search
            }, f)
        
        # Save categories (convert sets to lists for JSON serialization)
        category_dict = {k: list(v) for k, v in self.categories.items()}
        with open(os.path.join(path, "vector_categories.json"), "w") as f:
            json.dump(category_dict, f)
    
    def load(self, path: str) -> bool:
        try:
            # Load mappings and metadata
            with open(os.path.join(path, "vector_mappings.json"), "r") as f:
                data = json.load(f)
                self.vector_to_memory_id = {int(k): v for k, v in data["vector_to_memory_id"].items()}
                self.memory_id_to_vector = {k: int(v) for k, v in data["memory_id_to_vector"].items()}
                self.vector_count = data["vector_count"]
                self.metadata = {int(k): v for k, v in data["metadata"].items()}
                self.dim = data["dim"]
                self.max_elements = data["max_elements"]
                self.space = data["space"]
                self.ef_search = data["ef_search"]
            
            # Load categories
            with open(os.path.join(path, "vector_categories.json"), "r") as f:
                categories = json.load(f)
                self.categories = {k: set(v) for k, v in categories.items()}
            
            # Recreate index
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(os.path.join(path, "vector_index.bin"), max_elements=self.max_elements)
            self.index.set_ef(self.ef_search)
            
            return True
        except Exception as e:
            print(f"Error loading HNSW vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_count": self.vector_count,
            "dim": self.dim,
            "max_elements": self.max_elements,
            "space": self.space,
            "categories": {k: len(v) for k, v in self.categories.items()},
            "metadata_count": len(self.metadata)
        } 