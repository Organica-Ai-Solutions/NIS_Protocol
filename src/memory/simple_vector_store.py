"""
Simple Vector Store - Alternative to hnswlib for GLIBCXX compatibility

This module provides a lightweight vector database functionality without hnswlib dependency,
using numpy and sklearn for vector similarity search.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class SimpleVectorStore:
    """
    Simple vector database implementation without hnswlib dependency.
    
    This class provides vector storage and similarity search using pure numpy
    and sklearn, avoiding GLIBCXX compatibility issues.
    """
    
    def __init__(self, dim: int = 768, max_elements: int = 100000, space: str = "cosine"):
        """
        Initialize the simple vector store.
        
        Args:
            dim: Dimensionality of the embedding vectors
            max_elements: Maximum number of vectors to store
            space: Distance metric to use (cosine, l2)
        """
        self.dim = dim
        self.max_elements = max_elements
        self.space = space
        
        # Storage for vectors
        self.vectors = np.empty((max_elements, dim), dtype=np.float32)
        self.vector_count = 0
        
        # Track mappings between vector IDs and memory IDs
        self.vector_to_memory_id = {}
        self.memory_id_to_vector = {}
        
        # Metadata storage
        self.metadata = {}
        
        # Category to memory ID mapping for filtering
        self.categories = defaultdict(set)
        
        # Active vector indices (for handling deletions)
        self.active_indices = set()
    
    def add(self, memory_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None) -> int:
        """
        Add a vector embedding to the store.
        
        Args:
            memory_id: Unique identifier for the memory
            vector: The embedding vector (numpy array)
            metadata: Optional metadata to associate with the vector
            
        Returns:
            The ID of the added vector
        """
        if self.vector_count >= self.max_elements:
            raise RuntimeError(f"Maximum number of vectors ({self.max_elements}) reached")
        
        # Ensure vector is the right shape and type
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {vector.shape}")
        
        # Add to storage
        vector_id = self.vector_count
        self.vectors[vector_id] = vector
        self.active_indices.add(vector_id)
        
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
    
    def search(self, 
               query_vector: np.ndarray, 
               top_k: int = 10, 
               filter_categories: Dict[str, Any] = None) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """
        Search for vectors similar to the query vector.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filter_categories: Optional category filters (e.g., {"tag": "important"})
            
        Returns:
            List of (memory_id, similarity_score, metadata) tuples
        """
        if len(self.active_indices) == 0:
            return []
        
        # Ensure vector is the right shape and type
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        if query_vector.shape != (self.dim,):
            raise ValueError(f"Query vector must have shape ({self.dim},), got {query_vector.shape}")
        
        # Get active vectors
        active_indices = list(self.active_indices)
        
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
            
            # Filter active indices to only include candidates
            active_indices = [idx for idx in active_indices if idx in candidate_set]
        
        if not active_indices:
            return []
        
        # Get vectors for similarity computation
        active_vectors = self.vectors[active_indices]
        query_vector = query_vector.reshape(1, -1)
        
        # Compute similarities
        if self.space == "cosine":
            similarities = cosine_similarity(query_vector, active_vectors)[0]
        elif self.space == "l2":
            distances = euclidean_distances(query_vector, active_vectors)[0]
            # Convert distances to similarities (higher is better)
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unsupported space: {self.space}")
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Convert to results format
        results = []
        for i, similarity_idx in enumerate(top_indices):
            vector_id = active_indices[similarity_idx]
            memory_id = self.vector_to_memory_id.get(vector_id)
            
            if memory_id:
                similarity = float(similarities[similarity_idx])
                meta = self.metadata.get(vector_id)
                results.append((memory_id, similarity, meta))
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if memory_id not in self.memory_id_to_vector:
            return False
        
        # Get vector ID
        vector_id = self.memory_id_to_vector[memory_id]
        
        # Remove from active indices
        self.active_indices.discard(vector_id)
        
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
        """
        Get a vector by memory ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        if memory_id not in self.memory_id_to_vector:
            return None
        
        vector_id = self.memory_id_to_vector[memory_id]
        if vector_id not in self.active_indices:
            return None
        
        vector = self.vectors[vector_id]
        meta = self.metadata.get(vector_id, {})
        
        return (vector, meta)
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save vectors (only active ones)
        active_vectors = {}
        for idx in self.active_indices:
            active_vectors[str(idx)] = self.vectors[idx].tolist()
        
        with open(os.path.join(path, "simple_vectors.json"), "w") as f:
            json.dump(active_vectors, f)
        
        # Save mappings and metadata
        with open(os.path.join(path, "simple_mappings.json"), "w") as f:
            json.dump({
                "vector_to_memory_id": {str(k): v for k, v in self.vector_to_memory_id.items()},
                "memory_id_to_vector": {k: int(v) for k, v in self.memory_id_to_vector.items()},
                "vector_count": self.vector_count,
                "metadata": {str(k): v for k, v in self.metadata.items()},
                "dim": self.dim,
                "max_elements": self.max_elements,
                "space": self.space,
                "active_indices": list(self.active_indices)
            }, f)
        
        # Save categories (convert sets to lists for JSON serialization)
        category_dict = {k: list(v) for k, v in self.categories.items()}
        with open(os.path.join(path, "simple_categories.json"), "w") as f:
            json.dump(category_dict, f)
    
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load mappings and metadata
            with open(os.path.join(path, "simple_mappings.json"), "r") as f:
                data = json.load(f)
                self.vector_to_memory_id = {int(k): v for k, v in data["vector_to_memory_id"].items()}
                self.memory_id_to_vector = {k: int(v) for k, v in data["memory_id_to_vector"].items()}
                self.vector_count = data["vector_count"]
                self.metadata = {int(k): v for k, v in data["metadata"].items()}
                self.dim = data["dim"]
                self.max_elements = data["max_elements"]
                self.space = data["space"]
                self.active_indices = set(data["active_indices"])
            
            # Load categories
            with open(os.path.join(path, "simple_categories.json"), "r") as f:
                categories = json.load(f)
                self.categories = {k: set(v) for k, v in categories.items()}
            
            # Load vectors
            with open(os.path.join(path, "simple_vectors.json"), "r") as f:
                vectors_data = json.load(f)
                self.vectors = np.empty((self.max_elements, self.dim), dtype=np.float32)
                for idx_str, vector_list in vectors_data.items():
                    idx = int(idx_str)
                    self.vectors[idx] = np.array(vector_list, dtype=np.float32)
            
            return True
        except Exception as e:
            print(f"Error loading simple vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "vector_count": len(self.active_indices),
            "total_capacity": self.vector_count,
            "dim": self.dim,
            "max_elements": self.max_elements,
            "space": self.space,
            "categories": {k: len(v) for k, v in self.categories.items()},
            "metadata_count": len(self.metadata)
        }


# Compatibility alias - can be used as a drop-in replacement
VectorStore = SimpleVectorStore
