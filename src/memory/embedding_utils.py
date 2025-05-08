"""
NIS Protocol Embedding Utilities

This module provides utilities for generating and working with embeddings,
which are vector representations of textual data used for semantic memory search.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
import hashlib
import json
import time
import pickle

class EmbeddingProvider:
    """
    Base class for embedding providers.
    
    This abstract class defines the interface for embedding providers
    that convert textual data into vector representations.
    """
    
    def __init__(self):
        """Initialize the embedding provider."""
        pass
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("Subclasses must implement get_embedding()")
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement get_batch_embeddings()")
    
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            The dimensionality of the embeddings
        """
        raise NotImplementedError("Subclasses must implement get_dimensions()")


class CachedEmbeddingProvider:
    """
    Wrapper class that adds caching to any embedding provider.
    
    This class wraps an embedding provider and caches embeddings locally
    to avoid redundant computation and API calls.
    """
    
    def __init__(self, provider: EmbeddingProvider, cache_dir: str = ".embeddings_cache"):
        """
        Initialize the cached embedding provider.
        
        Args:
            provider: The embedding provider to wrap
            cache_dir: Directory to store cached embeddings
        """
        self.provider = provider
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for the given text, using cache if available.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        # Generate a deterministic hash of the text
        hash_id = self._hash_text(text)
        cache_path = os.path.join(self.cache_dir, f"{hash_id}.npy")
        
        # Check if embedding is already cached
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
        
        # Generate new embedding
        embedding = self.provider.get_embedding(text)
        
        # Cache the embedding
        try:
            np.save(cache_path, embedding)
        except Exception as e:
            print(f"Error caching embedding: {e}")
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of texts, using cache when available.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        # Check which embeddings are already cached
        cached_embeddings = {}
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            hash_id = self._hash_text(text)
            cache_path = os.path.join(self.cache_dir, f"{hash_id}.npy")
            
            if os.path.exists(cache_path):
                try:
                    cached_embeddings[i] = np.load(cache_path)
                except Exception:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Get embeddings for texts not in cache
        if texts_to_embed:
            new_embeddings = self.provider.get_batch_embeddings(texts_to_embed)
            
            # Cache new embeddings
            for i, (idx, text) in enumerate(zip(indices_to_embed, texts_to_embed)):
                hash_id = self._hash_text(text)
                cache_path = os.path.join(self.cache_dir, f"{hash_id}.npy")
                
                try:
                    np.save(cache_path, new_embeddings[i])
                    cached_embeddings[idx] = new_embeddings[i]
                except Exception as e:
                    print(f"Error caching embedding: {e}")
        
        # Combine cached and new embeddings in the original order
        embeddings = np.zeros((len(texts), self.provider.get_dimensions()), dtype=np.float32)
        for i, embedding in cached_embeddings.items():
            embeddings[i] = embedding
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            The dimensionality of the embeddings
        """
        return self.provider.get_dimensions()
    
    def _hash_text(self, text: str) -> str:
        """
        Generate a deterministic hash of the text.
        
        Args:
            text: The text to hash
            
        Returns:
            Hex digest of the hash
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    A simple embedding provider that uses a deterministic algorithm.
    
    This provider is intended for testing and development purposes
    when more sophisticated models are unavailable.
    """
    
    def __init__(self, dimensions: int = 768):
        """
        Initialize the simple embedding provider.
        
        Args:
            dimensions: Dimensionality of the embeddings
        """
        super().__init__()
        self.dimensions = dimensions
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get a deterministic embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        # Use the hash of the text to seed a random number generator
        np.random.seed(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32))
        
        # Generate a pseudo-random vector
        embedding = np.random.normal(0, 1, self.dimensions).astype(np.float32)
        
        # Normalize to unit length
        return embedding / np.linalg.norm(embedding)
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        embeddings = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        
        for i, text in enumerate(texts):
            embeddings[i] = self.get_embedding(text)
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            The dimensionality of the embeddings
        """
        return self.dimensions


try:
    from sentence_transformers import SentenceTransformer
    
    class SentenceTransformerProvider(EmbeddingProvider):
        """
        Embedding provider that uses SentenceTransformer models.
        
        This provider generates high-quality embeddings using
        transformer-based models.
        """
        
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            """
            Initialize the SentenceTransformer provider.
            
            Args:
                model_name: Name of the SentenceTransformer model to use
            """
            super().__init__()
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        
        def get_embedding(self, text: str) -> np.ndarray:
            """
            Get an embedding for the given text.
            
            Args:
                text: The text to embed
                
            Returns:
                Embedding vector
            """
            return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
            """
            Get embeddings for a batch of texts.
            
            Args:
                texts: List of texts to embed
                
            Returns:
                Array of embedding vectors
            """
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        def get_dimensions(self) -> int:
            """
            Get the dimensionality of the embeddings.
            
            Returns:
                The dimensionality of the embeddings
            """
            return self.model.get_sentence_embedding_dimension()
            
except ImportError:
    # SentenceTransformer is not available, so we don't define the provider class
    pass


# Factory function to get the best available embedding provider
def get_embedding_provider(provider_type: str = "auto", 
                           cache: bool = True, 
                           cache_dir: str = ".embeddings_cache",
                           dimensions: int = 768) -> EmbeddingProvider:
    """
    Get the best available embedding provider.
    
    Args:
        provider_type: Type of embedding provider to use ('auto', 'sentence_transformer', 'simple')
        cache: Whether to cache embeddings
        cache_dir: Directory to store cached embeddings
        dimensions: Dimensionality for the simple provider
        
    Returns:
        An embedding provider instance
    """
    # Determine which provider to use
    if provider_type == "auto":
        try:
            from sentence_transformers import SentenceTransformer
            provider = SentenceTransformerProvider()
        except ImportError:
            provider = SimpleEmbeddingProvider(dimensions=dimensions)
    elif provider_type == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
            provider = SentenceTransformerProvider()
        except ImportError:
            raise ImportError("SentenceTransformer is not available. Please install it with 'pip install sentence-transformers'")
    elif provider_type == "simple":
        provider = SimpleEmbeddingProvider(dimensions=dimensions)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    # Wrap in a cache if requested
    if cache:
        return CachedEmbeddingProvider(provider, cache_dir=cache_dir)
    else:
        return provider 