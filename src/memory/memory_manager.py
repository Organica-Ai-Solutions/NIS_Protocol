"""
NIS Protocol Memory Management System

This module provides the memory management system for the NIS Protocol,
enabling agents to store and retrieve information.
"""

import time
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """Abstract base class for memory storage backends.
    
    This class defines the interface that all storage backends must implement.
    """
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with an optional time-to-live.
        
        Args:
            key: The unique key for the value
            value: The value to store
            ttl: Optional time-to-live in seconds
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.
        
        Args:
            key: The unique key for the value
            
        Returns:
            The stored value, or None if not found or expired
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value by key.
        
        Args:
            key: The unique key for the value
            
        Returns:
            True if the value was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def search(self, pattern: Dict[str, Any]) -> List[Any]:
        """Search for values matching a pattern.
        
        Args:
            pattern: Dictionary with search patterns
            
        Returns:
            List of matching values
        """
        pass


class InMemoryStorage(StorageBackend):
    """Simple in-memory storage backend.
    
    This backend is suitable for development and testing, but not for
    production use as it does not persist data across restarts.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with an optional time-to-live.
        
        Args:
            key: The unique key for the value
            value: The value to store
            ttl: Optional time-to-live in seconds
        """
        self.data[key] = value
        
        if ttl is not None:
            self.expiry[key] = time.time() + ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.
        
        Args:
            key: The unique key for the value
            
        Returns:
            The stored value, or None if not found or expired
        """
        # Check if the key exists
        if key not in self.data:
            return None
        
        # Check if the key has expired
        if key in self.expiry and time.time() > self.expiry[key]:
            # Delete the expired key
            del self.data[key]
            del self.expiry[key]
            return None
        
        return self.data[key]
    
    def delete(self, key: str) -> bool:
        """Delete a value by key.
        
        Args:
            key: The unique key for the value
            
        Returns:
            True if the value was deleted, False if not found
        """
        if key in self.data:
            del self.data[key]
            
            if key in self.expiry:
                del self.expiry[key]
                
            return True
            
        return False
    
    def search(self, pattern: Dict[str, Any]) -> List[Any]:
        """Search for values matching a pattern.
        
        Args:
            pattern: Dictionary with search patterns
            
        Returns:
            List of matching values
        """
        results = []
        
        for key, value in self.data.items():
            # Skip expired keys
            if key in self.expiry and time.time() > self.expiry[key]:
                continue
                
            # Check if the value matches the pattern
            if isinstance(value, dict) and self._matches_pattern(value, pattern):
                results.append(value)
        
        return results
    
    def _matches_pattern(self, value: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if a value matches a pattern.
        
        Args:
            value: The value to check
            pattern: The pattern to match
            
        Returns:
            True if the value matches the pattern, False otherwise
        """
        for pattern_key, pattern_value in pattern.items():
            # Skip if the pattern key doesn't exist in the value
            if pattern_key not in value:
                return False
                
            # If the pattern value is a dict, recursively check it
            if isinstance(pattern_value, dict) and isinstance(value[pattern_key], dict):
                if not self._matches_pattern(value[pattern_key], pattern_value):
                    return False
            # Otherwise, check for equality
            elif value[pattern_key] != pattern_value:
                return False
        
        return True


class MemoryManager:
    """Memory management for NIS Protocol.
    
    The Memory Manager provides a unified interface for storing and retrieving
    information, regardless of the underlying storage backend.
    """
    
    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        """Initialize the memory manager.
        
        Args:
            storage_backend: Optional storage backend (defaults to InMemoryStorage)
        """
        self.storage = storage_backend or InMemoryStorage()
        self.access_log: List[Tuple[float, str, str]] = []
    
    def store(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Store data in memory.
        
        Args:
            key: The unique key for the data
            data: The data to store
            ttl: Optional time-to-live in seconds
        """
        # Add metadata to the stored data
        stored_data = data.copy()
        stored_data["__metadata__"] = {
            "stored_at": time.time(),
            "ttl": ttl
        }
        
        # Store the data
        self.storage.set(key, stored_data, ttl)
        
        # Log the access
        self._log_access("store", key)
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from memory.
        
        Args:
            key: The unique key for the data
            
        Returns:
            The stored data, or None if not found or expired
        """
        # Retrieve the data
        data = self.storage.get(key)
        
        # Log the access
        self._log_access("retrieve", key)
        
        # Return the data (excluding metadata)
        if data is not None and "__metadata__" in data:
            result = data.copy()
            del result["__metadata__"]
            return result
            
        return data
    
    def forget(self, key: str) -> bool:
        """Remove data from memory.
        
        Args:
            key: The unique key for the data
            
        Returns:
            True if the data was removed, False if not found
        """
        # Delete the data
        result = self.storage.delete(key)
        
        # Log the access
        self._log_access("forget", key)
        
        return result
    
    def search(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search memory for matching patterns.
        
        Args:
            pattern: Dictionary with search patterns
            
        Returns:
            List of matching memory items
        """
        # Search for matching data
        results = self.storage.search(pattern)
        
        # Log the access
        self._log_access("search", json.dumps(pattern))
        
        # Return the data (excluding metadata)
        cleaned_results = []
        for item in results:
            if isinstance(item, dict) and "__metadata__" in item:
                result = item.copy()
                del result["__metadata__"]
                cleaned_results.append(result)
            else:
                cleaned_results.append(item)
                
        return cleaned_results
    
    def get_access_log(self, operation: Optional[str] = None) -> List[Tuple[float, str, str]]:
        """Get the memory access log.
        
        Args:
            operation: Optional operation to filter by ("store", "retrieve", "forget", or "search")
            
        Returns:
            List of (timestamp, operation, key) tuples
        """
        if operation is None:
            return self.access_log
            
        return [entry for entry in self.access_log if entry[1] == operation]
    
    def _log_access(self, operation: str, key: str) -> None:
        """Log a memory access.
        
        Args:
            operation: The operation performed ("store", "retrieve", "forget", or "search")
            key: The key that was accessed
        """
        self.access_log.append((time.time(), operation, key))
        
        # Limit log size to prevent memory issues
        max_log_size = 1000
        if len(self.access_log) > max_log_size:
            self.access_log = self.access_log[-max_log_size:] 