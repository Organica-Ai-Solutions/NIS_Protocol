"""
Response Cache - NIS Protocol v4.0
Cache LLM responses to avoid redundant API calls and reduce costs.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import logging

logger = logging.getLogger("nis.cache")


@dataclass
class CacheEntry:
    """Cached response entry"""
    query_hash: str
    query: str
    response: str
    provider: str
    model: str
    timestamp: float
    ttl: int  # Time to live in seconds
    hit_count: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0


class ResponseCache:
    """
    LRU cache for LLM responses.
    Saves money by returning cached responses for identical queries.
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        storage_path: str = "data/cache",
        persist: bool = True
    ):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.storage_path = Path(storage_path)
        self.persist = persist
        
        if persist:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "tokens_saved": 0,
            "cost_saved": 0.0,
            "evictions": 0
        }
        
        # Load persisted cache
        if persist:
            self._load()
        
        logger.info(f"🗄️ Response cache initialized: {len(self.cache)} entries")
    
    def _hash_query(self, query: str, provider: str = "", model: str = "", context: str = "") -> str:
        """Generate hash for cache key"""
        key = f"{query}|{provider}|{model}|{context}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def get(
        self,
        query: str,
        provider: str = "",
        model: str = "",
        context: str = ""
    ) -> Optional[str]:
        """
        Get cached response if available and not expired.
        
        Returns:
            Cached response string or None if not found/expired
        """
        query_hash = self._hash_query(query, provider, model, context)
        
        with self.lock:
            entry = self.cache.get(query_hash)
            
            if entry is None:
                self.stats["misses"] += 1
                return None
            
            # Check expiration
            if time.time() > entry.timestamp + entry.ttl:
                del self.cache[query_hash]
                self.stats["misses"] += 1
                return None
            
            # Cache hit
            entry.hit_count += 1
            self.stats["hits"] += 1
            self.stats["tokens_saved"] += entry.tokens_saved
            self.stats["cost_saved"] += entry.cost_saved
            
            logger.debug(f"Cache hit: {query[:50]}...")
            return entry.response
    
    def set(
        self,
        query: str,
        response: str,
        provider: str = "",
        model: str = "",
        context: str = "",
        ttl: Optional[int] = None,
        tokens_used: int = 0,
        cost: float = 0.0
    ):
        """
        Cache a response.
        
        Args:
            query: The original query
            response: The LLM response
            provider: LLM provider name
            model: Model name
            context: Additional context (affects cache key)
            ttl: Time to live in seconds
            tokens_used: Tokens this response used (for savings tracking)
            cost: Cost of this response (for savings tracking)
        """
        query_hash = self._hash_query(query, provider, model, context)
        
        entry = CacheEntry(
            query_hash=query_hash,
            query=query[:500],  # Truncate for storage
            response=response,
            provider=provider,
            model=model,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
            hit_count=0,
            tokens_saved=tokens_used,
            cost_saved=cost
        )
        
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_entries:
                self._evict_lru()
            
            self.cache[query_hash] = entry
        
        # Periodic persist
        if self.persist and len(self.cache) % 50 == 0:
            self._save()
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Sort by last access (timestamp + hit recency approximation)
        entries = list(self.cache.items())
        entries.sort(key=lambda x: x[1].timestamp + (x[1].hit_count * 60))
        
        # Remove bottom 10%
        to_remove = max(1, len(entries) // 10)
        for key, _ in entries[:to_remove]:
            del self.cache[key]
            self.stats["evictions"] += 1
    
    def invalidate(self, query: str, provider: str = "", model: str = "", context: str = ""):
        """Invalidate a specific cache entry"""
        query_hash = self._hash_query(query, provider, model, context)
        with self.lock:
            if query_hash in self.cache:
                del self.cache[query_hash]
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
        if self.persist:
            cache_file = self.storage_path / "response_cache.json"
            if cache_file.exists():
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total, 1) * 100
        
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate_percent": round(hit_rate, 1),
            "tokens_saved": self.stats["tokens_saved"],
            "cost_saved_usd": round(self.stats["cost_saved"], 4),
            "evictions": self.stats["evictions"]
        }
    
    def _load(self):
        """Load cache from disk"""
        cache_file = self.storage_path / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = CacheEntry(**entry_data)
                        # Skip expired entries
                        if time.time() <= entry.timestamp + entry.ttl:
                            self.cache[entry.query_hash] = entry
                    self.stats = data.get("stats", self.stats)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save(self):
        """Save cache to disk"""
        if not self.persist:
            return
        
        cache_file = self.storage_path / "response_cache.json"
        try:
            data = {
                "entries": [asdict(e) for e in self.cache.values()],
                "stats": self.stats
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def save(self):
        """Force save cache"""
        self._save()


# Global instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create the global response cache"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
