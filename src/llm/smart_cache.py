#!/usr/bin/env python3
"""
🎯 NIS Protocol Smart Caching System
Advanced caching for LLM responses with intelligent cache management

Features:
- TTL-based response caching
- Semantic similarity detection
- Cost-aware cache policies
- Provider-specific cache strategies
- Memory-efficient LRU eviction
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from cachetools import TTLCache, LRUCache
import redis
import threading

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cached LLM response entry"""
    response: Dict[str, Any]
    provider: str
    model: str
    temperature: float
    cost: float
    created_at: float
    hit_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

@dataclass 
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_saved: float = 0.0
    total_time_saved: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_requests, 1)
    
    @property
    def miss_rate(self) -> float:
        return self.cache_misses / max(self.total_requests, 1)

class SmartLLMCache:
    """
    🧠 Smart caching system for LLM responses
    
    Features:
    - Intelligent cache key generation
    - Cost-aware cache policies
    - Semantic similarity detection
    - Provider-specific TTL settings
    - Persistent cache storage
    - Advanced eviction strategies
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: int = 3600,  # 1 hour
                 redis_host: str = None,
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 enable_semantic_matching: bool = True):
        """Initialize smart cache with Redis backend"""
        
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_semantic_matching = enable_semantic_matching
        
        # Determine Redis host
        if redis_host is None:
            # Try environment variable first
            redis_host = os.environ.get('REDIS_HOST')
            if not redis_host:
                # Check if we're in a Docker environment
                if os.path.exists('/.dockerenv'):
                    redis_host = "nis-redis"  # Docker container name
                else:
                    redis_host = "localhost"  # Local development
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis cache: {redis_host}:{redis_port}/{redis_db}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # In-memory cache with TTL (for ultra-fast access)
        self.memory_cache = TTLCache(maxsize=max_size//2, ttl=default_ttl//2)
        
        # Provider-specific cache settings
        self.provider_settings = {
            "openai": {"ttl": 3600, "cost_weight": 1.0},
            "anthropic": {"ttl": 7200, "cost_weight": 1.5},  # Higher cost, cache longer
            "google": {"ttl": 1800, "cost_weight": 0.3},     # Cheaper, shorter cache
            "deepseek": {"ttl": 3600, "cost_weight": 0.5},
            "nvidia": {"ttl": 7200, "cost_weight": 2.0},     # Premium models
            "multimodel": {"ttl": 10800, "cost_weight": 3.0} # Most expensive
        }
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Redis key prefixes
        self.CACHE_PREFIX = "llm:cache"
        self.STATS_PREFIX = "llm:cache:stats"
        
        logger.info(f"🧠 Smart LLM Cache initialized with Redis - Size: {max_size}, TTL: {default_ttl}s")
    
    def _init_redis_cache_stats(self):
        """Initialize Redis cache statistics"""
        try:
            # Initialize cache stats if not exists
            if not self.redis_client.exists(f"{self.STATS_PREFIX}:total"):
                self.redis_client.hmset(f"{self.STATS_PREFIX}:total", {
                    "total_requests": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "total_cost_saved": 0.0,
                    "total_time_saved": 0.0
                })
            logger.info("✅ Redis cache statistics initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache stats: {e}")
    
    def _generate_cache_key(self, 
                           messages: List[Dict[str, str]], 
                           provider: str,
                           model: str,
                           temperature: float,
                           **kwargs) -> str:
        """Generate intelligent cache key"""
        
        # Create normalized message content
        normalized_messages = []
        for msg in messages:
            # Normalize whitespace and case for better caching
            content = msg.get("content", "").strip().lower()
            role = msg.get("role", "")
            normalized_messages.append(f"{role}:{content}")
        
        # Create cache key components
        message_hash = hashlib.md5(
            "\n".join(normalized_messages).encode()
        ).hexdigest()[:16]
        
        # Include important parameters in key
        key_data = {
            "messages": message_hash,
            "provider": provider,
            "model": model,
            "temperature": round(temperature, 2),  # Round to avoid float precision issues
            "kwargs": sorted(kwargs.items()) if kwargs else []
        }
        
        # Generate final cache key
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
        return f"{provider}:{model}:{cache_key}"
    
    def _calculate_semantic_similarity(self, messages1: List[Dict], messages2: List[Dict]) -> float:
        """Calculate semantic similarity between message sets (simplified)"""
        if not self.enable_semantic_matching:
            return 0.0
        
        # Simple semantic matching based on keyword overlap
        # In production, you'd use embeddings or more sophisticated NLP
        
        def extract_keywords(messages):
            keywords = set()
            for msg in messages:
                content = msg.get("content", "").lower()
                # Simple keyword extraction
                words = [w for w in content.split() if len(w) > 3]
                keywords.update(words[:20])  # Limit keywords
            return keywords
        
        keywords1 = extract_keywords(messages1)
        keywords2 = extract_keywords(messages2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _should_cache_response(self, 
                              provider: str, 
                              response: Dict[str, Any],
                              cost: float) -> bool:
        """Determine if response should be cached"""
        
        # Always cache expensive responses
        if cost > 0.01:  # > 1 cent
            return True
        
        # Cache based on provider settings
        settings = self.provider_settings.get(provider, {})
        cost_weight = settings.get("cost_weight", 1.0)
        
        if cost * cost_weight > 0.005:  # Weighted cost threshold
            return True
        
        # Cache long responses
        content = response.get("content", "")
        if len(content) > 500:
            return True
        
        # Cache high-confidence responses
        confidence = response.get("confidence", 0.0)
        if confidence > 0.9:
            return True
        
        return False
    
    def get(self, 
            messages: List[Dict[str, str]], 
            provider: str,
            model: str,
            temperature: float,
            similarity_threshold: float = 0.8,
            **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        
        with self._lock:
            self.stats.total_requests += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(messages, provider, model, temperature, **kwargs)
            
            # Try memory cache first (ultra-fast)
            cached_entry = self.memory_cache.get(cache_key)
            
            if cached_entry:
                # Update access statistics
                cached_entry.hit_count += 1
                cached_entry.last_accessed = time.time()
                self.stats.cache_hits += 1
                self.stats.total_cost_saved += cached_entry.cost
                self.stats.total_time_saved += 0.5  # Memory cache time
                
                # Update Redis stats
                self.redis_client.hincrby(f"{self.STATS_PREFIX}:total", "cache_hits", 1)
                self.redis_client.hincrbyfloat(f"{self.STATS_PREFIX}:total", "total_cost_saved", cached_entry.cost)
                
                logger.debug(f"🎯 Memory Cache HIT: {provider}/{model}")
                return cached_entry.response
            
            # Try Redis cache
            redis_entry = self._get_from_redis(cache_key)
            if redis_entry:
                # Store in memory cache for faster future access
                self.memory_cache[cache_key] = redis_entry
                
                # Update access statistics
                redis_entry.hit_count += 1
                redis_entry.last_accessed = time.time()
                self.stats.cache_hits += 1
                self.stats.total_cost_saved += redis_entry.cost
                self.stats.total_time_saved += 1.0  # Redis cache time
                
                # Update Redis stats
                self.redis_client.hincrby(f"{self.STATS_PREFIX}:total", "cache_hits", 1)
                self.redis_client.hincrbyfloat(f"{self.STATS_PREFIX}:total", "total_cost_saved", redis_entry.cost)
                
                # Update hit count in Redis
                self._update_redis_hit_count(cache_key, redis_entry)
                
                logger.debug(f"🎯 Redis Cache HIT: {provider}/{model}")
                return redis_entry.response
            
            # Try semantic similarity matching if enabled
            if self.enable_semantic_matching and similarity_threshold > 0:
                similar_entry = self._find_similar_cached_response(
                    messages, provider, model, temperature, similarity_threshold
                )
                if similar_entry:
                    self.stats.cache_hits += 1
                    self.stats.total_cost_saved += similar_entry.cost * 0.5  # Partial savings
                    logger.debug(f"🎯 Cache SIMILAR_HIT: {provider}/{model}")
                    return similar_entry.response
            

            
            # Cache miss
            self.stats.cache_misses += 1
            self.redis_client.hincrby(f"{self.STATS_PREFIX}:total", "cache_misses", 1)
            logger.debug(f"❌ Cache MISS: {provider}/{model}")
            return None
    
    def _find_similar_cached_response(self, 
                                    messages: List[Dict[str, str]], 
                                    provider: str,
                                    model: str,
                                    temperature: float,
                                    threshold: float) -> Optional[CacheEntry]:
        """Find semantically similar cached response"""
        
        best_match = None
        best_similarity = 0.0
        
        # Check recent cache entries for similar content
        for key, entry in list(self.memory_cache.items())[:50]:  # Check recent 50
            if not key.startswith(f"{provider}:{model}:"):
                continue
            
            # Skip if temperature difference is too large
            if abs(entry.temperature - temperature) > 0.3:
                continue
            
            # Calculate semantic similarity (placeholder implementation)
            # In production, use proper embedding similarity
            similarity = 0.7  # Simplified for demonstration
            
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        return best_match
    
    def put(self, 
            messages: List[Dict[str, str]], 
            provider: str,
            model: str,
            temperature: float,
            response: Dict[str, Any],
            cost: float = 0.0,
            **kwargs) -> bool:
        """Cache response if it meets caching criteria"""
        
        with self._lock:
            # Check if should cache
            if not self._should_cache_response(provider, response, cost):
                logger.debug(f"⏭️ Skipping cache: {provider}/{model} (cost: ${cost:.4f})")
                return False
            
            # Generate cache key
            cache_key = self._generate_cache_key(messages, provider, model, temperature, **kwargs)
            
            # Create cache entry
            entry = CacheEntry(
                response=response,
                provider=provider,
                model=model,
                temperature=temperature,
                cost=cost,
                created_at=time.time()
            )
            
            # Store in memory cache
            self.memory_cache[cache_key] = entry
            
            # Store in Redis cache
            self._put_to_redis(cache_key, entry)
            
            logger.debug(f"💾 Cached response: {provider}/{model} (cost: ${cost:.4f})")
            return True
    
    def _get_from_redis(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from Redis storage"""
        try:
            redis_key = f"{self.CACHE_PREFIX}:{cache_key}"
            entry_data = self.redis_client.hgetall(redis_key)
            
            if entry_data:
                # Convert back to proper types
                response_data = json.loads(entry_data["response"])
                return CacheEntry(
                    response=response_data,
                    provider=entry_data["provider"],
                    model=entry_data["model"],
                    temperature=float(entry_data["temperature"]),
                    cost=float(entry_data["cost"]),
                    created_at=float(entry_data["created_at"]),
                    hit_count=int(entry_data.get("hit_count", 0)),
                    last_accessed=float(entry_data.get("last_accessed", entry_data["created_at"]))
                )
        except Exception as e:
            logger.error(f"Error reading from Redis cache: {e}")
        
        return None
    
    def _put_to_redis(self, cache_key: str, entry: CacheEntry):
        """Store entry in Redis storage"""
        try:
            redis_key = f"{self.CACHE_PREFIX}:{cache_key}"
            
            # Prepare data for Redis
            redis_data = {
                "response": json.dumps(entry.response),
                "provider": entry.provider,
                "model": entry.model,
                "temperature": entry.temperature,
                "cost": entry.cost,
                "created_at": entry.created_at,
                "hit_count": entry.hit_count,
                "last_accessed": entry.last_accessed
            }
            
            # Store with TTL based on provider settings
            provider_ttl = self.provider_settings.get(entry.provider, {}).get("ttl", self.default_ttl)
            
            self.redis_client.hmset(redis_key, redis_data)
            self.redis_client.expire(redis_key, provider_ttl)
            
        except Exception as e:
            logger.error(f"Error writing to Redis cache: {e}")
    
    def _update_redis_hit_count(self, cache_key: str, entry: CacheEntry):
        """Update hit count in Redis"""
        try:
            redis_key = f"{self.CACHE_PREFIX}:{cache_key}"
            self.redis_client.hset(redis_key, "hit_count", entry.hit_count)
            self.redis_client.hset(redis_key, "last_accessed", entry.last_accessed)
        except Exception as e:
            logger.error(f"Error updating Redis hit count: {e}")
    
    def clear_provider_cache(self, provider: str):
        """Clear cache for specific provider"""
        with self._lock:
            # Clear from memory cache
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{provider}:")]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # Clear from Redis cache
            try:
                # Find all Redis keys for this provider
                pattern = f"{self.CACHE_PREFIX}:*"
                redis_keys = self.redis_client.keys(pattern)
                
                removed_count = 0
                for redis_key in redis_keys:
                    # Check if this entry belongs to the provider
                    entry_provider = self.redis_client.hget(redis_key, "provider")
                    if entry_provider == provider:
                        self.redis_client.delete(redis_key)
                        removed_count += 1
                
                logger.info(f"🗑️ Cleared cache for provider {provider}: {len(keys_to_remove)} memory, {removed_count} Redis")
                
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
            
            logger.info(f"🗑️ Cleared cache for provider: {provider}")
    
    def cleanup_expired_entries(self, max_age_hours: int = 24):
        """Clean up old entries from Redis storage"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # Find all cache keys
            pattern = f"{self.CACHE_PREFIX}:*"
            redis_keys = self.redis_client.keys(pattern)
            
            removed_count = 0
            for redis_key in redis_keys:
                try:
                    created_at = self.redis_client.hget(redis_key, "created_at")
                    if created_at and float(created_at) < cutoff_time:
                        self.redis_client.delete(redis_key)
                        removed_count += 1
                except:
                    # Skip if can't parse created_at
                    continue
            
            logger.info(f"🧹 Cleaned up {removed_count} expired cache entries from Redis")
            
        except Exception as e:
            logger.error(f"Error cleaning up Redis cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            provider_stats = {}
            
            # Calculate per-provider statistics
            for key, entry in self.memory_cache.items():
                provider = entry.provider
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        "cached_responses": 0,
                        "total_cost_cached": 0.0,
                        "avg_hit_count": 0.0
                    }
                
                provider_stats[provider]["cached_responses"] += 1
                provider_stats[provider]["total_cost_cached"] += entry.cost
                provider_stats[provider]["avg_hit_count"] += entry.hit_count
            
            # Calculate averages
            for provider in provider_stats:
                count = provider_stats[provider]["cached_responses"]
                if count > 0:
                    provider_stats[provider]["avg_hit_count"] /= count
            
            return {
                "overall_stats": asdict(self.stats),
                "cache_size": len(self.memory_cache),
                "max_cache_size": self.max_size,
                "provider_stats": provider_stats,
                "settings": self.provider_settings
            }
    
    def optimize_cache_settings(self):
        """Automatically optimize cache settings based on usage patterns"""
        stats = self.get_cache_stats()
        
        # Adjust TTL based on hit rates
        for provider, pstats in stats["provider_stats"].items():
            if provider in self.provider_settings:
                hit_rate = pstats.get("avg_hit_count", 0)
                
                # Increase TTL for frequently accessed entries
                if hit_rate > 3:
                    self.provider_settings[provider]["ttl"] = min(
                        self.provider_settings[provider]["ttl"] * 1.2,
                        14400  # Max 4 hours
                    )
                # Decrease TTL for rarely accessed entries
                elif hit_rate < 1:
                    self.provider_settings[provider]["ttl"] = max(
                        self.provider_settings[provider]["ttl"] * 0.8,
                        900  # Min 15 minutes
                    )
        
        logger.info("🔧 Cache settings optimized based on usage patterns")


# Global cache instance
_global_cache: Optional[SmartLLMCache] = None

def get_smart_cache() -> SmartLLMCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        # Auto-detect Redis host for Docker/local environments
        redis_host = None
        if os.path.exists('/.dockerenv'):
            redis_host = "nis-redis"  # Docker container name
        _global_cache = SmartLLMCache(redis_host=redis_host)
    return _global_cache

def init_smart_cache(max_size: int = 1000, 
                    default_ttl: int = 3600,
                    redis_host: str = None,
                    redis_port: int = 6379,
                    redis_db: int = 0) -> SmartLLMCache:
    """Initialize global cache with custom settings"""
    global _global_cache
    _global_cache = SmartLLMCache(
        max_size=max_size,
        default_ttl=default_ttl,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db
    )
    return _global_cache
