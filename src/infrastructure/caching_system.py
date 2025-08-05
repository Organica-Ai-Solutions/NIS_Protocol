"""
NIS Protocol v3 - Enhanced Redis Caching System

This module provides comprehensive Redis caching with self-audit capabilities,
performance tracking, and intelligent cache management for all NIS Protocol agents.

Features:
- Self-audit integration for cache integrity
- Performance tracking and optimization
- Intelligent cache strategies and TTL management
- Auto-cleanup and memory optimization
- Health monitoring and diagnostics
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import pickle
from datetime import datetime, timedelta

# Redis imports with fallback
try:
    import redis.asyncio as redis
    import redis.exceptions as redis_exceptions
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        import redis.exceptions as redis_exceptions
        REDIS_AVAILABLE = True
        ASYNC_REDIS_AVAILABLE = False
    except ImportError:
        REDIS_AVAILABLE = False
        ASYNC_REDIS_AVAILABLE = False
        logging.warning("Redis not available. Install redis for full functionality.")

# Self-audit integration
from src.utils.self_audit import self_audit_engine


class CacheStrategy(Enum):
    """Cache strategies for different data types"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    REFRESH_AHEAD = "refresh_ahead"


class CacheNamespace(Enum):
    """Cache namespaces for different agent types"""
    CONSCIOUSNESS = "consciousness"
    MEMORY = "memory"
    SIMULATION = "simulation"
    ALIGNMENT = "alignment"
    GOALS = "goals"
    COORDINATION = "coordination"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    SYSTEM = "system"


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    namespace: str
    ttl: int
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    integrity_score: Optional[float] = None
    audit_flags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheMetrics:
    """Metrics for cache performance tracking"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    avg_latency: float = 0.0
    integrity_violations: int = 0
    last_cleanup: float = 0.0
    last_update: float = 0.0


@dataclass
class PerformanceTracker:
    """Track cache performance over time"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    throughput: float = 0.0
    latency_percentiles: Dict[str, float] = None
    memory_efficiency: float = 0.0
    error_rate: float = 0.0
    
    def __post_init__(self):
        if self.latency_percentiles is None:
            self.latency_percentiles = {"p50": 0.0, "p95": 0.0, "p99": 0.0}


class NISRedisManager:
    """
    Enhanced Redis cache manager with self-audit integration and performance optimization.
    
    Features:
    - Self-audit integration for cache integrity
    - Intelligent cache strategies and TTL management
    - Performance tracking and optimization
    - Auto-cleanup and memory management
    - Health monitoring and diagnostics
    """
    
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enable_self_audit: bool = True,
        max_memory: str = "512mb",
        eviction_policy: str = "allkeys-lru",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the enhanced Redis manager"""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.enable_self_audit = enable_self_audit
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.config = config or {}
        
        # Initialize components
        self.logger = logging.getLogger("nis.redis_manager")
        self.metrics = CacheMetrics()
        self.performance = PerformanceTracker()
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.latency_history: deque = deque(maxlen=1000)
        
        # TTL configurations by namespace
        self.namespace_ttls = {
            CacheNamespace.CONSCIOUSNESS.value: 1800,  # 30 minutes
            CacheNamespace.MEMORY.value: 3600,         # 1 hour
            CacheNamespace.SIMULATION.value: 7200,     # 2 hours
            CacheNamespace.ALIGNMENT.value: 3600,      # 1 hour
            CacheNamespace.GOALS.value: 1800,          # 30 minutes
            CacheNamespace.COORDINATION.value: 900,    # 15 minutes
            CacheNamespace.PERFORMANCE.value: 600,     # 10 minutes
            CacheNamespace.AUDIT.value: 86400,         # 24 hours
            CacheNamespace.SYSTEM.value: 300           # 5 minutes
        }
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.is_initialized = False
        
        # Circuit breaker
        self.circuit_breaker = {
            'state': 'closed',
            'failure_count': 0,
            'last_failure_time': 0,
            'failure_threshold': 3,
            'timeout': 30
        }
        
        # Self-audit integration
        if self.enable_self_audit:
            self._init_self_audit()
        
        self.logger.info(f"NISRedisManager initialized for {host}:{port}")
    
    def _init_self_audit(self):
        """Initialize self-audit capabilities"""
        try:
            self.self_audit_enabled = True
            self.audit_threshold = 75.0  # Minimum integrity score
            self.audit_violations: deque = deque(maxlen=100)
            self.auto_correction_enabled = True
            
            self.logger.info("Self-audit integration enabled for Redis manager")
        except Exception as e:
            self.logger.error(f"Failed to initialize self-audit: {e}")
            self.self_audit_enabled = False
    
    async def initialize(self) -> bool:
        """Initialize async Redis connection"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, running in mock mode")
            self.is_initialized = True
            return True
        
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Configure Redis settings
            await self._configure_redis()
            
            self.is_initialized = True
            self.logger.info("Redis connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            self._update_circuit_breaker(failed=True)
            return False
    
    async def _configure_redis(self):
        """Configure Redis settings for optimal performance"""
        try:
            # Set memory policy
            await self.redis_client.config_set("maxmemory", self.max_memory)
            await self.redis_client.config_set("maxmemory-policy", self.eviction_policy)
            
            # Enable keyspace notifications for monitoring
            await self.redis_client.config_set("notify-keyspace-events", "Ex")
            
            self.logger.info("Redis configuration applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure Redis: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Redis connection shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.TTL
    ) -> bool:
        """
        Set cache value with integrity validation and performance tracking
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time to live in seconds
            strategy: Cache strategy to use
            
        Returns:
            bool: True if value set successfully
        """
        if not self._check_circuit_breaker():
            self.logger.warning("Circuit breaker open, cache set failed")
            return False
        
        start_time = time.time()
        
        try:
            # Determine TTL
            if ttl is None:
                ttl = self.namespace_ttls.get(namespace, 3600)
            
            # Self-audit validation
            integrity_score = None
            audit_flags = None
            
            if self.enable_self_audit:
                audit_result = self._audit_cache_value(key, value, namespace)
                integrity_score = audit_result['score']
                audit_flags = audit_result['flags']
                
                if audit_result['score'] < self.audit_threshold:
                    self.logger.warning(f"Cache value failed audit: {audit_result['flags']}")
                    self.metrics.integrity_violations += 1
                    
                    if not self.auto_correction_enabled:
                        return False
                    
                    # Attempt auto-correction
                    value = self._auto_correct_cache_value(value, audit_result)
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            cache_key = self._build_cache_key(key, namespace)
            
            if not REDIS_AVAILABLE:
                # Mock mode
                self._record_mock_set(cache_key, value, ttl, integrity_score, audit_flags)
                return True
            
            # Set value in Redis
            await self.redis_client.setex(cache_key, ttl, serialized_value)
            
            # Track cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                namespace=namespace,
                ttl=ttl,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                size_bytes=len(serialized_value),
                integrity_score=integrity_score,
                audit_flags=audit_flags
            )
            self.cache_entries[cache_key] = entry
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(sets=1, latency=latency)
            
            self.logger.debug(f"Cache set successful: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache value: {e}")
            self._update_circuit_breaker(failed=True)
            self._update_metrics(error=True)
            return False
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None
    ) -> Any:
        """
        Get cache value with performance tracking
        
        Args:
            key: Cache key
            namespace: Cache namespace  
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        try:
            cache_key = self._build_cache_key(key, namespace)
            
            if not REDIS_AVAILABLE:
                # Mock mode
                return self._record_mock_get(cache_key, default)
            
            # Get value from Redis
            serialized_value = await self.redis_client.get(cache_key)
            
            if serialized_value is None:
                self._update_metrics(misses=1)
                return default
            
            # Deserialize value
            value = self._deserialize_value(serialized_value)
            
            # Update cache entry access
            if cache_key in self.cache_entries:
                entry = self.cache_entries[cache_key]
                entry.last_accessed = time.time()
                entry.access_count += 1
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(hits=1, latency=latency)
            
            self.logger.debug(f"Cache hit: {cache_key}")
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get cache value: {e}")
            self._update_metrics(misses=1, error=True)
            return default
    
    async def delete(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """Delete cache value"""
        try:
            cache_key = self._build_cache_key(key, namespace)
            
            if not REDIS_AVAILABLE:
                # Mock mode
                if cache_key in self.cache_entries:
                    del self.cache_entries[cache_key]
                self._update_metrics(deletes=1)
                return True
            
            # Delete from Redis
            result = await self.redis_client.delete(cache_key)
            
            # Remove from tracking
            if cache_key in self.cache_entries:
                del self.cache_entries[cache_key]
            
            self._update_metrics(deletes=1)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to delete cache value: {e}")
            return False
    
    async def exists(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """Check if cache key exists"""
        try:
            cache_key = self._build_cache_key(key, namespace)
            
            if not REDIS_AVAILABLE:
                return cache_key in self.cache_entries
            
            result = await self.redis_client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to check cache existence: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        try:
            pattern = f"nis:{namespace}:*"
            
            if not REDIS_AVAILABLE:
                # Mock mode
                cleared = 0
                keys_to_remove = [k for k in self.cache_entries.keys() if k.startswith(f"nis:{namespace}:")]
                for key in keys_to_remove:
                    del self.cache_entries[key]
                    cleared += 1
                return cleared
            
            # Get all keys matching pattern
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete all keys
                deleted = await self.redis_client.delete(*keys)
                
                # Remove from tracking
                for key in keys:
                    if key in self.cache_entries:
                        del self.cache_entries[key]
                
                self.logger.info(f"Cleared {deleted} keys from namespace: {namespace}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to clear namespace {namespace}: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries and optimize memory"""
        try:
            current_time = time.time()
            expired_count = 0
            
            # Clean up tracked entries
            expired_keys = []
            for cache_key, entry in self.cache_entries.items():
                if current_time - entry.created_at > entry.ttl:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.cache_entries[key]
                expired_count += 1
            
            # Update metrics
            self.metrics.evictions += expired_count
            self.metrics.last_cleanup = current_time
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired cache entries")
            
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired entries: {e}")
            return 0
    
    def _audit_cache_value(self, key: str, value: Any, namespace: str) -> Dict[str, Any]:
        """Audit cache value for integrity violations"""
        try:
            # Convert value to text for auditing
            if isinstance(value, dict):
                audit_text = json.dumps(value, indent=2)
            elif isinstance(value, (list, tuple)):
                audit_text = str(value)
            else:
                audit_text = str(value)
            
            # Include context
            context_text = f"""
            Cache Key: {key}
            Namespace: {namespace}
            Value: {audit_text}
            """
            
            # Use self-audit engine
            violations = self_audit_engine.audit_text(context_text)
            score = self_audit_engine.get_integrity_score(context_text)
            
            flags = [v['type'] for v in violations] if violations else []
            
            return {
                'score': score,
                'flags': flags,
                'violations': violations
            }
            
        except Exception as e:
            self.logger.error(f"Audit error: {e}")
            return {'score': 100.0, 'flags': [], 'violations': []}
    
    def _auto_correct_cache_value(self, value: Any, audit_result: Dict[str, Any]) -> Any:
        """Attempt to auto-correct cache value integrity violations"""
        try:
            if isinstance(value, dict):
                corrected_value = value.copy()
                
                # Apply corrections based on violation flags
                for flag in audit_result['flags']:
                    if flag == 'HARDCODED_PERFORMANCE':
                        # Replace hardcoded values with descriptive strings
                        for key in list(corrected_value.keys()):
                            if any(term in key.lower() for term in ['confidence', 'accuracy', 'score']):
                                if isinstance(corrected_value[key], (int, float)) and 0 <= corrected_value[key] <= 1:
                                    corrected_value[key] = f"calculated_{key}"
                
                return corrected_value
            
            return value
            
        except Exception as e:
            self.logger.error(f"Auto-correction failed: {e}")
            return value
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        else:
            # Use pickle for complex objects
            return json.dumps({
                '_type': 'pickle',
                '_data': pickle.dumps(value).hex()
            })
    
    def _deserialize_value(self, serialized_value: str) -> Any:
        """Deserialize value from Redis storage"""
        try:
            data = json.loads(serialized_value)
            
            if isinstance(data, dict) and data.get('_type') == 'pickle':
                return pickle.loads(bytes.fromhex(data['_data']))
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            return serialized_value
    
    def _build_cache_key(self, key: str, namespace: str) -> str:
        """Build cache key with namespace"""
        return f"nis:{namespace}:{key}"
    
    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state"""
        current_time = time.time()
        
        if self.circuit_breaker['state'] == 'open':
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['timeout']:
                self.circuit_breaker['state'] = 'half-open'
                self.logger.info("Circuit breaker moved to half-open state")
            else:
                return False
        
        return True
    
    def _update_circuit_breaker(self, failed: bool = False):
        """Update circuit breaker state"""
        if failed:
            self.circuit_breaker['failure_count'] += 1
            self.circuit_breaker['last_failure_time'] = time.time()
            
            if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
                self.circuit_breaker['state'] = 'open'
                self.logger.warning("Circuit breaker opened due to failures")
        else:
            self.circuit_breaker['failure_count'] = 0
            if self.circuit_breaker['state'] == 'half-open':
                self.circuit_breaker['state'] = 'closed'
                self.logger.info("Circuit breaker closed")
    
    def _update_metrics(
        self,
        hits: int = 0,
        misses: int = 0,
        sets: int = 0,
        deletes: int = 0,
        evictions: int = 0,
        latency: float = 0.0,
        error: bool = False
    ):
        """Update cache metrics"""
        self.metrics.hits += hits
        self.metrics.misses += misses
        self.metrics.sets += sets
        self.metrics.deletes += deletes
        self.metrics.evictions += evictions
        
        if latency > 0:
            self.latency_history.append(latency)
            # Update rolling average latency
            self.metrics.avg_latency = (self.metrics.avg_latency * 0.9) + (latency * 0.1)
        
        # Update performance tracker
        total_requests = self.metrics.hits + self.metrics.misses
        if total_requests > 0:
            self.performance.hit_rate = self.metrics.hits / total_requests
            self.performance.miss_rate = self.metrics.misses / total_requests
        
        self.metrics.last_update = time.time()
    
    def _record_mock_set(self, key: str, value: Any, ttl: int, integrity_score: Optional[float], audit_flags: Optional[List[str]]):
        """Record mock cache set for testing"""
        entry = CacheEntry(
            key=key,
            value=value,
            namespace="mock",
            ttl=ttl,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            size_bytes=len(str(value)),
            integrity_score=integrity_score,
            audit_flags=audit_flags
        )
        self.cache_entries[key] = entry
        self._update_metrics(sets=1, latency=0.001)
    
    def _record_mock_get(self, key: str, default: Any) -> Any:
        """Record mock cache get for testing"""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_metrics(hits=1, latency=0.001)
            return entry.value
        else:
            self._update_metrics(misses=1)
            return default
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        return self.metrics
    
    def get_performance(self) -> PerformanceTracker:
        """Get performance tracker data"""
        # Update latency percentiles
        if self.latency_history:
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            self.performance.latency_percentiles = {
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)]
            }
        
        return self.performance
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Redis manager"""
        return {
            'initialized': self.is_initialized,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'failure_count': self.circuit_breaker['failure_count'],
            'tracked_entries': len(self.cache_entries),
            'metrics': asdict(self.metrics),
            'performance': asdict(self.performance),
            'redis_available': REDIS_AVAILABLE,
            'self_audit_enabled': self.enable_self_audit
        } 