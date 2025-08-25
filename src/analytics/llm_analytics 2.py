#!/usr/bin/env python3
"""
🎯 NIS Protocol LLM Analytics System
AWS-style analytics for LLM usage with Redis backend

Features:
- Input/output token tracking
- Cost breakdown by provider/user/time
- Real-time usage monitoring
- Performance analytics
- Redis-based data storage
- Interactive dashboard endpoints
"""

import json
import logging
import os
import time
import redis
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """LLM request analytics record"""
    request_id: str
    timestamp: float
    user_id: str
    provider: str
    model: str
    agent_type: str
    
    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Costs
    input_cost: float
    output_cost: float
    total_cost: float
    
    # Performance
    latency_ms: float
    cache_hit: bool
    consensus_mode: Optional[str] = None
    providers_used: Optional[List[str]] = None
    
    # Quality metrics
    confidence: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0

@dataclass
class AnalyticsMetrics:
    """Aggregated analytics metrics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    # Provider breakdown
    provider_stats: Dict[str, Dict[str, Any]] = None
    
    # Time-based stats
    hourly_usage: Dict[str, int] = None
    daily_cost: Dict[str, float] = None

class LLMAnalytics:
    """
    📊 AWS-style LLM Analytics System
    
    Tracks all LLM usage with detailed metrics:
    - Token consumption (input/output)
    - Cost breakdown by provider/user/time
    - Performance monitoring
    - Cache efficiency
    - Error tracking
    """
    
    def __init__(self, redis_host: str = None, redis_port: int = 6379, redis_db: int = 1):
        """Initialize analytics with Redis backend"""
        
        # Auto-detect Redis host
        if redis_host is None:
            # Try environment variable first
            redis_host = os.environ.get('REDIS_HOST')
            if not redis_host:
                # Check if we're in a Docker environment
                if os.path.exists('/.dockerenv'):
                    redis_host = "nis-redis"  # Docker container name
                else:
                    redis_host = "localhost"  # Local development
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis analytics backend: {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # Analytics keys
        self.REQUESTS_KEY = "llm:requests"
        self.HOURLY_STATS_KEY = "llm:hourly"
        self.DAILY_STATS_KEY = "llm:daily"
        self.PROVIDER_STATS_KEY = "llm:providers"
        self.USER_STATS_KEY = "llm:users"
        self.COST_TRACKING_KEY = "llm:costs"
        
        # Provider cost rates (per 1K tokens)
        self.provider_costs = {
            "openai": {"input": 0.0025, "output": 0.0075},
            "anthropic": {"input": 0.003, "output": 0.015},
            "google": {"input": 0.000375, "output": 0.0015},
            "deepseek": {"input": 0.00055, "output": 0.002},
            "nvidia": {"input": 0.005, "output": 0.015},
            "bitnet": {"input": 0.0, "output": 0.0}
        }
        
        logger.info("📊 LLM Analytics system initialized")
    
    def record_request(self, 
                      request_id: str,
                      user_id: str,
                      provider: str,
                      model: str,
                      agent_type: str,
                      input_tokens: int,
                      output_tokens: int,
                      latency_ms: float,
                      cache_hit: bool = False,
                      consensus_mode: Optional[str] = None,
                      providers_used: Optional[List[str]] = None,
                      confidence: float = 0.0,
                      error: Optional[str] = None,
                      retry_count: int = 0) -> None:
        """Record LLM request for analytics"""
        
        timestamp = time.time()
        total_tokens = input_tokens + output_tokens
        
        # Calculate costs
        costs = self.provider_costs.get(provider, {"input": 0.01, "output": 0.02})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        # Create request record
        request = LLMRequest(
            request_id=request_id,
            timestamp=timestamp,
            user_id=user_id,
            provider=provider,
            model=model,
            agent_type=agent_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            consensus_mode=consensus_mode,
            providers_used=providers_used,
            confidence=confidence,
            error=error,
            retry_count=retry_count
        )
        
        # Store in Redis
        self._store_request(request)
        self._update_aggregated_stats(request)
        
        logger.debug(f"📊 Recorded LLM request: {provider} - {total_tokens} tokens, ${total_cost:.4f}")
    
    def _store_request(self, request: LLMRequest) -> None:
        """Store individual request in Redis"""
        try:
            # Store detailed request
            request_key = f"{self.REQUESTS_KEY}:{request.request_id}"
            self.redis_client.hset(request_key, mapping=asdict(request))
            self.redis_client.expire(request_key, 7 * 24 * 3600)  # Keep for 7 days
            
            # Add to time-ordered list for recent requests
            self.redis_client.zadd(
                f"{self.REQUESTS_KEY}:timeline", 
                {request.request_id: request.timestamp}
            )
            
            # Trim old entries (keep last 10,000 requests)
            self.redis_client.zremrangebyrank(f"{self.REQUESTS_KEY}:timeline", 0, -10001)
            
        except Exception as e:
            logger.error(f"Failed to store request: {e}")
    
    def _update_aggregated_stats(self, request: LLMRequest) -> None:
        """Update aggregated statistics"""
        try:
            current_hour = datetime.fromtimestamp(request.timestamp).strftime("%Y-%m-%d:%H")
            current_day = datetime.fromtimestamp(request.timestamp).strftime("%Y-%m-%d")
            
            # Update hourly stats
            hourly_key = f"{self.HOURLY_STATS_KEY}:{current_hour}"
            self.redis_client.hincrby(hourly_key, "requests", 1)
            self.redis_client.hincrby(hourly_key, "tokens", request.total_tokens)
            self.redis_client.hincrbyfloat(hourly_key, "cost", request.total_cost)
            self.redis_client.hincrbyfloat(hourly_key, "latency_sum", request.latency_ms)
            if request.cache_hit:
                self.redis_client.hincrby(hourly_key, "cache_hits", 1)
            if request.error:
                self.redis_client.hincrby(hourly_key, "errors", 1)
            self.redis_client.expire(hourly_key, 30 * 24 * 3600)  # Keep for 30 days
            
            # Update daily stats
            daily_key = f"{self.DAILY_STATS_KEY}:{current_day}"
            self.redis_client.hincrby(daily_key, "requests", 1)
            self.redis_client.hincrby(daily_key, "tokens", request.total_tokens)
            self.redis_client.hincrbyfloat(daily_key, "cost", request.total_cost)
            self.redis_client.expire(daily_key, 90 * 24 * 3600)  # Keep for 90 days
            
            # Update provider stats
            provider_key = f"{self.PROVIDER_STATS_KEY}:{request.provider}"
            self.redis_client.hincrby(provider_key, "requests", 1)
            self.redis_client.hincrby(provider_key, "tokens", request.total_tokens)
            self.redis_client.hincrbyfloat(provider_key, "cost", request.total_cost)
            self.redis_client.hincrbyfloat(provider_key, "latency_sum", request.latency_ms)
            
            # Update user stats
            user_key = f"{self.USER_STATS_KEY}:{request.user_id}"
            self.redis_client.hincrby(user_key, "requests", 1)
            self.redis_client.hincrby(user_key, "tokens", request.total_tokens)
            self.redis_client.hincrbyfloat(user_key, "cost", request.total_cost)
            self.redis_client.expire(user_key, 90 * 24 * 3600)  # Keep for 90 days
            
        except Exception as e:
            logger.error(f"Failed to update aggregated stats: {e}")
    
    def get_recent_requests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent LLM requests"""
        try:
            # Get recent request IDs
            request_ids = self.redis_client.zrevrange(
                f"{self.REQUESTS_KEY}:timeline", 0, limit - 1
            )
            
            requests = []
            for request_id in request_ids:
                request_key = f"{self.REQUESTS_KEY}:{request_id}"
                request_data = self.redis_client.hgetall(request_key)
                if request_data:
                    # Convert numeric fields
                    for field in ['timestamp', 'input_tokens', 'output_tokens', 'total_tokens', 
                                'input_cost', 'output_cost', 'total_cost', 'latency_ms', 'confidence']:
                        if field in request_data:
                            try:
                                request_data[field] = float(request_data[field])
                            except:
                                pass
                    
                    # Convert boolean fields
                    request_data['cache_hit'] = request_data.get('cache_hit', 'False') == 'True'
                    
                    # Parse lists
                    if request_data.get('providers_used'):
                        try:
                            request_data['providers_used'] = json.loads(request_data['providers_used'])
                        except:
                            pass
                    
                    requests.append(request_data)
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to get recent requests: {e}")
            return []
    
    def get_usage_analytics(self, 
                          hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive usage analytics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Generate hourly keys for the period
            hourly_data = []
            current_time = start_time
            
            while current_time <= end_time:
                hour_key = current_time.strftime("%Y-%m-%d:%H")
                hourly_key = f"{self.HOURLY_STATS_KEY}:{hour_key}"
                
                hour_stats = self.redis_client.hgetall(hourly_key)
                if hour_stats:
                    # Convert to proper types
                    hour_data = {
                        "hour": hour_key,
                        "requests": int(hour_stats.get("requests", 0)),
                        "tokens": int(hour_stats.get("tokens", 0)),
                        "cost": float(hour_stats.get("cost", 0)),
                        "cache_hits": int(hour_stats.get("cache_hits", 0)),
                        "errors": int(hour_stats.get("errors", 0)),
                        "avg_latency": 0
                    }
                    
                    # Calculate average latency
                    if hour_data["requests"] > 0:
                        latency_sum = float(hour_stats.get("latency_sum", 0))
                        hour_data["avg_latency"] = latency_sum / hour_data["requests"]
                        hour_data["cache_hit_rate"] = hour_data["cache_hits"] / hour_data["requests"]
                        hour_data["error_rate"] = hour_data["errors"] / hour_data["requests"]
                    
                    hourly_data.append(hour_data)
                else:
                    # No data for this hour
                    hourly_data.append({
                        "hour": hour_key,
                        "requests": 0,
                        "tokens": 0,
                        "cost": 0,
                        "cache_hits": 0,
                        "errors": 0,
                        "avg_latency": 0,
                        "cache_hit_rate": 0,
                        "error_rate": 0
                    })
                
                current_time += timedelta(hours=1)
            
            # Calculate totals
            total_requests = sum(h["requests"] for h in hourly_data)
            total_tokens = sum(h["tokens"] for h in hourly_data)
            total_cost = sum(h["cost"] for h in hourly_data)
            total_cache_hits = sum(h["cache_hits"] for h in hourly_data)
            total_errors = sum(h["errors"] for h in hourly_data)
            
            # Calculate averages
            avg_cache_hit_rate = total_cache_hits / max(total_requests, 1)
            avg_error_rate = total_errors / max(total_requests, 1)
            avg_latency = sum(h["avg_latency"] * h["requests"] for h in hourly_data) / max(total_requests, 1)
            
            return {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours_back
                },
                "totals": {
                    "requests": total_requests,
                    "tokens": total_tokens,
                    "cost": round(total_cost, 4),
                    "cache_hits": total_cache_hits,
                    "errors": total_errors
                },
                "averages": {
                    "cache_hit_rate": round(avg_cache_hit_rate, 3),
                    "error_rate": round(avg_error_rate, 3),
                    "latency_ms": round(avg_latency, 1),
                    "cost_per_request": round(total_cost / max(total_requests, 1), 4),
                    "tokens_per_request": round(total_tokens / max(total_requests, 1), 1)
                },
                "hourly_breakdown": hourly_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage analytics: {e}")
            return {}
    
    def get_provider_analytics(self) -> Dict[str, Any]:
        """Get provider-specific analytics"""
        try:
            provider_stats = {}
            
            for provider in self.provider_costs.keys():
                provider_key = f"{self.PROVIDER_STATS_KEY}:{provider}"
                stats = self.redis_client.hgetall(provider_key)
                
                if stats:
                    requests = int(stats.get("requests", 0))
                    tokens = int(stats.get("tokens", 0))
                    cost = float(stats.get("cost", 0))
                    latency_sum = float(stats.get("latency_sum", 0))
                    
                    provider_stats[provider] = {
                        "requests": requests,
                        "tokens": tokens,
                        "cost": round(cost, 4),
                        "avg_latency": round(latency_sum / max(requests, 1), 1),
                        "cost_per_token": round(cost / max(tokens, 1), 6),
                        "tokens_per_request": round(tokens / max(requests, 1), 1)
                    }
                else:
                    provider_stats[provider] = {
                        "requests": 0,
                        "tokens": 0,
                        "cost": 0,
                        "avg_latency": 0,
                        "cost_per_token": 0,
                        "tokens_per_request": 0
                    }
            
            return provider_stats
            
        except Exception as e:
            logger.error(f"Failed to get provider analytics: {e}")
            return {}
    
    def get_user_analytics(self, limit: int = 10) -> Dict[str, Any]:
        """Get user usage analytics"""
        try:
            # Get all user keys
            user_keys = self.redis_client.keys(f"{self.USER_STATS_KEY}:*")
            user_stats = {}
            
            for user_key in user_keys:
                user_id = user_key.split(":")[-1]
                stats = self.redis_client.hgetall(user_key)
                
                if stats:
                    requests = int(stats.get("requests", 0))
                    tokens = int(stats.get("tokens", 0))
                    cost = float(stats.get("cost", 0))
                    
                    user_stats[user_id] = {
                        "requests": requests,
                        "tokens": tokens,
                        "cost": round(cost, 4),
                        "avg_tokens_per_request": round(tokens / max(requests, 1), 1),
                        "avg_cost_per_request": round(cost / max(requests, 1), 4)
                    }
            
            # Sort by cost and return top users
            sorted_users = sorted(
                user_stats.items(), 
                key=lambda x: x[1]["cost"], 
                reverse=True
            )[:limit]
            
            return dict(sorted_users)
            
        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return {}
    
    def get_token_breakdown(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get detailed token usage breakdown"""
        try:
            # Get recent requests for token analysis
            recent_requests = self.get_recent_requests(1000)
            
            # Filter by time period
            cutoff_time = time.time() - (hours_back * 3600)
            filtered_requests = [
                r for r in recent_requests 
                if r.get("timestamp", 0) > cutoff_time
            ]
            
            # Analyze token patterns
            total_input_tokens = sum(r.get("input_tokens", 0) for r in filtered_requests)
            total_output_tokens = sum(r.get("output_tokens", 0) for r in filtered_requests)
            
            # Provider breakdown
            provider_tokens = defaultdict(lambda: {"input": 0, "output": 0, "requests": 0})
            
            for request in filtered_requests:
                provider = request.get("provider", "unknown")
                provider_tokens[provider]["input"] += request.get("input_tokens", 0)
                provider_tokens[provider]["output"] += request.get("output_tokens", 0)
                provider_tokens[provider]["requests"] += 1
            
            # Agent type breakdown
            agent_tokens = defaultdict(lambda: {"input": 0, "output": 0, "requests": 0})
            
            for request in filtered_requests:
                agent_type = request.get("agent_type", "unknown")
                agent_tokens[agent_type]["input"] += request.get("input_tokens", 0)
                agent_tokens[agent_type]["output"] += request.get("output_tokens", 0)
                agent_tokens[agent_type]["requests"] += 1
            
            return {
                "summary": {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "input_output_ratio": round(total_input_tokens / max(total_output_tokens, 1), 2),
                    "avg_input_per_request": round(total_input_tokens / max(len(filtered_requests), 1), 1),
                    "avg_output_per_request": round(total_output_tokens / max(len(filtered_requests), 1), 1)
                },
                "by_provider": dict(provider_tokens),
                "by_agent_type": dict(agent_tokens),
                "period_hours": hours_back,
                "total_requests": len(filtered_requests)
            }
            
        except Exception as e:
            logger.error(f"Failed to get token breakdown: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old analytics data"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            # Clean old requests from timeline
            removed_requests = self.redis_client.zremrangebyscore(
                f"{self.REQUESTS_KEY}:timeline", 0, cutoff_time
            )
            
            # Clean old hourly stats
            cutoff_date = datetime.fromtimestamp(cutoff_time)
            removed_hourly = 0
            
            for i in range(days_to_keep + 10):  # Buffer for safety
                old_date = cutoff_date - timedelta(days=i)
                for hour in range(24):
                    old_hour_key = f"{self.HOURLY_STATS_KEY}:{old_date.strftime('%Y-%m-%d')}:{hour:02d}"
                    if self.redis_client.delete(old_hour_key):
                        removed_hourly += 1
            
            logger.info(f"🧹 Cleaned up analytics: {removed_requests} requests, {removed_hourly} hourly stats")
            
            return {
                "removed_requests": removed_requests,
                "removed_hourly_stats": removed_hourly
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {}


# Global analytics instance
_global_analytics: Optional[LLMAnalytics] = None

def get_llm_analytics() -> LLMAnalytics:
    """Get global LLM analytics instance"""
    global _global_analytics
    if _global_analytics is None:
        # Auto-detect Redis host for Docker/local environments
        redis_host = None
        if os.path.exists('/.dockerenv'):
            redis_host = "nis-redis"  # Docker container name
        _global_analytics = LLMAnalytics(redis_host=redis_host)
    return _global_analytics

def init_llm_analytics(redis_host: str = None, 
                      redis_port: int = 6379, 
                      redis_db: int = 1) -> LLMAnalytics:
    """Initialize global LLM analytics"""
    global _global_analytics
    _global_analytics = LLMAnalytics(redis_host, redis_port, redis_db)
    return _global_analytics
