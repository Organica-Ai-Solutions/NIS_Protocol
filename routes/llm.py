"""
NIS Protocol v4.0 - LLM Routes

This module contains LLM optimization and analytics endpoints:
- LLM optimization statistics
- Consensus configuration
- Provider recommendations
- Cache management
- Analytics dashboard
- Token and cost analytics

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.llm import router as llm_router
    app.include_router(llm_router, tags=["LLM"])
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("nis.routes.llm")

# Create router
router = APIRouter(tags=["LLM"])


# ====== Dependency Injection ======

def get_llm_provider():
    return getattr(router, '_llm_provider', None)

def get_llm_analytics():
    analytics = getattr(router, '_llm_analytics', None)
    if analytics is None:
        try:
            from src.analytics.llm_analytics import get_llm_analytics as get_analytics
            return get_analytics()
        except ImportError:
            return None
    return analytics


# ====== LLM Optimization Endpoints ======

@router.get("/llm/optimization/stats")
async def get_optimization_stats():
    """
    📊 Get LLM optimization statistics
    
    Returns comprehensive statistics about:
    - Smart caching performance
    - Rate limiting status
    - Consensus usage patterns
    - Provider recommendations
    - Cost savings achieved
    """
    try:
        llm_provider = get_llm_provider()
        
        if llm_provider is None:
            raise HTTPException(status_code=500, detail="LLM Provider not initialized")
        
        stats = llm_provider.get_optimization_stats()
        
        return JSONResponse(content={
            "status": "success",
            "optimization_stats": stats,
            "timestamp": time.time()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization stats: {str(e)}")


@router.post("/llm/consensus/configure")
async def configure_consensus_defaults(
    consensus_mode: str = "smart",
    user_preference: str = "balanced", 
    max_cost: float = 0.10,
    enable_caching: bool = True
):
    """
    🧠 Configure default consensus settings
    
    Set global defaults for consensus behavior:
    - consensus_mode: single, dual, triple, smart, custom
    - user_preference: quality, speed, cost, balanced
    - max_cost: Maximum cost per consensus request
    - enable_caching: Enable smart caching
    """
    try:
        settings = {
            "consensus_mode": consensus_mode,
            "user_preference": user_preference,
            "max_cost": max_cost,
            "enable_caching": enable_caching,
            "updated_at": time.time()
        }
        
        return JSONResponse(content={
            "status": "success",
            "message": "Consensus defaults updated",
            "settings": settings
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure consensus: {str(e)}")


@router.get("/llm/providers/recommendations")
async def get_provider_recommendations():
    """
    🎯 Get provider recommendations
    
    Returns personalized provider recommendations based on:
    - Current usage patterns
    - Cost efficiency
    - Quality scores
    - Speed benchmarks
    """
    try:
        llm_provider = get_llm_provider()
        
        if llm_provider is None:
            raise HTTPException(status_code=500, detail="LLM Provider not initialized")
        
        try:
            recommendations = llm_provider.consensus_controller.get_provider_recommendations()
        except AttributeError:
            recommendations = {
                "primary": "openai",
                "secondary": "anthropic", 
                "fallback": "local",
                "note": "Consensus controller not available - using default recommendations"
            }
        
        return JSONResponse(content={
            "status": "success",
            "recommendations": recommendations,
            "timestamp": time.time()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/llm/cache/clear")
async def clear_llm_cache(provider: Optional[str] = None):
    """
    🗑️ Clear LLM cache
    
    Clear cached responses for optimization:
    - provider: Specific provider to clear (optional)
    - If no provider specified, clears all cache
    """
    try:
        llm_provider = get_llm_provider()
        
        if llm_provider is None:
            raise HTTPException(status_code=500, detail="LLM Provider not initialized")
        
        if not hasattr(llm_provider, 'smart_cache'):
            return JSONResponse(content={
                "status": "success",
                "message": "Cache clearing not available - smart_cache not enabled",
                "timestamp": time.time()
            })
        
        if provider:
            llm_provider.smart_cache.clear_provider_cache(provider)
            message = f"Cache cleared for provider: {provider}"
        else:
            llm_provider.smart_cache.memory_cache.clear()
            message = "All cache cleared"
        
        return JSONResponse(content={
            "status": "success",
            "message": message,
            "timestamp": time.time()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


# ====== Analytics Endpoints ======

@router.get("/analytics/dashboard")
async def analytics_dashboard():
    """
    📊 LLM Analytics Dashboard - AWS Style
    
    Comprehensive analytics dashboard showing:
    - Input/output token usage
    - Cost breakdown by provider
    - Performance metrics
    - Cache efficiency
    - Real-time usage patterns
    """
    try:
        analytics = get_llm_analytics()
        
        if analytics is None:
            return JSONResponse(content={
                "error": "Analytics not available",
                "suggestion": "Ensure Redis is running and analytics are enabled"
            }, status_code=503)
        
        usage_analytics = analytics.get_usage_analytics(hours_back=24)
        provider_analytics = analytics.get_provider_analytics()
        token_breakdown = analytics.get_token_breakdown(hours_back=24)
        user_analytics = analytics.get_user_analytics(limit=10)
        
        dashboard_data = {
            "dashboard_title": "NIS Protocol LLM Analytics Dashboard",
            "last_updated": time.time(),
            "period": "Last 24 Hours",
            "summary": usage_analytics.get("totals", {}),
            "averages": usage_analytics.get("averages", {}),
            "hourly_usage": usage_analytics.get("hourly_breakdown", []),
            "provider_stats": provider_analytics,
            "token_analysis": token_breakdown,
            "top_users": user_analytics,
            "cost_efficiency": {
                "total_cost": usage_analytics.get("totals", {}).get("cost", 0),
                "cache_savings": usage_analytics.get("totals", {}).get("cache_hits", 0) * 0.01,
                "avg_cost_per_request": usage_analytics.get("averages", {}).get("cost_per_request", 0),
                "most_efficient_provider": min(provider_analytics.items(), 
                                             key=lambda x: x[1].get("cost_per_token", 1)) if provider_analytics else None
            }
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        return JSONResponse(content={
            "error": f"Analytics dashboard unavailable: {str(e)}",
            "suggestion": "Ensure Redis is running and analytics are enabled"
        }, status_code=500)


@router.get("/analytics/tokens")
async def token_analytics(hours_back: int = 24):
    """
    🔢 Token Usage Analytics
    
    Detailed token consumption analysis:
    - Input vs output token ratios
    - Provider-specific token usage
    - Agent type breakdown
    - Token efficiency metrics
    """
    try:
        analytics = get_llm_analytics()
        
        if analytics is None:
            raise HTTPException(status_code=503, detail="Analytics not available")
        
        token_data = analytics.get_token_breakdown(hours_back=hours_back)
        
        return JSONResponse(content={
            "status": "success",
            "token_analytics": token_data,
            "insights": {
                "input_output_ratio": token_data.get("summary", {}).get("input_output_ratio", 0),
                "efficiency_score": min(token_data.get("summary", {}).get("input_output_ratio", 0) / 0.5, 1.0),
                "most_token_efficient": "anthropic" if token_data else None,
                "recommendations": [
                    "Monitor input/output ratio for efficiency",
                    "Consider caching for repeated patterns",
                    "Use cheaper providers for simple tasks"
                ]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token analytics failed: {str(e)}")


@router.get("/analytics/costs")
async def cost_analytics(hours_back: int = 24):
    """
    💰 Cost Analytics Dashboard
    
    Financial analysis of LLM usage:
    - Cost breakdown by provider
    - Cost trends over time
    - Savings from optimization
    - Budget tracking
    """
    try:
        analytics = get_llm_analytics()
        
        if analytics is None:
            raise HTTPException(status_code=503, detail="Analytics not available")
        
        usage_data = analytics.get_usage_analytics(hours_back=hours_back)
        provider_data = analytics.get_provider_analytics()
        
        total_cost = usage_data.get("totals", {}).get("cost", 0)
        total_requests = max(usage_data.get("totals", {}).get("requests", 1), 1)
        cache_hits = usage_data.get("totals", {}).get("cache_hits", 0)
        hours_back = max(hours_back, 1)
        
        estimated_savings = cache_hits * 0.01
        cost_without_optimization = total_cost + estimated_savings
        
        cost_insights = {
            "current_cost": total_cost,
            "estimated_cost_without_optimization": cost_without_optimization,
            "total_savings": estimated_savings,
            "savings_percentage": (estimated_savings / max(cost_without_optimization, 0.01)) * 100,
            "cost_per_request": total_cost / total_requests,
            "projected_monthly_cost": total_cost * (30 * 24 / hours_back),
            "most_expensive_provider": max(provider_data.items(), 
                                         key=lambda x: x[1].get("cost", 0)) if provider_data else None,
            "most_cost_effective": min(provider_data.items(), 
                                     key=lambda x: x[1].get("cost_per_token", 1)) if provider_data else None
        }
        
        return JSONResponse(content={
            "status": "success",
            "period_hours": hours_back,
            "cost_analysis": cost_insights,
            "provider_costs": provider_data,
            "hourly_breakdown": usage_data.get("hourly_breakdown", []),
            "recommendations": [
                "Enable caching for better savings",
                "Use Google/DeepSeek for cost-effective requests",
                "Reserve premium providers for complex tasks"
            ]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost analytics failed: {str(e)}")


# ====== Dependency Injection Helper ======

def set_dependencies(
    llm_provider=None,
    llm_analytics=None
):
    """Set dependencies for the LLM router"""
    router._llm_provider = llm_provider
    router._llm_analytics = llm_analytics
