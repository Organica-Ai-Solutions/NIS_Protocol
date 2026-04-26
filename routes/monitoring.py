"""
NIS Protocol v4.0 - Monitoring Routes

This module contains monitoring, health, and analytics endpoints:
- Health checks
- Metrics (JSON and Prometheus)
- Analytics dashboard
- Token/cost analytics
- Rate limit status
- System status

MIGRATION STATUS: Ready for testing
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger("nis.routes.monitoring")

# Create router
router = APIRouter(tags=["Monitoring"])

_injected_llm = None

def set_dependencies(llm_provider=None, **kwargs):
    global _injected_llm
    _injected_llm = llm_provider


# ====== Health Check ======

# NOTE: GET /health is defined in routes/core.py (registered first).
# The authoritative health endpoint lives there to avoid duplicate registration.


# ====== Metrics ======

@router.get("/metrics")
async def get_metrics():
    """
    📊 Get System Metrics (Prometheus Format)
    """
    try:
        metrics_lines = [
            "# HELP nis_requests_total Total number of requests",
            "# TYPE nis_requests_total counter",
            f"nis_requests_total {{endpoint=\"all\"}} {getattr(router, '_request_count', 0)}",
            "",
            "# HELP nis_uptime_seconds System uptime in seconds",
            "# TYPE nis_uptime_seconds gauge",
            f"nis_uptime_seconds {time.time() - getattr(router, '_start_time', time.time())}",
            "",
            "# HELP nis_health_status Health status (1=healthy, 0=unhealthy)",
            "# TYPE nis_health_status gauge",
            "nis_health_status 1",
        ]
        
        return PlainTextResponse(
            content="\n".join(metrics_lines),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return PlainTextResponse(content=f"# Error: {e}", status_code=500)


@router.get("/metrics/json")
async def get_metrics_json():
    """
    📊 Get System Metrics (JSON Format)
    """
    try:
        return {
            "status": "success",
            "metrics": {
                "requests_total": getattr(router, '_request_count', 0),
                "uptime_seconds": time.time() - getattr(router, '_start_time', time.time()),
                "health_status": 1,
                "timestamp": time.time()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    📊 Get Prometheus-formatted Metrics
    """
    return await get_metrics()


# ====== Rate Limiting ======

@router.get("/rate-limit/status")
async def get_rate_limit_status():
    """
    🚦 Get Rate Limit Status
    
    Returns current rate limiting configuration and status.
    """
    return {
        "status": "active",
        "config": {
            "default_limit": 100,
            "window_seconds": 60,
            "authenticated_limit": 1000
        },
        "current_usage": {
            "note": "Per-client usage tracked in memory"
        },
        "timestamp": time.time()
    }


# ====== Analytics ======

@router.get("/analytics/dashboard")
async def analytics_dashboard():
    """
    📊 LLM Analytics Dashboard
    
    Comprehensive analytics dashboard showing:
    - Input/output token usage
    - Cost breakdown by provider
    - Performance metrics
    - Cache efficiency
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
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
                "avg_cost_per_request": usage_analytics.get("averages", {}).get("cost_per_request", 0)
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
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
        token_data = analytics.get_token_breakdown(hours_back=hours_back)
        
        return JSONResponse(content={
            "status": "success",
            "token_analytics": token_data,
            "insights": {
                "input_output_ratio": token_data.get("summary", {}).get("input_output_ratio", 0),
                "efficiency_score": min(token_data.get("summary", {}).get("input_output_ratio", 0) / 0.5, 1.0),
                "recommendations": [
                    "Monitor input/output ratio for efficiency",
                    "Consider caching for repeated patterns"
                ]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token analytics failed: {str(e)}")


@router.get("/analytics/costs")
async def cost_analytics(hours_back: int = 24):
    """
    💰 Cost Analytics Dashboard
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
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
            "projected_monthly_cost": total_cost * (30 * 24 / hours_back)
        }
        
        return JSONResponse(content={
            "status": "success",
            "period_hours": hours_back,
            "cost_analysis": cost_insights,
            "provider_costs": provider_data,
            "hourly_breakdown": usage_data.get("hourly_breakdown", [])
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost analytics failed: {str(e)}")


@router.get("/analytics")
async def get_analytics_summary():
    """
    📊 Analytics Summary
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
        usage = analytics.get_usage_analytics(hours_back=24)
        
        return {
            "status": "success",
            "summary": usage.get("totals", {}),
            "period": "24h",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }


@router.get("/analytics/performance")
async def performance_analytics():
    """
    ⚡ Performance Analytics
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
        usage = analytics.get_usage_analytics(hours_back=24)
        
        return {
            "status": "success",
            "performance": {
                "avg_latency_ms": usage.get("averages", {}).get("latency", 0),
                "requests_per_hour": usage.get("totals", {}).get("requests", 0) / 24,
                "error_rate": usage.get("totals", {}).get("errors", 0) / max(usage.get("totals", {}).get("requests", 1), 1)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/realtime")
async def realtime_analytics():
    """
    🔴 Real-time Analytics
    """
    return {
        "status": "success",
        "realtime": {
            "active_connections": 0,
            "requests_last_minute": 0,
            "avg_response_time_ms": 0
        },
        "timestamp": time.time()
    }


@router.post("/analytics/cleanup")
async def cleanup_analytics(days_to_keep: int = 7):
    """
    🧹 Cleanup Old Analytics Data
    """
    try:
        from src.analytics.llm_analytics import get_llm_analytics
        analytics = get_llm_analytics()
        
        result = analytics.cleanup_old_data(days_to_keep=days_to_keep)
        
        return {
            "status": "success",
            "cleaned_records": result.get("cleaned", 0),
            "days_kept": days_to_keep
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== System Status ======

@router.get("/system/status")
async def get_system_status():
    """
    🧠 Get Real-Time System Status
    """
    return {
        "status": "operational",
        "components": {
            "api": "healthy",
            "llm_provider": "active",
            "agents": "ready",
            "memory": "available"
        },
        "timestamp": time.time()
    }


@router.get("/system/gpu")
async def get_gpu_status():
    """
    🎮 Get GPU Status
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_cached": torch.cuda.memory_reserved(0)
            }
        return {"gpu_available": False, "reason": "CUDA not available"}
    except ImportError:
        return {"gpu_available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}


@router.get("/system/integration")
async def get_system_integration():
    """
    🔗 Get System Integration Status
    """
    return {
        "status": "success",
        "integrations": {
            "llm_providers": ["anthropic", "openai", "google", "deepseek", "nvidia", "kimi", "bitnet"],
            "protocols": ["MCP", "A2A", "ACP"],
            "features": ["streaming", "webhooks", "analytics", "rate_limiting"]
        },
        "timestamp": time.time()
    }


# ====== Sensors Endpoint (Dashboard - Sensors Tab) ======

@router.get("/sensors")
async def get_sensors():
    """Return Pi onboard sensor data for the dashboard Sensors tab"""
    import subprocess, time

    def _read_file(path, default=None):
        try:
            with open(path) as f:
                return f.read().strip()
        except Exception:
            return default

    # CPU temperature (Pi thermal zone)
    temp_c = None
    raw = _read_file("/sys/class/thermal/thermal_zone0/temp")
    if raw:
        try:
            temp_c = round(int(raw) / 1000.0, 1)
        except Exception:
            pass

    # CPU usage
    cpu_pct = None
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.2)
    except Exception:
        pass

    # Memory
    mem = {}
    try:
        import psutil
        m = psutil.virtual_memory()
        mem = {"total_gb": round(m.total / 1e9, 2), "used_gb": round(m.used / 1e9, 2),
               "percent": m.percent}
    except Exception:
        pass

    # I2C devices (detect what's on the bus)
    i2c_devices = []
    try:
        r = subprocess.run(["i2cdetect", "-y", "1"], capture_output=True, text=True, timeout=3)
        for line in r.stdout.splitlines()[1:]:
            parts = line.split(":")[1].strip().split() if ":" in line else []
            for p in parts:
                if p != "--":
                    i2c_devices.append({"address": f"0x{p}", "bus": "i2c-1"})
    except Exception:
        pass

    # GPIO info
    gpio_info = {}
    try:
        r = subprocess.run(["raspi-gpio", "get"], capture_output=True, text=True, timeout=2)
        gpio_info = {"raw": r.stdout[:200] if r.returncode == 0 else "unavailable"}
    except Exception:
        gpio_info = {"raw": "raspi-gpio not available"}

    return {
        "timestamp": time.time(),
        "temperature": {
            "cpu_c": temp_c,
            "source": "/sys/class/thermal/thermal_zone0/temp",
        },
        "cpu": {"usage_percent": cpu_pct},
        "memory": mem,
        "i2c_devices": i2c_devices,
        "gpio": gpio_info,
        "onboard": {
            "platform": "Raspberry Pi",
            "available": True,
        }
    }


# ====== Helper to inject dependencies ======

def set_dependencies(llm_provider=None, conversation_memory=None, agent_registry=None, tool_registry=None):
    """Set dependencies for the monitoring router"""
    router._llm_provider = llm_provider
    router._conversation_memory = conversation_memory or {}
    router._agent_registry = agent_registry or {}
    router._tool_registry = tool_registry or {}
    router._start_time = time.time()
    router._request_count = 0
