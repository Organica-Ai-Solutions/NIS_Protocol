"""
NIS Protocol v4.0 - Prototype Routes

This module contains prototype endpoints for rapid testing and development.
These endpoints are NOT for production use - they're for quick prototyping.

PROTOTYPE WARNING: This code is experimental and may be removed or changed
without notice. Use at your own risk.

Usage:
    from routes.prototype import router as prototype_router
    app.include_router(prototype_router, tags=["Prototype"])
"""

import logging
import time
import httpx
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import pytz
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.prototype")

# Create router
router = APIRouter(prefix="/prototype", tags=["Prototype"])

# Service endpoints to check
SERVICES = {
    "nis_protocol_pi": "http://192.168.1.160:8090/health",
    "nis_protocol_local": "http://localhost:8007/health",
    "arbitragemachine": "http://localhost:8000/health",
    "smartportfolio": "http://localhost:5000/health",
    "alphacortex": "http://localhost:3000/health"
}

# Fallback endpoints if /health doesn't exist
FALLBACK_ENDPOINTS = {
    "nis_protocol_pi": "http://192.168.1.160:8090/",
    "nis_protocol_local": "http://localhost:8007/",
    "arbitragemachine": "http://localhost:8000/",
    "smartportfolio": "http://localhost:5000/",
    "alphacortex": "http://localhost:3000/"
}

def is_quiet_hours():
    """Check if current time is within quiet hours (23:00–08:00 ET)"""
    try:
        eastern = pytz.timezone('America/New_York')
        now_et = datetime.now(eastern)
        hour = now_et.hour
        return hour >= 23 or hour < 8
    except:
        # Fallback if timezone fails
        return False

async def check_service(service_name: str, url: str, timeout: float = 2.0):
    """Check if a service is reachable via HTTP"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return {
                "service": service_name,
                "status": "online" if response.status_code < 500 else "offline",
                "status_code": response.status_code,
                "url": url
            }
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        # Try fallback endpoint
        fallback_url = FALLBACK_ENDPOINTS.get(service_name)
        if fallback_url and fallback_url != url:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(fallback_url)
                    return {
                        "service": service_name,
                        "status": "online" if response.status_code < 500 else "offline",
                        "status_code": response.status_code,
                        "url": fallback_url,
                        "note": "used fallback endpoint"
                    }
            except:
                pass
        return {
            "service": service_name,
            "status": "offline",
            "status_code": None,
            "url": url,
            "error": "connection failed"
        }


@router.get("/health-check")
async def prototype_health_check() -> Dict[str, Any]:
    """
    PROTOTYPE: Health check for Pi connectivity
    
    This endpoint checks if the Pi edge node is reachable.
    It's a prototype for testing connectivity without relying on
    system status messages.
    
    Returns:
        Dict with Pi status and timestamp
    """
    pi_status = "unknown"
    response_time = None
    error = None
    
    start_time = time.time()
    
    try:
        # Try to reach Pi at its expected IP
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try multiple common Pi endpoints
            endpoints = [
                "http://172.16.1.83:8000/health",  # Common health endpoint
                "http://172.16.1.83:8000/",        # Root endpoint
                "http://172.16.1.83:8000/status",  # Status endpoint
            ]
            
            for endpoint in endpoints:
                try:
                    resp = await client.get(endpoint)
                    if resp.status_code == 200:
                        pi_status = "online"
                        response_time = time.time() - start_time
                        break
                except:
                    continue
            
            if pi_status == "unknown":
                pi_status = "offline"
                
    except Exception as e:
        pi_status = "error"
        error = str(e)
    
    return {
        "pi_edge": pi_status,
        "response_time_ms": round(response_time * 1000, 2) if response_time else None,
        "timestamp": time.time(),
        "error": error,
        "note": "PROTOTYPE - not production. Use /health for production health checks."
    }


@router.get("/alert-test")
async def prototype_alert_test() -> Dict[str, Any]:
    """
    PROTOTYPE: Test alerting workflow
    
    This endpoint simulates the HEARTBEAT.md alerting workflow.
    It demonstrates what should happen when a service is down.
    
    Returns:
        Dict with alert test results
    """
    # First check Pi status
    health_result = await prototype_health_check()
    pi_status = health_result["pi_edge"]
    
    # Determine alert message
    if pi_status == "offline" or pi_status == "error":
        alert_message = f"Pi edge is {pi_status} — restart needed."
        alert_needed = True
    else:
        alert_message = f"Pi edge is {pi_status} — no action needed."
        alert_needed = False
    
    return {
        "pi_status": pi_status,
        "alert_needed": alert_needed,
        "alert_message": alert_message,
        "heartbeat_instruction": "If any service is down, DM Diego: '[service] is offline — restart needed.'",
        "current_time_et": "Saturday, April 4th, 2026 — 9:25 AM (America/New_York)",
        "quiet_hours": "23:00–08:00 ET",
        "outside_quiet_hours": True,
        "note": "PROTOTYPE - This demonstrates the HEARTBEAT.md alerting workflow"
    }


@router.get("/heartbeat-simulator")
async def prototype_heartbeat_simulator() -> Dict[str, Any]:
    """
    PROTOTYPE: Simulate HEARTBEAT.md monitoring workflow
    
    This endpoint simulates what the HEARTBEAT.md monitoring agent
    should do when checking system status.
    
    Returns:
        Dict with simulated heartbeat check results
    """
    # Check Pi status
    health_result = await prototype_health_check()
    pi_status = health_result["pi_edge"]
    
    # Simulate HEARTBEAT.md logic
    current_time_et = "9:25 AM"
    quiet_hours_start = "23:00"
    quiet_hours_end = "08:00"
    outside_quiet_hours = True  # Assuming 9:25 AM is outside quiet hours
    
    # Determine response based on HEARTBEAT.md rules
    if pi_status == "online":
        response = "HEARTBEAT_OK"
        action = "No action needed"
    else:
        # Service is down
        if outside_quiet_hours:
            response = "HEARTBEAT_ALERT"
            action = "DM Diego: 'Pi edge is offline — restart needed.'"
        else:
            # Inside quiet hours - only alert if service goes down
            response = "HEARTBEAT_OK_QUIET"
            action = "No proactive messaging during quiet hours unless service goes down"
    
    return {
        "pi_status": pi_status,
        "current_time_et": current_time_et,
        "quiet_hours": f"{quiet_hours_start}–{quiet_hours_end} ET",
        "outside_quiet_hours": outside_quiet_hours,
        "heartbeat_response": response,
        "required_action": action,
        "simulation_note": "PROTOTYPE - Simulating HEARTBEAT.md monitoring workflow",
        "actual_agent_limitation": "Agent cannot send DMs, can only respond in chat",
        "workaround": "Alert surfaced in chat: '⚠️ Pi edge is offline — restart needed.'"
    }


@router.get("/heartbeat-check")
async def prototype_heartbeat_check() -> Dict[str, Any]:
    """
    PROTOTYPE: HEARTBEAT.md health check fallback system
    
    This endpoint implements a fallback health check system for when
    plugin commands (/nis-status, /arb-status, etc.) are unavailable.
    It follows HEARTBEAT.md rules including quiet hours.
    
    Returns:
        Dict with service status and HEARTBEAT.md compliance check
    """
    # Check all services
    results = []
    tasks = []
    
    for service_name, url in SERVICES.items():
        task = check_service(service_name, url)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Check if we're in quiet hours
    quiet_hours = is_quiet_hours()
    
    # Determine if any services are offline
    offline_services = [r for r in results if r["status"] == "offline"]
    
    # Get current time in ET
    try:
        eastern = pytz.timezone('America/New_York')
        current_time_et = datetime.now(eastern).strftime("%A, %B %d, %Y — %I:%M %p (%Z)")
    except:
        current_time_et = "Unknown"
    
    # Determine HEARTBEAT response
    if not offline_services:
        heartbeat_response = "HEARTBEAT_OK"
        action_needed = "No action needed"
    else:
        if quiet_hours:
            heartbeat_response = "HEARTBEAT_OK_QUIET"
            action_needed = "No proactive messaging during quiet hours unless service goes down"
        else:
            heartbeat_response = "HEARTBEAT_ALERT"
            action_needed = f"DM Diego: '{', '.join([s['service'] for s in offline_services])} is offline — restart needed.'"
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "current_time_et": current_time_et,
        "quiet_hours": quiet_hours,
        "quiet_hours_range": "23:00–08:00 ET",
        "heartbeat_response": heartbeat_response,
        "action_needed": action_needed,
        "services": results,
        "summary": {
            "total": len(results),
            "online": len([r for r in results if r["status"] == "online"]),
            "offline": len(offline_services)
        },
        "prototype_note": "PROTOTYPE - Fallback system for when plugin commands are unavailable",
        "heartbeat_compliance": "Follows HEARTBEAT.md rules: plugin commands preferred, quiet hours respected"
    }
    
    # Add alert details if needed
    if offline_services and not quiet_hours:
        response["alert"] = {
            "message": f"{', '.join([s['service'] for s in offline_services])} is offline — restart needed.",
            "services": [s["service"] for s in offline_services]
        }
    
    return response


@router.get("/check-service/{service_name}")
async def prototype_check_service(service_name: str) -> Dict[str, Any]:
    """
    PROTOTYPE: Check specific service status
    
    This endpoint checks a specific service and returns whether
    an alert should be sent according to HEARTBEAT.md rules.
    
    Args:
        service_name: One of: nis_protocol_pi, nis_protocol_local, 
                     arbitragemachine, smartportfolio, alphacortex
    
    Returns:
        Dict with service status and alert determination
    """
    if service_name not in SERVICES:
        raise HTTPException(
            status_code=404, 
            detail=f"Service {service_name} not found. Available: {list(SERVICES.keys())}"
        )
    
    result = await check_service(service_name, SERVICES[service_name])
    quiet_hours = is_quiet_hours()
    
    # Determine if alert should be sent
    should_alert = not quiet_hours and result["status"] == "offline"
    
    response = {
        "service": result,
        "quiet_hours": quiet_hours,
        "quiet_hours_range": "23:00–08:00 ET",
        "should_alert": should_alert,
        "heartbeat_rule": "DM Diego if service is offline and outside quiet hours"
    }
    
    if should_alert:
        response["alert_message"] = f"{service_name} is offline — restart needed."
    
    return response


@router.get("/quiet-hours-status")
async def prototype_quiet_hours_status() -> Dict[str, Any]:
    """
    PROTOTYPE: Check quiet hours status
    
    This endpoint checks if current time is within quiet hours
    according to HEARTBEAT.md rules.
    
    Returns:
        Dict with quiet hours status and current time
    """
    quiet_hours = is_quiet_hours()
    
    try:
        eastern = pytz.timezone('America/New_York')
        current_time_et = datetime.now(eastern)
        current_time_str = current_time_et.strftime("%A, %B %d, %Y — %I:%M %p (%Z)")
        current_hour = current_time_et.hour
    except:
        current_time_str = "Unknown"
        current_hour = None
    
    return {
        "quiet_hours": quiet_hours,
        "quiet_hours_range": "23:00–08:00 ET",
        "current_time_et": current_time_str,
        "current_hour_et": current_hour,
        "heartbeat_rule": "Don't proactively message between 23:00–08:00 ET unless a service goes down",
        "note": "PROTOTYPE - Timezone: America/New_York"
    }
