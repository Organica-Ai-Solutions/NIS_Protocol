# PROTOTYPE — Health Check Fallback System
# This is a prototype for when plugin commands are unavailable
# Not for production use

from fastapi import APIRouter, HTTPException
import httpx
import asyncio
from datetime import datetime
import pytz

router = APIRouter(prefix="/prototype/health", tags=["prototype"])

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

def is_quiet_hours():
    """Check if current time is within quiet hours (23:00–08:00 ET)"""
    eastern = pytz.timezone('America/New_York')
    now_et = datetime.now(eastern)
    hour = now_et.hour
    return hour >= 23 or hour < 8

@router.get("/check-all")
async def check_all_services():
    """Check all services - PROTOTYPE endpoint"""
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
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "quiet_hours": quiet_hours,
        "services": results,
        "summary": {
            "total": len(results),
            "online": len([r for r in results if r["status"] == "online"]),
            "offline": len(offline_services)
        },
        "prototype_note": "This is a prototype fallback system. Use plugin commands when available."
    }
    
    # If not quiet hours and services are offline, include alert message
    if not quiet_hours and offline_services:
        offline_names = [s["service"] for s in offline_services]
        response["alert"] = {
            "message": f"{', '.join(offline_names)} is offline — restart needed.",
            "services": offline_names
        }
    
    return response

@router.get("/check/{service_name}")
async def check_specific_service(service_name: str):
    """Check a specific service"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not configured")
    
    result = await check_service(service_name, SERVICES[service_name])
    quiet_hours = is_quiet_hours()
    
    response = {
        "service": result,
        "quiet_hours": quiet_hours,
        "should_alert": not quiet_hours and result["status"] == "offline"
    }
    
    if response["should_alert"]:
        response["alert_message"] = f"{service_name} is offline — restart needed."
    
    return response

@router.get("/quiet-hours")
async def check_quiet_hours():
    """Check if currently in quiet hours"""
    return {
        "quiet_hours": is_quiet_hours(),
        "current_time_et": datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "quiet_hours_range": "23:00–08:00 ET"
    }