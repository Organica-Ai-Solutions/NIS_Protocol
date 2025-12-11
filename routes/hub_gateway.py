"""
NIS Protocol Hub Gateway Routes

Provides WebSocket and REST API endpoints for NIS-HUB connections.
This enables the Cloud ←→ Hub communication layer.

Architecture:
    NIS Protocol (Cloud) ←→ NIS-HUB (Bridge) ←→ NeuroLinux (Edge)
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hub-gateway", tags=["Hub Gateway"])

# ============================================================
# Hub Connection Manager
# ============================================================

class HubConnectionManager:
    """Manages WebSocket connections from NIS-HUB instances."""
    
    def __init__(self):
        self.active_hubs: Dict[str, WebSocket] = {}
        self.hub_info: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, callable] = {}
        
    async def connect(self, websocket: WebSocket, hub_id: str, region: str = "default"):
        """Accept a new hub connection."""
        await websocket.accept()
        self.active_hubs[hub_id] = websocket
        self.hub_info[hub_id] = {
            "hub_id": hub_id,
            "region": region,
            "connected_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "fleet_summary": {}
        }
        logger.info(f"Hub connected: {hub_id} (region: {region})")
        
    def disconnect(self, hub_id: str):
        """Remove a hub connection."""
        if hub_id in self.active_hubs:
            del self.active_hubs[hub_id]
        if hub_id in self.hub_info:
            del self.hub_info[hub_id]
        logger.info(f"Hub disconnected: {hub_id}")
        
    async def send_to_hub(self, hub_id: str, message: Dict[str, Any]):
        """Send message to specific hub."""
        if hub_id in self.active_hubs:
            await self.active_hubs[hub_id].send_json(message)
            
    async def broadcast_to_all_hubs(self, message: Dict[str, Any]):
        """Broadcast message to all connected hubs."""
        for hub_id, websocket in self.active_hubs.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to hub {hub_id}: {e}")
                
    async def broadcast_to_region(self, region: str, message: Dict[str, Any]):
        """Broadcast message to all hubs in a region."""
        for hub_id, info in self.hub_info.items():
            if info.get("region") == region and hub_id in self.active_hubs:
                try:
                    await self.active_hubs[hub_id].send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to hub {hub_id}: {e}")
                    
    def get_hub_count(self) -> int:
        """Get number of connected hubs."""
        return len(self.active_hubs)
    
    def get_all_hubs(self) -> List[Dict[str, Any]]:
        """Get info for all connected hubs."""
        return list(self.hub_info.values())
    
    def get_hubs_by_region(self, region: str) -> List[Dict[str, Any]]:
        """Get hubs in a specific region."""
        return [info for info in self.hub_info.values() if info.get("region") == region]


# Global hub manager
hub_manager = HubConnectionManager()


# ============================================================
# Request/Response Models
# ============================================================

class TaskDispatchRequest(BaseModel):
    """Request to dispatch task to hub(s)."""
    target_hub: Optional[str] = Field(None, description="Specific hub ID or None for broadcast")
    target_region: Optional[str] = Field(None, description="Target region")
    target_device: Optional[str] = Field(None, description="Target device ID")
    target_product: Optional[str] = Field(None, description="Target product type")
    task_type: str = Field(..., description="Task type")
    payload: Dict[str, Any] = Field(..., description="Task payload")
    priority: str = Field("normal", description="Task priority")


class FleetCommandRequest(BaseModel):
    """Request to send fleet command."""
    target_hub: Optional[str] = Field(None, description="Specific hub or None for all")
    command: str = Field(..., description="Command name")
    target: str = Field(..., description="Target within hub")
    params: Optional[Dict[str, Any]] = Field(None, description="Command parameters")


class BroadcastRequest(BaseModel):
    """Request to broadcast message to hubs."""
    target_region: Optional[str] = Field(None, description="Target region or None for all")
    message: Dict[str, Any] = Field(..., description="Message to broadcast")
    priority: str = Field("normal", description="Message priority")


# ============================================================
# WebSocket Endpoint for Hub Connections
# ============================================================

@router.websocket("/ws/hub")
async def hub_websocket(
    websocket: WebSocket,
    hub_id: str = Query(..., description="Hub identifier"),
    region: str = Query("default", description="Hub region")
):
    """
    WebSocket endpoint for NIS-HUB connections.
    
    Handles:
    - Hub registration and heartbeats
    - Fleet status updates
    - Task dispatch from cloud
    - Device events aggregation
    """
    await hub_manager.connect(websocket, hub_id, region)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "data": {
                "message": "Connected to NIS Protocol Cloud",
                "hub_id": hub_id,
                "cloud_version": "4.0.2",
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        # Message loop
        while True:
            try:
                data = await websocket.receive_json()
                await handle_hub_message(hub_id, data)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"}
                })
                
    except Exception as e:
        logger.error(f"Hub WebSocket error: {e}")
    finally:
        hub_manager.disconnect(hub_id)


async def handle_hub_message(hub_id: str, message: Dict[str, Any]):
    """Handle incoming message from hub."""
    msg_type = message.get("type")
    data = message.get("data", {})
    
    logger.debug(f"Message from hub {hub_id}: {msg_type}")
    
    if msg_type == "hub_registration":
        # Hub registration
        hub_manager.hub_info[hub_id].update({
            "capabilities": data.get("capabilities", {}),
            "version": data.get("version", "unknown")
        })
        logger.info(f"Hub registered: {hub_id} v{data.get('version')}")
        
    elif msg_type == "hub_heartbeat":
        # Hub heartbeat
        hub_manager.hub_info[hub_id]["last_heartbeat"] = datetime.utcnow().isoformat()
        hub_manager.hub_info[hub_id]["fleet_summary"] = data.get("fleet_summary", {})
        
    elif msg_type == "fleet_status":
        # Fleet status update
        hub_manager.hub_info[hub_id]["fleet"] = data.get("fleet", {})
        logger.debug(f"Fleet status from {hub_id}")
        
    elif msg_type == "device_registered":
        # New device registered at hub
        device = data.get("device", {})
        logger.info(f"Device registered via {hub_id}: {device.get('device_id')}")
        
    elif msg_type == "device_offline":
        # Device went offline
        device_id = data.get("device_id")
        logger.info(f"Device offline via {hub_id}: {device_id}")
        
    elif msg_type == "task_progress":
        # Task progress update
        task_id = data.get("task_id")
        status = data.get("status")
        logger.debug(f"Task {task_id} progress: {status}")
        
    elif msg_type == "task_completed":
        # Task completed
        task_id = data.get("task_id")
        result = data.get("result", {})
        logger.info(f"Task completed: {task_id}")
        
    elif msg_type == "telemetry_batch":
        # Telemetry batch from hub
        count = data.get("count", 0)
        logger.debug(f"Telemetry batch from {hub_id}: {count} points")
        
    elif msg_type == "alert":
        # Alert from hub
        alert_type = data.get("alert_type")
        severity = data.get("severity")
        logger.warning(f"Alert from {hub_id}: {alert_type} ({severity})")


# ============================================================
# REST API Endpoints
# ============================================================

@router.get("/status")
async def get_gateway_status() -> Dict[str, Any]:
    """Get hub gateway status."""
    return {
        "success": True,
        "connected_hubs": hub_manager.get_hub_count(),
        "hubs": hub_manager.get_all_hubs(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/hubs")
async def get_connected_hubs() -> Dict[str, Any]:
    """Get all connected hubs."""
    return {
        "success": True,
        "hubs": hub_manager.get_all_hubs(),
        "total": hub_manager.get_hub_count()
    }


@router.get("/hubs/{hub_id}")
async def get_hub_info(hub_id: str) -> Dict[str, Any]:
    """Get specific hub information."""
    if hub_id not in hub_manager.hub_info:
        raise HTTPException(status_code=404, detail="Hub not found")
    
    return {
        "success": True,
        "hub": hub_manager.hub_info[hub_id]
    }


@router.get("/regions/{region}/hubs")
async def get_hubs_by_region(region: str) -> Dict[str, Any]:
    """Get hubs in a specific region."""
    hubs = hub_manager.get_hubs_by_region(region)
    return {
        "success": True,
        "region": region,
        "hubs": hubs,
        "total": len(hubs)
    }


@router.post("/task/dispatch")
async def dispatch_task(request: TaskDispatchRequest) -> Dict[str, Any]:
    """
    Dispatch task to hub(s) for execution on edge devices.
    
    Routes task based on:
    - target_hub: Specific hub
    - target_region: All hubs in region
    - target_device: Specific device (routed through appropriate hub)
    - target_product: All devices of product type
    """
    import uuid
    task_id = f"cloud_task_{uuid.uuid4().hex[:12]}"
    
    message = {
        "type": "task_dispatch",
        "data": {
            "task_id": task_id,
            "target": request.target_device or request.target_product or "broadcast",
            "payload": {
                "task_type": request.task_type,
                **request.payload
            },
            "priority": request.priority,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    hubs_reached = 0
    
    if request.target_hub:
        # Send to specific hub
        if request.target_hub in hub_manager.active_hubs:
            await hub_manager.send_to_hub(request.target_hub, message)
            hubs_reached = 1
    elif request.target_region:
        # Send to region
        hubs = hub_manager.get_hubs_by_region(request.target_region)
        for hub in hubs:
            await hub_manager.send_to_hub(hub["hub_id"], message)
            hubs_reached += 1
    else:
        # Broadcast to all
        await hub_manager.broadcast_to_all_hubs(message)
        hubs_reached = hub_manager.get_hub_count()
    
    return {
        "success": True,
        "task_id": task_id,
        "hubs_reached": hubs_reached,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/command")
async def send_fleet_command(request: FleetCommandRequest) -> Dict[str, Any]:
    """Send command to fleet through hub(s)."""
    message = {
        "type": "fleet_command",
        "data": {
            "command": request.command,
            "target": request.target,
            "params": request.params or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if request.target_hub:
        if request.target_hub not in hub_manager.active_hubs:
            raise HTTPException(status_code=404, detail="Hub not found")
        await hub_manager.send_to_hub(request.target_hub, message)
        return {"success": True, "hub": request.target_hub}
    else:
        await hub_manager.broadcast_to_all_hubs(message)
        return {"success": True, "hubs_reached": hub_manager.get_hub_count()}


@router.post("/broadcast")
async def broadcast_to_hubs(request: BroadcastRequest) -> Dict[str, Any]:
    """Broadcast message to hubs."""
    message = {
        "type": "broadcast",
        "data": {
            "message": request.message,
            "priority": request.priority,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if request.target_region:
        await hub_manager.broadcast_to_region(request.target_region, message)
        hubs = hub_manager.get_hubs_by_region(request.target_region)
        return {"success": True, "region": request.target_region, "hubs_reached": len(hubs)}
    else:
        await hub_manager.broadcast_to_all_hubs(message)
        return {"success": True, "hubs_reached": hub_manager.get_hub_count()}


@router.post("/config/push")
async def push_config_to_hubs(
    config_type: str,
    config: Dict[str, Any],
    target_hub: Optional[str] = None
) -> Dict[str, Any]:
    """Push configuration update to hub(s)."""
    message = {
        "type": "config_update",
        "data": {
            "config_type": config_type,
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if target_hub:
        if target_hub not in hub_manager.active_hubs:
            raise HTTPException(status_code=404, detail="Hub not found")
        await hub_manager.send_to_hub(target_hub, message)
        return {"success": True, "hub": target_hub}
    else:
        await hub_manager.broadcast_to_all_hubs(message)
        return {"success": True, "hubs_reached": hub_manager.get_hub_count()}


@router.post("/ota/trigger")
async def trigger_ota_update(
    device_ids: List[str],
    version: str,
    target_hub: Optional[str] = None
) -> Dict[str, Any]:
    """Trigger OTA update for devices through hub(s)."""
    message = {
        "type": "ota_trigger",
        "data": {
            "device_ids": device_ids,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if target_hub:
        if target_hub not in hub_manager.active_hubs:
            raise HTTPException(status_code=404, detail="Hub not found")
        await hub_manager.send_to_hub(target_hub, message)
        return {"success": True, "hub": target_hub, "devices": len(device_ids)}
    else:
        await hub_manager.broadcast_to_all_hubs(message)
        return {"success": True, "hubs_reached": hub_manager.get_hub_count(), "devices": len(device_ids)}


# ============================================================
# Fleet Aggregation Endpoints
# ============================================================

@router.get("/fleet/summary")
async def get_aggregated_fleet_summary() -> Dict[str, Any]:
    """Get aggregated fleet summary across all hubs."""
    total_devices = 0
    by_product = {}
    by_region = {}
    
    for hub_id, info in hub_manager.hub_info.items():
        fleet = info.get("fleet_summary", {})
        region = info.get("region", "unknown")
        
        # Aggregate device counts
        hub_devices = fleet.get("total_devices", 0)
        total_devices += hub_devices
        
        # By region
        if region not in by_region:
            by_region[region] = {"hubs": 0, "devices": 0}
        by_region[region]["hubs"] += 1
        by_region[region]["devices"] += hub_devices
        
        # By product type
        for product, count in fleet.get("by_product_type", {}).items():
            by_product[product] = by_product.get(product, 0) + count
    
    return {
        "success": True,
        "summary": {
            "total_hubs": hub_manager.get_hub_count(),
            "total_devices": total_devices,
            "by_region": by_region,
            "by_product_type": by_product
        },
        "timestamp": datetime.utcnow().isoformat()
    }
