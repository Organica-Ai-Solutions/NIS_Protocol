"""
NIS Protocol v4.0 - Webhook Routes

This module contains webhook management endpoints:
- Register webhooks
- List webhooks
- Delete webhooks
- Webhook triggering utility

MIGRATION STATUS: Ready for testing
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("nis.routes.webhooks")

# Create router
router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

# Webhook storage (in production, use Redis or database)
webhooks: Dict[str, Dict[str, Any]] = {}


# ====== Request Models ======

class WebhookRegisterRequest(BaseModel):
    url: str
    events: List[str] = ["chat.completed", "agent.completed", "error"]
    secret: Optional[str] = None


# ====== Endpoints ======

@router.post("/register")
async def register_webhook(request: WebhookRegisterRequest):
    """
    🔔 Register Webhook
    
    Register a webhook URL to receive event notifications.
    
    Events:
    - chat.completed: When a chat response is generated
    - agent.completed: When an agent task completes
    - collaboration.completed: When multi-agent collaboration finishes
    - error: When an error occurs
    """
    webhook_id = str(uuid.uuid4())[:8]
    
    webhooks[webhook_id] = {
        "url": request.url,
        "events": request.events,
        "secret": request.secret,
        "created_at": time.time(),
        "calls_made": 0,
        "last_called": None
    }
    
    logger.info(f"🔔 Webhook registered: {webhook_id} -> {request.url}")
    
    return {
        "webhook_id": webhook_id,
        "url": request.url,
        "events": request.events,
        "status": "registered"
    }


@router.get("/list")
async def list_webhooks():
    """
    📋 List Webhooks
    
    Returns all registered webhooks with their status.
    """
    return {
        "webhooks": [
            {
                "id": wid, 
                "url": w["url"], 
                "events": w["events"], 
                "calls_made": w["calls_made"],
                "created_at": w["created_at"],
                "last_called": w["last_called"]
            }
            for wid, w in webhooks.items()
        ],
        "total": len(webhooks)
    }


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """
    🗑️ Delete Webhook
    
    Remove a registered webhook by ID.
    """
    if webhook_id in webhooks:
        del webhooks[webhook_id]
        logger.info(f"🗑️ Webhook deleted: {webhook_id}")
        return {"status": "deleted", "webhook_id": webhook_id}
    raise HTTPException(status_code=404, detail="Webhook not found")


# ====== Utility Functions ======

async def trigger_webhooks(event: str, data: dict):
    """
    Trigger all webhooks registered for an event.
    
    This function is called internally when events occur.
    """
    import aiohttp
    
    for webhook_id, webhook in webhooks.items():
        if event in webhook["events"]:
            try:
                payload = {
                    "event": event,
                    "data": data,
                    "timestamp": time.time(),
                    "webhook_id": webhook_id
                }
                
                headers = {"Content-Type": "application/json"}
                if webhook.get("secret"):
                    signature = hashlib.sha256(
                        f"{webhook['secret']}{json.dumps(payload)}".encode()
                    ).hexdigest()
                    headers["X-Webhook-Signature"] = signature
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook["url"],
                        json=payload,
                        headers=headers,
                        timeout=10
                    ) as response:
                        webhook["calls_made"] += 1
                        webhook["last_called"] = time.time()
                        
                        if response.status >= 400:
                            logger.warning(f"Webhook {webhook_id} returned {response.status}")
                        else:
                            logger.info(f"✅ Webhook {webhook_id} triggered: {event}")
                            
            except Exception as e:
                logger.error(f"❌ Webhook {webhook_id} failed: {e}")


def get_webhooks():
    """Get the webhooks dictionary for external access"""
    return webhooks
