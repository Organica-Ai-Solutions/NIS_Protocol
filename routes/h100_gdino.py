"""
NIS Protocol v4.0 - Grounding DINO Proxy Route
================================================
Proxies open-vocabulary detection requests to the H100 GDINO server at port 8400.
Grounding DINO understands text prompts like "lighter . bin ." — no COCO class needed.

Endpoints:
  GET  /gdino/status   — GDINO server health (ping H100:8400)
  POST /gdino/detect   — Proxy detect call: image_base64 + text_prompts → detections
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.h100_gdino")

router = APIRouter(prefix="/gdino", tags=["Grounding DINO"])

H100_GDINO_URL = os.environ.get("H100_GDINO_URL", "http://172.16.1.83:8400")

_gdino_last_ok   = False
_gdino_last_ping = 0.0
_PING_TTL        = 30.0   # re-ping at most every 30 s


async def _ping_gdino() -> bool:
    """Check if GDINO server is alive. Cached for _PING_TTL seconds."""
    global _gdino_last_ok, _gdino_last_ping
    if time.time() - _gdino_last_ping < _PING_TTL:
        return _gdino_last_ok
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=4.0, write=2.0, pool=2.0)) as c:
            r = await c.get(f"{H100_GDINO_URL}/health")
            _gdino_last_ok = r.status_code == 200
    except Exception:
        _gdino_last_ok = False
    _gdino_last_ping = time.time()
    return _gdino_last_ok


class DetectRequest(BaseModel):
    image_base64: str
    text_prompts: str  = Field(default="lighter . bin .",
                               description="Dot-separated noun phrases, e.g. 'lighter . bin . cup .'")
    box_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


@router.get("/status")
async def gdino_status():
    """Ping H100:8400 GDINO server and report availability."""
    alive = await _ping_gdino()
    return {
        "gdino_available": alive,
        "h100_gdino_url":  H100_GDINO_URL,
        "last_ping_ts":    _gdino_last_ping,
    }


@router.post("/detect")
async def gdino_detect(req: DetectRequest):
    """
    Proxy image + text prompt to H100 Grounding DINO server.
    Returns open-vocabulary detections — any noun phrase you specify.
    """
    t0 = time.time()
    alive = await _ping_gdino()
    if not alive:
        return {
            "ok":        False,
            "error":     f"GDINO server unreachable at {H100_GDINO_URL}",
            "detections": [],
            "n":         0,
        }

    try:
        payload: Dict[str, Any] = {
            "image_base64":  req.image_base64,
            "text_prompts":  req.text_prompts,
            "box_threshold": req.box_threshold,
            "text_threshold": req.text_threshold,
        }
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=3.0, pool=3.0)
        ) as c:
            r = await c.post(f"{H100_GDINO_URL}/detect", json=payload)
            if r.status_code == 200:
                data: Dict[str, Any] = r.json()
                return {
                    "ok":         True,
                    "detections": data.get("detections", []),
                    "n":          data.get("n", 0),
                    "latency_ms": round((time.time() - t0) * 1000),
                    "source":     "h100_gdino",
                }
            return {
                "ok":    False,
                "error": f"GDINO returned HTTP {r.status_code}",
                "detections": [],
                "n":     0,
            }
    except Exception as e:
        logger.warning("GDINO proxy error: %s", e)
        return {"ok": False, "error": str(e), "detections": [], "n": 0}
