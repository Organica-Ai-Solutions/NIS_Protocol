"""
NIS Protocol — Server-Sent Events (SSE) Channel Adapter
=========================================================
Third channel after HTTP REST and WebSocket.

Provides a persistent streaming connection for:
- Drive execution results (arm_watchdog, cosmos_heartbeat, …)
- AuditChain entries as they are appended
- System health state changes
- NeuroKernel DIKW layer updates

Usage:
    curl -N http://localhost:8000/events/stream
    curl -N "http://localhost:8000/events/stream?topics=drives,audit"

Clients receive newline-delimited SSE:
    data: {"topic": "drives", "drive_id": "arm_watchdog", "success": true, ...}

    data: {"topic": "audit", "agent_id": "chat/groq", "action_type": "llm_call", ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Set

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("nis.events")

router = APIRouter(prefix="/events", tags=["Events / SSE"])

# ── Event bus ─────────────────────────────────────────────────────────────────
# All active SSE subscribers keyed by a unique connection id.
# Each value is an asyncio.Queue that receives raw event dicts.

_subscribers: Dict[str, asyncio.Queue] = {}

VALID_TOPICS = {"drives", "audit", "health", "dikw", "arm", "cosmos", "cookoff"}


def publish(topic: str, payload: dict) -> None:
    """
    Publish an event to all SSE subscribers interested in *topic*.
    Non-blocking — drops events for slow/dead connections.
    Called from DriveScheduler._run_drive(), AuditChain.log(), etc.
    """
    if not _subscribers:
        return
    msg = {"topic": topic, "ts": time.time(), **payload}
    for q in list(_subscribers.values()):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # slow subscriber — drop rather than block


# ── SSE stream endpoint ───────────────────────────────────────────────────────

@router.get(
    "/stream",
    summary="SSE event stream",
    response_description="text/event-stream — one JSON object per line",
)
async def sse_stream(
    request: Request,
    topics: Optional[str] = Query(
        None,
        description="Comma-separated topics to subscribe to. "
                    "Omit for all. Valid: drives, audit, health, dikw, arm, cosmos",
    ),
    heartbeat: int = Query(15, ge=5, le=60, description="Heartbeat interval in seconds"),
):
    """
    Server-Sent Events stream.

    Streams NIS Protocol events as they happen:
    - **drives** — each Drive execution result (arm_watchdog, cosmos_heartbeat …)
    - **audit**  — each AuditChain entry appended
    - **health** — system health snapshots from system_report drive
    - **dikw**   — NeuroKernel DIKW layer transitions
    - **arm**    — xArm movement events (pick, place, home, dance)
    - **cosmos** — Cosmos Reason2 inference results
    """
    import uuid
    conn_id = uuid.uuid4().hex[:8]
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers[conn_id] = q
    logger.info(f"SSE client connected: {conn_id} topics={topics!r}")

    # Parse topic filter
    topic_filter: Optional[Set[str]] = None
    if topics:
        requested = {t.strip().lower() for t in topics.split(",") if t.strip()}
        topic_filter = requested & VALID_TOPICS
        if not topic_filter:
            topic_filter = None  # invalid topics → all

    async def _generate() -> AsyncIterator[str]:
        # Send connection confirmation
        yield _fmt({"topic": "system", "event": "connected",
                     "conn_id": conn_id, "topics": list(topic_filter or VALID_TOPICS)})
        last_hb = time.time()
        try:
            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    if topic_filter and msg.get("topic") not in topic_filter:
                        continue
                    yield _fmt(msg)
                except asyncio.TimeoutError:
                    pass

                # Heartbeat
                if time.time() - last_hb >= heartbeat:
                    yield _fmt({"topic": "heartbeat", "ts": time.time()})
                    last_hb = time.time()
        finally:
            _subscribers.pop(conn_id, None)
            logger.info(f"SSE client disconnected: {conn_id}")

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/topics", summary="List valid SSE topics")
async def list_topics():
    return {
        "topics": sorted(VALID_TOPICS),
        "active_connections": len(_subscribers),
        "usage": "GET /events/stream?topics=drives,audit",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"
