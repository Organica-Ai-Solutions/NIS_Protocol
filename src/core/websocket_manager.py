#!/usr/bin/env python3
"""
NIS Protocol WebSocket Management System
Real-time communication between backend and frontend

This system enables real-time state synchronization and automatic
frontend updates based on backend state changes.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum

from .state_manager import nis_state_manager, StateEventType, StateEvent

logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    """Types of WebSocket connections"""
    DASHBOARD = "dashboard"
    CHAT = "chat"
    ADMIN = "admin"
    MONITORING = "monitoring"
    AGENT = "agent"

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection"""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str]
    session_id: Optional[str]
    connection_type: ConnectionType
    connected_at: float
    last_ping: float
    subscribed_events: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "connection_type": self.connection_type.value,
            "connected_at": self.connected_at,
            "last_ping": self.last_ping,
            "subscribed_events": list(self.subscribed_events)
        }

class NISWebSocketManager:
    """
    🔌 NIS Protocol WebSocket Manager
    
    Manages real-time connections between backend and frontend:
    - Connection lifecycle management
    - Message routing and broadcasting
    - Event subscription management
    - Connection health monitoring
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        
        # Performance metrics
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "broadcasts_sent": 0
        }
        
        # Health monitoring
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10   # seconds
        
        logger.info("🔌 NIS WebSocket Manager initialized")
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_type: ConnectionType = ConnectionType.DASHBOARD,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Accept and register a new WebSocket connection
        
        Returns:
            connection_id: Unique identifier for the connection
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        current_time = time.time()
        
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            connection_type=connection_type,
            connected_at=current_time,
            last_ping=current_time,
            subscribed_events=set()
        )
        
        # Register connection
        self.connections[connection_id] = connection
        
        # Index by user and session
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(connection_id)
        
        # Update metrics
        self.metrics["total_connections"] += 1
        self.metrics["active_connections"] = len(self.connections)
        
        # Register with state manager
        nis_state_manager.add_websocket_connection(connection_id, websocket)
        
        # Send initial state
        await self.send_initial_state(connection_id)
        
        # Emit connection event
        await nis_state_manager.emit_event(
            StateEventType.SYSTEM_STATUS_CHANGE,
            {
                "event": "websocket_connected",
                "connection_id": connection_id,
                "connection_type": connection_type.value,
                "user_id": user_id,
                "active_connections": len(self.connections)
            }
        )
        
        logger.info(f"WebSocket connected: {connection_id} ({connection_type.value})")
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> None:
        """Disconnect and cleanup a WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from indexes
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        if connection.session_id and connection.session_id in self.session_connections:
            self.session_connections[connection.session_id].discard(connection_id)
            if not self.session_connections[connection.session_id]:
                del self.session_connections[connection.session_id]
        
        # Remove from state manager
        nis_state_manager.remove_websocket_connection(connection_id)
        
        # Remove connection
        del self.connections[connection_id]
        
        # Update metrics
        self.metrics["active_connections"] = len(self.connections)
        
        # Emit disconnection event
        await nis_state_manager.emit_event(
            StateEventType.SYSTEM_STATUS_CHANGE,
            {
                "event": "websocket_disconnected",
                "connection_id": connection_id,
                "active_connections": len(self.connections)
            }
        )
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific connection
        
        Returns:
            success: Whether message was sent successfully
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            self.metrics["messages_sent"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.metrics["connection_errors"] += 1
            # Remove dead connection
            await self.disconnect(connection_id)
            return False
    
    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any]
    ) -> int:
        """
        Send message to all connections for a user
        
        Returns:
            count: Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            return 0
        
        connection_ids = list(self.user_connections[user_id])
        sent_count = 0
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def send_to_session(
        self,
        session_id: str,
        message: Dict[str, Any]
    ) -> int:
        """
        Send message to all connections for a session
        
        Returns:
            count: Number of connections message was sent to
        """
        if session_id not in self.session_connections:
            return 0
        
        connection_ids = list(self.session_connections[session_id])
        sent_count = 0
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        connection_type: Optional[ConnectionType] = None,
        exclude_connections: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast message to all or filtered connections
        
        Args:
            message: Message to broadcast
            connection_type: Filter by connection type
            exclude_connections: Set of connection IDs to exclude
            
        Returns:
            count: Number of connections message was sent to
        """
        if exclude_connections is None:
            exclude_connections = set()
        
        sent_count = 0
        
        for connection_id, connection in self.connections.items():
            if connection_id in exclude_connections:
                continue
            
            if connection_type and connection.connection_type != connection_type:
                continue
            
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        self.metrics["broadcasts_sent"] += 1
        
        return sent_count
    
    async def send_initial_state(self, connection_id: str) -> None:
        """Send initial system state to a new connection"""
        current_state = nis_state_manager.get_state_dict()
        
        initial_message = {
            "type": "initial_state",
            "data": current_state,
            "timestamp": time.time(),
            "connection_id": connection_id
        }
        
        await self.send_message(connection_id, initial_message)
    
    async def handle_client_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Handle incoming message from client"""
        self.metrics["messages_received"] += 1
        
        message_type = message.get("type")
        
        if message_type == "ping":
            await self.handle_ping(connection_id)
        elif message_type == "subscribe":
            await self.handle_subscribe(connection_id, message.get("events", []))
        elif message_type == "unsubscribe":
            await self.handle_unsubscribe(connection_id, message.get("events", []))
        elif message_type == "request_state":
            await self.send_initial_state(connection_id)
        else:
            logger.warning(f"Unknown message type from {connection_id}: {message_type}")
    
    async def handle_ping(self, connection_id: str) -> None:
        """Handle ping message"""
        if connection_id in self.connections:
            self.connections[connection_id].last_ping = time.time()
            await self.send_message(connection_id, {"type": "pong", "timestamp": time.time()})
    
    async def handle_subscribe(self, connection_id: str, events: List[str]) -> None:
        """Handle event subscription"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.subscribed_events.update(events)
        
        await self.send_message(connection_id, {
            "type": "subscription_confirmed",
            "events": events,
            "timestamp": time.time()
        })
    
    async def handle_unsubscribe(self, connection_id: str, events: List[str]) -> None:
        """Handle event unsubscription"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.subscribed_events.difference_update(events)
        
        await self.send_message(connection_id, {
            "type": "unsubscription_confirmed",
            "events": events,
            "timestamp": time.time()
        })
    
    async def start_health_monitoring(self) -> None:
        """Start connection health monitoring"""
        while True:
            await asyncio.sleep(self.ping_interval)
            await self.check_connection_health()
    
    async def check_connection_health(self) -> None:
        """Check health of all connections"""
        current_time = time.time()
        dead_connections = []
        
        for connection_id, connection in self.connections.items():
            if current_time - connection.last_ping > self.ping_timeout:
                dead_connections.append(connection_id)
        
        # Remove dead connections
        for connection_id in dead_connections:
            await self.disconnect(connection_id)
            logger.warning(f"Removed dead connection: {connection_id}")
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection"""
        if connection_id not in self.connections:
            return None
        
        return self.connections[connection_id].to_dict()
    
    def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get information about all connections"""
        return [conn.to_dict() for conn in self.connections.values()]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket manager metrics"""
        return {
            **self.metrics,
            "connections_by_type": {
                conn_type.value: sum(1 for conn in self.connections.values() 
                                   if conn.connection_type == conn_type)
                for conn_type in ConnectionType
            },
            "users_connected": len(self.user_connections),
            "sessions_active": len(self.session_connections)
        }

# Global WebSocket manager instance
nis_websocket_manager = NISWebSocketManager()

# Convenience functions
async def broadcast_to_all(message: Dict[str, Any]) -> int:
    """Broadcast message to all connections"""
    return await nis_websocket_manager.broadcast(message)

async def send_to_user(user_id: str, message: Dict[str, Any]) -> int:
    """Send message to all user connections"""
    return await nis_websocket_manager.send_to_user(user_id, message)

async def send_to_connection(connection_id: str, message: Dict[str, Any]) -> bool:
    """Send message to specific connection"""
    return await nis_websocket_manager.send_message(connection_id, message)
