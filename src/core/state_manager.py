#!/usr/bin/env python3
"""
NIS Protocol State Management System
Real-time state synchronization for backend → frontend communication

This system allows the powerful NIS backend to automatically control
and update frontend interfaces in real-time, similar to modern AI companies.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)

class StateEventType(Enum):
    """Types of state events that can be emitted"""
    SYSTEM_STATUS_CHANGE = "system_status_change"
    AGENT_STATUS_CHANGE = "agent_status_change"
    LLM_PROVIDER_CHANGE = "llm_provider_change"
    CHAT_MESSAGE = "chat_message"
    ANALYTICS_UPDATE = "analytics_update"
    ERROR_OCCURRED = "error_occurred"
    USER_ACTION_REQUIRED = "user_action_required"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    BACKEND_INSTRUCTION = "backend_instruction"
    UI_STATE_UPDATE = "ui_state_update"

@dataclass
class StateEvent:
    """Represents a state change event"""
    event_id: str
    event_type: StateEventType
    data: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "priority": self.priority
        }

@dataclass
class NISSystemState:
    """Complete NIS Protocol system state"""
    # System health
    system_health: str = "healthy"
    uptime: float = 0.0
    
    # Agent status
    active_agents: Dict[str, Dict[str, Any]] = None
    agent_performance: Dict[str, float] = None
    
    # LLM providers
    llm_providers: Dict[str, Dict[str, Any]] = None
    llm_performance: Dict[str, Dict[str, Any]] = None
    
    # Analytics
    total_requests: int = 0
    total_tokens: int = 0
    average_response_time: float = 0.0
    
    # Current operations
    active_conversations: Dict[str, Dict[str, Any]] = None
    pending_operations: List[Dict[str, Any]] = None
    
    # Recommendations
    ai_recommendations: List[Dict[str, Any]] = None
    system_alerts: List[Dict[str, Any]] = None
    
    # Agent Orchestrator specific data
    orchestrator_metrics: Dict[str, Any] = None
    total_agents: int = 0
    active_instances: int = 0
    active_connections: int = 0
    
    def __post_init__(self):
        if self.active_agents is None:
            self.active_agents = {}
        if self.agent_performance is None:
            self.agent_performance = {}
        if self.llm_providers is None:
            self.llm_providers = {}
        if self.llm_performance is None:
            self.llm_performance = {}
        if self.active_conversations is None:
            self.active_conversations = {}
        if self.pending_operations is None:
            self.pending_operations = []
        if self.ai_recommendations is None:
            self.ai_recommendations = []
        if self.system_alerts is None:
            self.system_alerts = []
        if self.orchestrator_metrics is None:
            self.orchestrator_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class NISStateManager:
    """
    🧠 NIS Protocol State Management System
    
    This system provides:
    - Real-time state synchronization
    - Event-driven updates
    - WebSocket integration
    - Automatic frontend control
    - Performance monitoring
    """
    
    def __init__(self):
        self.state = NISSystemState()
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self.websocket_connections: Dict[str, Any] = {}
        self.event_history: List[StateEvent] = []
        self.max_history = 1000
        
        # Performance tracking
        self.metrics = {
            "events_emitted": 0,
            "subscribers_count": 0,
            "websocket_connections": 0,
            "state_updates": 0
        }
        
        logger.info("🧠 NIS State Manager initialized")
    
    def get_state(self) -> NISSystemState:
        """Get current system state"""
        return self.state
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current system state as dictionary"""
        return self.state.to_dict()
    
    async def update_state(self, updates: Dict[str, Any], emit_event: bool = True) -> None:
        """
        Update system state and optionally emit event
        
        Args:
            updates: Dictionary of state updates
            emit_event: Whether to emit state change event
        """
        try:
            # Update state
            for key, value in updates.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
                else:
                    # Log unknown keys for debugging (but don't make it a warning)
                    logger.debug(f"Adding dynamic state key: {key}")
            
            self.metrics["state_updates"] += 1
            
            # Emit event if requested
            if emit_event:
                await self.emit_event(
                    StateEventType.UI_STATE_UPDATE,
                    {"updated_fields": list(updates.keys()), "new_state": updates}
                )
                
            logger.debug(f"State updated: {list(updates.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            await self.emit_event(
                StateEventType.ERROR_OCCURRED,
                {"error": str(e), "context": "state_update"}
            )
    
    async def emit_event(
        self, 
        event_type: StateEventType, 
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        priority: str = "normal"
    ) -> str:
        """
        Emit a state event to all subscribers
        
        Returns:
            event_id: Unique identifier for the event
        """
        event_id = str(uuid.uuid4())
        
        event = StateEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            priority=priority
        )
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notify subscribers
        await self._notify_subscribers(event)
        
        # Send to WebSocket connections
        await self._notify_websocket_connections(event)
        
        self.metrics["events_emitted"] += 1
        
        logger.debug(f"Event emitted: {event_type.value} ({event_id})")
        
        return event_id
    
    def subscribe(self, event_type: StateEventType, callback: Callable) -> str:
        """
        Subscribe to state events
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
            
        Returns:
            subscription_id: Unique identifier for the subscription
        """
        subscription_id = str(uuid.uuid4())
        
        # Use weak reference to avoid memory leaks
        weak_callback = weakref.ref(callback)
        self.subscribers[event_type.value].add((subscription_id, weak_callback))
        
        self.metrics["subscribers_count"] = sum(len(subs) for subs in self.subscribers.values())
        
        logger.debug(f"New subscription: {event_type.value} ({subscription_id})")
        
        return subscription_id
    
    def unsubscribe(self, event_type: StateEventType, subscription_id: str) -> bool:
        """
        Unsubscribe from state events
        
        Args:
            event_type: Type of events to unsubscribe from
            subscription_id: Subscription identifier
            
        Returns:
            success: Whether unsubscription was successful
        """
        subscribers = self.subscribers[event_type.value]
        
        for sub_id, weak_callback in list(subscribers):
            if sub_id == subscription_id:
                subscribers.remove((sub_id, weak_callback))
                self.metrics["subscribers_count"] = sum(len(subs) for subs in self.subscribers.values())
                logger.debug(f"Unsubscribed: {event_type.value} ({subscription_id})")
                return True
        
        return False
    
    async def _notify_subscribers(self, event: StateEvent) -> None:
        """Notify all subscribers of an event"""
        subscribers = self.subscribers[event.event_type.value]
        
        for sub_id, weak_callback in list(subscribers):
            callback = weak_callback()
            if callback is None:
                # Callback was garbage collected, remove subscription
                subscribers.remove((sub_id, weak_callback))
                continue
            
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    async def _notify_websocket_connections(self, event: StateEvent) -> None:
        """Send event to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(event.to_dict())
        
        # Send to all connections (or filter by user_id/session_id)
        for connection_id, websocket in list(self.websocket_connections.items()):
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                # Remove dead connection
                self.websocket_connections.pop(connection_id, None)
                self.metrics["websocket_connections"] = len(self.websocket_connections)
    
    def add_websocket_connection(self, connection_id: str, websocket: Any) -> None:
        """Add a WebSocket connection"""
        self.websocket_connections[connection_id] = websocket
        self.metrics["websocket_connections"] = len(self.websocket_connections)
        logger.info(f"WebSocket connection added: {connection_id}")
    
    def remove_websocket_connection(self, connection_id: str) -> None:
        """Remove a WebSocket connection"""
        if connection_id in self.websocket_connections:
            self.websocket_connections.pop(connection_id)
            self.metrics["websocket_connections"] = len(self.websocket_connections)
            logger.info(f"WebSocket connection removed: {connection_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics"""
        return {
            **self.metrics,
            "current_state_size": len(str(self.state.to_dict())),
            "event_history_size": len(self.event_history),
            "active_event_types": list(self.subscribers.keys())
        }
    
    async def generate_system_recommendations(self) -> List[Dict[str, Any]]:
        """Generate AI-powered system recommendations"""
        recommendations = []
        
        # Analyze system state and generate recommendations
        if self.state.average_response_time > 5000:  # 5 seconds
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "title": "High Response Time Detected",
                "description": f"Average response time is {self.state.average_response_time:.0f}ms",
                "action": "Consider optimizing LLM provider selection or adding caching"
            })
        
        if len(self.state.system_alerts) > 0:
            recommendations.append({
                "type": "alerts",
                "priority": "critical",
                "title": f"{len(self.state.system_alerts)} System Alerts",
                "description": "Active system alerts require attention",
                "action": "Review and resolve system alerts"
            })
        
        if len(self.state.active_agents) < 3:
            recommendations.append({
                "type": "agents",
                "priority": "medium", 
                "title": "Low Agent Activity",
                "description": f"Only {len(self.state.active_agents)} agents active",
                "action": "Consider activating additional agents for better coverage"
            })
        
        # Update state with recommendations
        await self.update_state({"ai_recommendations": recommendations})
        
        return recommendations

# Global state manager instance
nis_state_manager = NISStateManager()

# Convenience functions
async def emit_state_event(event_type: StateEventType, data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to emit state events"""
    return await nis_state_manager.emit_event(event_type, data, **kwargs)

async def update_system_state(updates: Dict[str, Any]) -> None:
    """Convenience function to update system state"""
    await nis_state_manager.update_state(updates)

def get_current_state() -> Dict[str, Any]:
    """Convenience function to get current state"""
    return nis_state_manager.get_state_dict()

def subscribe_to_events(event_type: StateEventType, callback: Callable) -> str:
    """Convenience function to subscribe to events"""
    return nis_state_manager.subscribe(event_type, callback)
