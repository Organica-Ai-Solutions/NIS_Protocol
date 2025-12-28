"""
A2A (Agent-to-Agent) Protocol Implementation for GenUI
Official GenUI A2UI WebSocket Protocol

This module implements the official A2A streaming protocol as specified by
the GenUI framework for real-time agent-to-UI communication.

Copyright 2025 Organica AI Solutions
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger("nis.a2a_protocol")


class A2AMessageType(Enum):
    """A2A Protocol Message Types"""
    AGENT_CARD = "agent_card"
    SURFACE_UPDATE = "surface_update"
    DATA_MODEL_UPDATE = "data_model_update"
    BEGIN_RENDERING = "begin_rendering"
    END_RENDERING = "end_rendering"
    ERROR = "error"
    TEXT_CHUNK = "text_chunk"
    USER_EVENT = "user_event"


class AgentCard:
    """
    Agent metadata card sent at connection start.
    Describes the AI agent's capabilities and identity.
    """
    
    def __init__(
        self,
        name: str = "NIS Protocol",
        description: str = "Neural Intelligence System - Advanced AI Agent",
        version: str = "4.0.1",
        capabilities: Optional[List[str]] = None,
        avatar_url: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or [
            "code_generation",
            "physics_simulation",
            "vision_analysis",
            "research",
            "robotics",
            "consciousness"
        ]
        self.avatar_url = avatar_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2A message format"""
        return {
            "type": A2AMessageType.AGENT_CARD.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "capabilities": self.capabilities,
                "avatar_url": self.avatar_url
            }
        }


class SurfaceUpdate:
    """
    Surface update message for dynamic UI rendering.
    Contains widget data to be rendered on a specific surface.
    """
    
    def __init__(
        self,
        surface_id: str,
        widget_type: str,
        widget_data: Dict[str, Any],
        replace: bool = False
    ):
        self.surface_id = surface_id
        self.widget_type = widget_type
        self.widget_data = widget_data
        self.replace = replace
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2A message format"""
        return {
            "type": A2AMessageType.SURFACE_UPDATE.value,
            "timestamp": datetime.utcnow().isoformat(),
            "surfaceId": self.surface_id,
            "replace": self.replace,
            "data": {
                "type": self.widget_type,
                **self.widget_data
            }
        }


class DataModelUpdate:
    """
    Data model update message.
    Updates the data model without changing the UI structure.
    """
    
    def __init__(
        self,
        model_id: str,
        updates: Dict[str, Any]
    ):
        self.model_id = model_id
        self.updates = updates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2A message format"""
        return {
            "type": A2AMessageType.DATA_MODEL_UPDATE.value,
            "timestamp": datetime.utcnow().isoformat(),
            "modelId": self.model_id,
            "updates": self.updates
        }


class BeginRendering:
    """Signal that rendering should begin"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": A2AMessageType.BEGIN_RENDERING.value,
            "timestamp": datetime.utcnow().isoformat(),
            "taskId": self.task_id
        }


class EndRendering:
    """Signal that rendering is complete"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": A2AMessageType.END_RENDERING.value,
            "timestamp": datetime.utcnow().isoformat(),
            "taskId": self.task_id
        }


class TextChunk:
    """Streaming text chunk for progressive text display"""
    
    def __init__(self, text: str, surface_id: str = "main"):
        self.text = text
        self.surface_id = surface_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": A2AMessageType.TEXT_CHUNK.value,
            "timestamp": datetime.utcnow().isoformat(),
            "surfaceId": self.surface_id,
            "text": self.text
        }


class A2ASession:
    """
    Manages an A2A protocol session.
    Handles message streaming and state management.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.task_id: Optional[str] = None
        self.context_id: Optional[str] = None
        self.surfaces: Dict[str, List[Dict[str, Any]]] = {}
        self.data_models: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info(f"A2A Session created: {self.session_id}")
    
    def start_task(self, task_id: Optional[str] = None) -> str:
        """Start a new task"""
        self.task_id = task_id or str(uuid.uuid4())
        self.context_id = str(uuid.uuid4())
        logger.info(f"Task started: {self.task_id}")
        return self.task_id
    
    def add_surface_widget(self, surface_id: str, widget: Dict[str, Any]):
        """Add widget to surface"""
        if surface_id not in self.surfaces:
            self.surfaces[surface_id] = []
        self.surfaces[surface_id].append(widget)
    
    def clear_surface(self, surface_id: str):
        """Clear all widgets from surface"""
        self.surfaces[surface_id] = []
    
    def update_data_model(self, model_id: str, updates: Dict[str, Any]):
        """Update data model"""
        if model_id not in self.data_models:
            self.data_models[model_id] = {}
        self.data_models[model_id].update(updates)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for user events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def handle_user_event(self, event: Dict[str, Any]):
        """Handle incoming user event"""
        event_type = event.get("type")
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error handling event {event_type}: {e}")


class A2AProtocolHandler:
    """
    Main A2A Protocol handler.
    Manages WebSocket connections and message streaming.
    """
    
    def __init__(self, llm_provider=None, a2ui_formatter=None):
        self.llm_provider = llm_provider
        self.a2ui_formatter = a2ui_formatter
        self.active_sessions: Dict[str, A2ASession] = {}
        logger.info("A2A Protocol Handler initialized")
    
    async def handle_connection(self, websocket):
        """
        Handle a new WebSocket connection.
        Implements the full A2A protocol flow.
        """
        session = A2ASession()
        self.active_sessions[session.session_id] = session
        
        try:
            # 1. Send AgentCard
            agent_card = AgentCard()
            await websocket.send_json(agent_card.to_dict())
            logger.info(f"Sent AgentCard to session {session.session_id}")
            
            # 2. Listen for messages
            while True:
                try:
                    message = await websocket.receive_json()
                    await self._process_message(websocket, session, message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self._send_error(websocket, str(e))
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Cleanup
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            logger.info(f"Session closed: {session.session_id}")
    
    async def _process_message(
        self,
        websocket,
        session: A2ASession,
        message: Dict[str, Any]
    ):
        """Process incoming message from client"""
        msg_type = message.get("type")
        
        if msg_type == "user_message":
            await self._handle_user_message(websocket, session, message)
        elif msg_type == "user_event":
            await session.handle_user_event(message)
        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _handle_user_message(
        self,
        websocket,
        session: A2ASession,
        message: Dict[str, Any]
    ):
        """
        Handle user message and generate A2UI response.
        Streams updates to the client in real-time.
        """
        user_text = message.get("text", "")
        surface_id = message.get("surfaceId", "main")
        
        # Start task
        task_id = session.start_task()
        
        # Send BeginRendering
        await websocket.send_json(BeginRendering(task_id).to_dict())
        
        try:
            # Generate response from LLM
            if self.llm_provider:
                response_text = await self._generate_llm_response(user_text)
            else:
                response_text = f"Echo: {user_text}"
            
            # Convert to A2UI widgets
            if self.a2ui_formatter:
                a2ui_message = self.a2ui_formatter.format_response(
                    response_text,
                    include_actions=True
                )
                
                # Stream each widget as SurfaceUpdate
                for widget in a2ui_message.get("widgets", []):
                    surface_update = SurfaceUpdate(
                        surface_id=surface_id,
                        widget_type=widget.get("type"),
                        widget_data=widget.get("data", {})
                    )
                    await websocket.send_json(surface_update.to_dict())
                    session.add_surface_widget(surface_id, widget)
                    
                    # Small delay for streaming effect
                    await asyncio.sleep(0.1)
            else:
                # Fallback: send as text chunk
                text_chunk = TextChunk(response_text, surface_id)
                await websocket.send_json(text_chunk.to_dict())
            
            # Send EndRendering
            await websocket.send_json(EndRendering(task_id).to_dict())
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self._send_error(websocket, str(e))
            await websocket.send_json(EndRendering(task_id).to_dict())
    
    async def _generate_llm_response(self, user_text: str) -> str:
        """Generate response from LLM provider"""
        try:
            # Use the LLM provider to generate response
            response = self.llm_provider.generate(
                prompt=user_text,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message to client"""
        await websocket.send_json({
            "type": A2AMessageType.ERROR.value,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message
        })


def create_a2a_handler(llm_provider=None, a2ui_formatter=None) -> A2AProtocolHandler:
    """
    Factory function to create A2A protocol handler.
    
    Args:
        llm_provider: LLM provider instance for generating responses
        a2ui_formatter: A2UI formatter instance for converting text to widgets
        
    Returns:
        Configured A2AProtocolHandler instance
    """
    return A2AProtocolHandler(
        llm_provider=llm_provider,
        a2ui_formatter=a2ui_formatter
    )
