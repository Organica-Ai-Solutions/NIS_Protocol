#!/usr/bin/env python3
"""
Simple test for NIS State Management System
Tests our WebSocket and state management without the full main.py complexity
"""

import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Import our state management system
from src.core.state_manager import (
    nis_state_manager, StateEventType, emit_state_event, 
    update_system_state, get_current_state
)
from src.core.websocket_manager import (
    nis_websocket_manager, ConnectionType
)
from src.core.agent_orchestrator import (
    nis_agent_orchestrator, AgentStatus, AgentType
)

app = FastAPI(title="NIS State Management Test")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    """Simple test page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NIS State Management Test</title>
        <script src="/static/js/nis-state-client.js"></script>
    </head>
    <body>
        <h1>🧠 NIS State Management Test</h1>
        <div id="status">Connecting...</div>
        <div id="messages"></div>
        <button onclick="testStateUpdate()">Test State Update</button>
        <button onclick="testEvent()">Test Event</button>
        <button onclick="testAgentActivation()">Test Agent Activation</button>
        <a href="/enhanced" target="_blank" style="display: inline-block; margin: 10px; padding: 10px 20px; background: #06b6d4; color: white; text-decoration: none; border-radius: 5px;">🧠 Enhanced Agent Chat</a>
        
        <script>
            const statusDiv = document.getElementById('status');
            const messagesDiv = document.getElementById('messages');
            
            // Initialize state manager
            const stateManager = new NISStateManager({
                debug: true,
                autoConnect: true
            });
            
            // Listen for connection events
            stateManager.client.on('connected', () => {
                statusDiv.innerHTML = '✅ Connected to NIS Protocol';
                statusDiv.style.color = 'green';
            });
            
            stateManager.client.on('disconnected', () => {
                statusDiv.innerHTML = '❌ Disconnected';
                statusDiv.style.color = 'red';
            });
            
            // Listen for state updates
            stateManager.client.on('initial_state', ({ state }) => {
                addMessage('📊 Initial state received: ' + JSON.stringify(state, null, 2));
            });
            
            stateManager.client.on('state_event', (event) => {
                addMessage('🎯 Event: ' + event.event_type + ' - ' + JSON.stringify(event.data));
            });
            
            function addMessage(msg) {
                const div = document.createElement('div');
                div.innerHTML = '<pre>' + msg + '</pre>';
                div.style.margin = '10px 0';
                div.style.padding = '10px';
                div.style.background = '#f0f0f0';
                div.style.borderRadius = '5px';
                messagesDiv.appendChild(div);
            }
            
            function testStateUpdate() {
                fetch('/api/state/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        updates: {
                            system_health: 'healthy',
                            total_requests: Math.floor(Math.random() * 1000),
                            active_agents: {
                                'test_agent_1': {'status': 'active'},
                                'test_agent_2': {'status': 'busy'}
                            }
                        }
                    })
                });
            }
            
            function testEvent() {
                fetch('/api/state/emit-event', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        event_type: 'recommendation_generated',
                        data: {
                            message: 'Test recommendation from frontend!',
                            timestamp: Date.now()
                        }
                    })
                });
            }
            
            function testAgentActivation() {
                fetch('/api/agents/activate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        agent_id: 'vision',
                        context: 'user_request_test'
                    })
                });
            }
        </script>
    </body>
    </html>
    """)

@app.get("/enhanced")
async def enhanced_agent_chat():
    """Enhanced Agent Chat with Brain Visualization"""
    with open("static/enhanced_agent_chat.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/state/{connection_type}")
async def websocket_state_endpoint(
    websocket: WebSocket, 
    connection_type: str,
    user_id: str = None,
    session_id: str = None
):
    """WebSocket endpoint for state management"""
    try:
        # Parse connection type
        try:
            conn_type = ConnectionType(connection_type)
        except ValueError:
            conn_type = ConnectionType.DASHBOARD
        
        # Connect to WebSocket manager
        connection_id = await nis_websocket_manager.connect(
            websocket=websocket,
            connection_type=conn_type,
            user_id=user_id,
            session_id=session_id
        )
        
        print(f"🔌 WebSocket connected: {connection_id} ({connection_type})")
        
        try:
            # Listen for client messages
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    await nis_websocket_manager.handle_client_message(connection_id, message)
                except json.JSONDecodeError:
                    print(f"Invalid JSON from {connection_id}: {data}")
                
        except WebSocketDisconnect:
            print(f"🔌 WebSocket disconnected: {connection_id}")
        
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup connection
        if 'connection_id' in locals():
            await nis_websocket_manager.disconnect(connection_id)

@app.get("/api/state/current")
async def get_current_system_state():
    """Get current system state"""
    try:
        state = get_current_state()
        
        # Add real-time metrics
        state["websocket_metrics"] = nis_websocket_manager.get_metrics()
        state["state_manager_metrics"] = nis_state_manager.get_metrics()
        state["timestamp"] = time.time()
        
        return {
            "success": True,
            "state": state,
            "message": "Current system state retrieved successfully"
        }
    except Exception as e:
        print(f"Failed to get current state: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/state/update")
async def update_system_state_endpoint(request: dict):
    """Update system state"""
    try:
        updates = request.get("updates", {})
        emit_event = request.get("emit_event", True)
        
        if not updates:
            return {"success": False, "error": "No updates provided"}
        
        await update_system_state(updates)
        
        if emit_event:
            await emit_state_event(
                StateEventType.SYSTEM_STATUS_CHANGE,
                {"manual_update": True, "updated_fields": list(updates.keys())}
            )
        
        return {
            "success": True,
            "message": f"System state updated: {list(updates.keys())}",
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"Failed to update state: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/state/emit-event")
async def emit_custom_event(request: dict):
    """Emit custom state event"""
    try:
        event_type_str = request.get("event_type")
        data = request.get("data", {})
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        priority = request.get("priority", "normal")
        
        if not event_type_str:
            return {"success": False, "error": "event_type is required"}
        
        try:
            event_type = StateEventType(event_type_str)
        except ValueError:
            return {"success": False, "error": f"Invalid event_type: {event_type_str}"}
        
        event_id = await emit_state_event(
            event_type=event_type,
            data=data,
            user_id=user_id,
            session_id=session_id,
            priority=priority
        )
        
        return {
            "success": True,
            "event_id": event_id,
            "message": f"Event emitted: {event_type_str}",
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"Failed to emit event: {e}")
        return {"success": False, "error": str(e)}

async def demo_state_updates():
    """Demo function to show automatic state updates"""
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds
        
        # Simulate system metrics
        await update_system_state({
            "uptime": time.time(),
            "total_requests": nis_state_manager.metrics["events_emitted"],
            "active_connections": len(nis_websocket_manager.connections),
            "system_health": "healthy"
        })
        
        # Emit a demo event occasionally
        if nis_state_manager.metrics["events_emitted"] % 3 == 0:
            await emit_state_event(
                StateEventType.RECOMMENDATION_GENERATED,
                {
                    "message": f"Demo recommendation #{nis_state_manager.metrics['events_emitted']}",
                    "confidence": 0.85,
                    "source": "demo_system"
                }
            )

# ====== AGENT ORCHESTRATOR ENDPOINTS ======

@app.get("/api/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = nis_agent_orchestrator.get_agent_status()
        return {
            "success": True,
            "agents": status,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of specific agent"""
    try:
        status = nis_agent_orchestrator.get_agent_status(agent_id)
        if not status:
            return {"success": False, "error": "Agent not found"}
        
        return {
            "success": True,
            "agent": status,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/agents/activate")
async def activate_agent(request: dict):
    """Activate a specific agent"""
    try:
        agent_id = request.get("agent_id")
        context = request.get("context", "manual_activation")
        force = request.get("force", False)
        
        if not agent_id:
            return {"success": False, "error": "agent_id is required"}
        
        success = await nis_agent_orchestrator.activate_agent(agent_id, context, force)
        
        return {
            "success": success,
            "message": f"Agent {agent_id} {'activated' if success else 'activation failed'}",
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/agents/process")
async def process_request(request: dict):
    """Process a request through the agent orchestrator"""
    try:
        input_data = request.get("input", {})
        
        result = await nis_agent_orchestrator.process_request(input_data)
        
        return {
            "success": True,
            "result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Start background tasks when server starts"""
    # Start the agent orchestrator
    await nis_agent_orchestrator.start_orchestrator()
    
    # Start demo updates
    asyncio.create_task(demo_state_updates())

if __name__ == "__main__":
    print("🚀 Starting NIS State Management Test Server...")
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
