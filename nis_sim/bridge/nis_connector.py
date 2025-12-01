"""
NIS Protocol Connector
Bridge between simulation and NIS Protocol API
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@dataclass
class NISConfig:
    """NIS Protocol connection config"""
    host: str = "localhost"
    port: int = 8000
    api_version: str = "v4"
    use_ssl: bool = False
    api_key: Optional[str] = None


class NISConnector:
    """
    Connects simulation agents to NIS Protocol API
    Sends sensor data, receives commands
    """
    
    def __init__(self, config: Optional[NISConfig] = None):
        self.config = config or NISConfig()
        self.base_url = f"{'https' if self.config.use_ssl else 'http'}://{self.config.host}:{self.config.port}"
        self.ws_url = f"{'wss' if self.config.use_ssl else 'ws'}://{self.config.host}:{self.config.port}"
        self.client: Optional[httpx.AsyncClient] = None
        self.ws_connection = None
        self._command_callback: Optional[Callable] = None
        
    async def connect(self):
        """Establish connection to NIS Protocol"""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required. Install with: pip install httpx")
        
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0
        )
        
        # Test connection
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                print(f"✅ Connected to NIS Protocol at {self.base_url}")
                return True
        except Exception as e:
            print(f"⚠️ Could not connect to NIS Protocol: {e}")
            print(f"   Simulation will run in standalone mode")
            return False
    
    async def disconnect(self):
        """Close connection"""
        if self.client:
            await self.client.aclose()
            self.client = None
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
    
    # ==================== REST API Methods ====================
    
    async def send_telemetry(self, agent_id: str, telemetry: Dict[str, Any]) -> Dict:
        """Send agent telemetry to NIS Protocol"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/robotics/telemetry",
                json={
                    "agent_id": agent_id,
                    "telemetry": telemetry,
                    "source": "simulation"
                }
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_command(self, agent_id: str) -> Optional[Dict]:
        """Get pending command for agent from NIS Protocol"""
        if not self.client:
            return None
        
        try:
            response = await self.client.get(
                f"/{self.config.api_version}/robotics/command/{agent_id}"
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    async def register_agent(self, agent_id: str, agent_type: str, capabilities: list) -> Dict:
        """Register simulated agent with NIS Protocol"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/robotics/register",
                json={
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "capabilities": capabilities,
                    "source": "simulation",
                    "simulated": True
                }
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def request_physics_validation(self, command: Dict) -> Dict:
        """Request physics validation from NIS Protocol"""
        if not self.client:
            return {"valid": True, "message": "No NIS connection, skipping validation"}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/physics/validate",
                json=command
            )
            return response.json()
        except Exception as e:
            return {"valid": True, "message": f"Validation error: {e}"}
    
    async def report_event(self, event_type: str, data: Dict) -> Dict:
        """Report simulation event to NIS Protocol"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/events",
                json={
                    "event_type": event_type,
                    "data": data,
                    "source": "simulation"
                }
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ==================== WebSocket Methods ====================
    
    async def connect_websocket(self, agent_id: str, on_command: Callable):
        """Connect WebSocket for real-time command streaming"""
        if not WEBSOCKETS_AVAILABLE:
            print("⚠️ websockets not available for real-time connection")
            return
        
        self._command_callback = on_command
        ws_endpoint = f"{self.ws_url}/{self.config.api_version}/robotics/ws/{agent_id}"
        
        try:
            self.ws_connection = await websockets.connect(ws_endpoint)
            print(f"✅ WebSocket connected for agent {agent_id}")
            
            # Start listening for commands
            asyncio.create_task(self._ws_listener())
        except Exception as e:
            print(f"⚠️ WebSocket connection failed: {e}")
    
    async def _ws_listener(self):
        """Listen for WebSocket messages"""
        if not self.ws_connection:
            return
        
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                if self._command_callback:
                    await self._command_callback(data)
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    async def send_ws_telemetry(self, telemetry: Dict):
        """Send telemetry over WebSocket"""
        if self.ws_connection:
            await self.ws_connection.send(json.dumps({
                "type": "telemetry",
                "data": telemetry
            }))
    
    # ==================== Consciousness Integration ====================
    
    async def trigger_consciousness_eval(self, agent_id: str, situation: Dict) -> Dict:
        """Request consciousness evaluation for autonomous decision"""
        if not self.client:
            return {"decision": "continue", "confidence": 0.5}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/consciousness/evaluate",
                json={
                    "agent_id": agent_id,
                    "situation": situation,
                    "source": "simulation"
                }
            )
            return response.json()
        except Exception as e:
            return {"decision": "continue", "confidence": 0.5, "error": str(e)}
    
    async def request_ethical_check(self, agent_id: str, action: str, context: Dict) -> Dict:
        """Request ethical evaluation for planned action"""
        if not self.client:
            return {"approved": True, "reason": "No NIS connection"}
        
        try:
            response = await self.client.post(
                f"/{self.config.api_version}/consciousness/ethics/evaluate",
                json={
                    "agent_id": agent_id,
                    "action": action,
                    "context": context
                }
            )
            return response.json()
        except Exception as e:
            return {"approved": True, "reason": f"Ethics check error: {e}"}


class SimulationBridge:
    """
    High-level bridge that manages multiple agents
    """
    
    def __init__(self, nis_config: Optional[NISConfig] = None):
        self.connector = NISConnector(nis_config)
        self.agents: Dict[str, Any] = {}
        self._running = False
    
    async def start(self):
        """Start the bridge"""
        await self.connector.connect()
        self._running = True
    
    async def stop(self):
        """Stop the bridge"""
        self._running = False
        await self.connector.disconnect()
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register a simulation agent"""
        self.agents[agent_id] = agent
    
    async def sync_loop(self, interval: float = 0.1):
        """
        Main sync loop - sends telemetry, receives commands
        Run this as a background task during simulation
        """
        while self._running:
            for agent_id, agent in self.agents.items():
                # Send telemetry
                state = agent.get_state()
                await self.connector.send_telemetry(agent_id, state)
                
                # Check for commands
                command = await self.connector.get_command(agent_id)
                if command:
                    agent.queue_command(command)
            
            await asyncio.sleep(interval)
