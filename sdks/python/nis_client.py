"""
NIS Protocol Python Client SDK
Simple, easy-to-use client for interacting with NIS Protocol backend
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Structured chat response"""
    response: str
    provider: str
    model: str
    confidence: float
    tokens_used: int
    real_ai: bool
    reasoning_trace: List[str]
    
    
@dataclass
class AgentStatus:
    """Agent status information"""
    name: str
    type: str
    status: str
    capabilities: List[str]


class NISClient:
    """
    NIS Protocol Python Client
    
    Easy-to-use client for interacting with NIS Protocol backend.
    
    Example:
        client = NISClient("http://localhost:8000")
        response = client.chat("Hello, how are you?")
        print(response.response)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize NIS client
        
        Args:
            base_url: Backend URL (default: http://localhost:8000)
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
            
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to backend"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
        
    # ====== CHAT METHODS ======
    
    def chat(
        self,
        message: str,
        user_id: str = "default",
        conversation_id: Optional[str] = None,
        provider: Optional[str] = None,
        agent_type: str = "reasoning"
    ) -> ChatResponse:
        """
        Send chat message and get response
        
        Args:
            message: User message
            user_id: User identifier
            conversation_id: Optional conversation ID
            provider: LLM provider (openai, anthropic, google, deepseek, kimi, smart)
            agent_type: Agent type (reasoning, creative, analytical)
            
        Returns:
            ChatResponse object with response and metadata
        """
        data = {
            "message": message,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "provider": provider,
            "agent_type": agent_type
        }
        
        result = self._request("POST", "/chat", json=data)
        
        return ChatResponse(
            response=result.get("response", ""),
            provider=result.get("provider", "unknown"),
            model=result.get("model", "unknown"),
            confidence=result.get("confidence", 0.0),
            tokens_used=result.get("tokens_used", 0),
            real_ai=result.get("real_ai", False),
            reasoning_trace=result.get("reasoning_trace", [])
        )
        
    def smart_consensus(self, message: str, **kwargs) -> ChatResponse:
        """Use Smart Consensus (multiple LLMs)"""
        return self.chat(message, provider="smart", **kwargs)
        
    # ====== AGENT METHODS ======
    
    def get_agents(self) -> List[AgentStatus]:
        """Get all registered agents"""
        result = self._request("GET", "/agents/status")
        agents = result.get("agents", [])
        
        return [
            AgentStatus(
                name=agent.get("name", ""),
                type=agent.get("type", ""),
                status=agent.get("status", ""),
                capabilities=agent.get("capabilities", [])
            )
            for agent in agents
        ]
        
    # ====== PHYSICS METHODS ======
    
    def validate_physics(
        self,
        scenario: str,
        domain: str = "mechanics",
        mode: str = "true_pinn"
    ) -> Dict[str, Any]:
        """
        Validate physics scenario
        
        Args:
            scenario: Physics scenario description
            domain: Physics domain (mechanics, electromagnetism, etc.)
            mode: Validation mode (true_pinn, enhanced_pinn, advanced_pinn)
            
        Returns:
            Physics validation results
        """
        data = {
            "scenario": scenario,
            "domain": domain,
            "mode": mode
        }
        
        return self._request("POST", "/physics/validate", json=data)
        
    def get_physics_capabilities(self) -> Dict[str, Any]:
        """Get physics system capabilities"""
        return self._request("GET", "/physics/capabilities")
        
    # ====== RESEARCH METHODS ======
    
    def deep_research(
        self,
        query: str,
        depth: str = "comprehensive",
        sources: int = 10
    ) -> Dict[str, Any]:
        """
        Perform deep research
        
        Args:
            query: Research query
            depth: Research depth (quick, standard, comprehensive)
            sources: Number of sources to analyze
            
        Returns:
            Research results with analysis and sources
        """
        data = {
            "query": query,
            "depth": depth,
            "max_sources": sources
        }
        
        return self._request("POST", "/research/deep", json=data)
        
    # ====== UTILITY METHODS ======
    
    def health(self) -> Dict[str, Any]:
        """Check backend health"""
        return self._request("GET", "/health")
        
    def version(self) -> str:
        """Get backend version"""
        health = self.health()
        return health.get("pattern", "unknown")
        
    def is_healthy(self) -> bool:
        """Check if backend is healthy"""
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except:
            return False


# ====== ASYNC CLIENT ======

try:
    import aiohttp
    
    class AsyncNISClient:
        """
        Async NIS Protocol Python Client
        
        Async version for use with asyncio applications.
        
        Example:
            async with AsyncNISClient("http://localhost:8000") as client:
                response = await client.chat("Hello!")
                print(response.response)
        """
        
        def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
            self.base_url = base_url.rstrip('/')
            self.api_key = api_key
            self.session = None
            
        async def __aenter__(self):
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            self.session = aiohttp.ClientSession(headers=headers)
            return self
            
        async def __aexit__(self, *args):
            if self.session:
                await self.session.close()
                
        async def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
            """Make async HTTP request"""
            url = f"{self.base_url}{endpoint}"
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
                
        async def chat(
            self,
            message: str,
            user_id: str = "default",
            conversation_id: Optional[str] = None,
            provider: Optional[str] = None,
            agent_type: str = "reasoning"
        ) -> ChatResponse:
            """Async chat method"""
            data = {
                "message": message,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "provider": provider,
                "agent_type": agent_type
            }
            
            result = await self._request("POST", "/chat", json=data)
            
            return ChatResponse(
                response=result.get("response", ""),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                confidence=result.get("confidence", 0.0),
                tokens_used=result.get("tokens_used", 0),
                real_ai=result.get("real_ai", False),
                reasoning_trace=result.get("reasoning_trace", [])
            )
            
        async def health(self) -> Dict[str, Any]:
            """Check backend health (async)"""
            return await self._request("GET", "/health")
            
except ImportError:
    AsyncNISClient = None


# ====== EXAMPLE USAGE ======

if __name__ == "__main__":
    # Sync example
    client = NISClient("http://localhost:8000")
    
    # Check health
    if client.is_healthy():
        print("✅ Backend is healthy!")
        
    # Simple chat
    response = client.chat("What is quantum computing?")
    print(f"Response: {response.response}")
    print(f"Provider: {response.provider}")
    print(f"Tokens: {response.tokens_used}")
    
    # Smart Consensus
    consensus = client.smart_consensus("Explain machine learning")
    print(f"Consensus response: {consensus.response}")
    
    # Get agents
    agents = client.get_agents()
    print(f"Active agents: {len(agents)}")
    
    # Physics validation
    physics = client.validate_physics(
        "Ball thrown at 45 degrees with initial velocity 20 m/s",
        domain="mechanics"
    )
    print(f"Physics result: {physics}")

