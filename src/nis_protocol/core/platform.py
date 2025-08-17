"""
NIS Protocol Platform Core
=========================

Central coordination and deployment platform for NIS Protocol agents and services.

This module provides the core platform functionality for:
- Agent registration and management
- Service orchestration
- Deployment coordination
- Resource management
- Health monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from .agent import NISAgent, NISLayer
from .registry import AgentRegistry
from .messaging import MessageBus


class PlatformState(Enum):
    """Platform operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEPLOYING = "deploying"
    SCALING = "scaling"


class DeploymentTarget(Enum):
    """Supported deployment targets."""
    EDGE = "edge"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


@dataclass
class PlatformConfig:
    """Platform configuration settings."""
    name: str = "nis-platform"
    version: str = "3.2.0"
    max_agents: int = 100
    message_queue_size: int = 10000
    health_check_interval: float = 30.0
    auto_scaling: bool = True
    logging_level: str = "INFO"
    deployment_target: DeploymentTarget = DeploymentTarget.LOCAL
    edge_config: Dict[str, Any] = field(default_factory=dict)
    cloud_config: Dict[str, Any] = field(default_factory=dict)


class NISPlatform:
    """
    Central coordination platform for NIS Protocol agents and services.
    
    The platform provides:
    - Agent lifecycle management
    - Inter-agent communication
    - Resource coordination
    - Deployment orchestration
    - Health monitoring and recovery
    - Scaling and load balancing
    
    Example:
        ```python
        from nis_protocol import NISPlatform, ConsciousnessAgent
        
        # Create platform
        platform = NISPlatform(config=PlatformConfig(
            name="my-ai-system",
            deployment_target=DeploymentTarget.EDGE
        ))
        
        # Add agents
        agent = ConsciousnessAgent("consciousness_001")
        platform.add_agent(agent)
        
        # Deploy and run
        await platform.deploy()
        await platform.start()
        ```
    """
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        """
        Initialize the NIS Protocol platform.
        
        Args:
            config: Platform configuration. Defaults to PlatformConfig()
        """
        self.config = config or PlatformConfig()
        self.state = PlatformState.INITIALIZING
        self.logger = logging.getLogger(f"nis.platform.{self.config.name}")
        
        # Core components
        self.registry = AgentRegistry()
        self.message_bus = MessageBus()
        self.agents: Dict[str, NISAgent] = {}
        
        # Runtime tracking
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self.health_status: Dict[str, Any] = {}
        
        # Deployment state
        self.deployment_info: Dict[str, Any] = {}
        
        self.logger.info(f"Initialized NIS Platform: {self.config.name}")
    
    async def add_agent(self, agent: NISAgent) -> bool:
        """
        Add an agent to the platform.
        
        Args:
            agent: NIS agent instance to add
            
        Returns:
            bool: True if agent was added successfully
        """
        try:
            # Check capacity
            if len(self.agents) >= self.config.max_agents:
                self.logger.error(f"Platform at capacity ({self.config.max_agents} agents)")
                return False
            
            # Register agent
            agent_id = agent.agent_id
            self.agents[agent_id] = agent
            self.registry.register(agent)
            
            # Connect to message bus
            await self.message_bus.connect_agent(agent)
            
            self.logger.info(f"Added agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add agent {agent.agent_id}: {e}")
            return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the platform.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            bool: True if agent was removed successfully
        """
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Agent not found: {agent_id}")
                return False
            
            agent = self.agents[agent_id]
            
            # Disconnect from message bus
            await self.message_bus.disconnect_agent(agent)
            
            # Unregister and remove
            self.registry.unregister(agent_id)
            del self.agents[agent_id]
            
            self.logger.info(f"Removed agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    async def start(self) -> bool:
        """
        Start the platform and all registered agents.
        
        Returns:
            bool: True if platform started successfully
        """
        try:
            self.logger.info("Starting NIS Platform...")
            self.state = PlatformState.INITIALIZING
            
            # Start message bus
            await self.message_bus.start()
            
            # Initialize all agents
            for agent_id, agent in self.agents.items():
                try:
                    # Initialize agent if it has an init method
                    if hasattr(agent, 'initialize'):
                        await agent.initialize()
                    self.logger.info(f"Initialized agent: {agent_id}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize agent {agent_id}: {e}")
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            # Update state
            self.state = PlatformState.RUNNING
            self.start_time = time.time()
            
            self.logger.info(f"NIS Platform started with {len(self.agents)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start platform: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def stop(self) -> bool:
        """
        Stop the platform and all agents.
        
        Returns:
            bool: True if platform stopped successfully
        """
        try:
            self.logger.info("Stopping NIS Platform...")
            self.state = PlatformState.STOPPED
            
            # Stop all agents
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, 'shutdown'):
                        await agent.shutdown()
                    self.logger.info(f"Stopped agent: {agent_id}")
                except Exception as e:
                    self.logger.error(f"Failed to stop agent {agent_id}: {e}")
            
            # Stop message bus
            await self.message_bus.stop()
            
            self.logger.info("NIS Platform stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop platform: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def deploy(self, target: Optional[DeploymentTarget] = None, **kwargs) -> bool:
        """
        Deploy the platform to a target environment.
        
        Args:
            target: Deployment target (edge, cloud, hybrid, etc.)
            **kwargs: Target-specific deployment parameters
            
        Returns:
            bool: True if deployment was successful
        """
        try:
            target = target or self.config.deployment_target
            self.state = PlatformState.DEPLOYING
            
            self.logger.info(f"Deploying to {target.value}...")
            
            if target == DeploymentTarget.EDGE:
                success = await self._deploy_edge(**kwargs)
            elif target == DeploymentTarget.CLOUD:
                success = await self._deploy_cloud(**kwargs)
            elif target == DeploymentTarget.HYBRID:
                success = await self._deploy_hybrid(**kwargs)
            elif target == DeploymentTarget.DOCKER:
                success = await self._deploy_docker(**kwargs)
            elif target == DeploymentTarget.KUBERNETES:
                success = await self._deploy_kubernetes(**kwargs)
            else:
                success = await self._deploy_local(**kwargs)
            
            if success:
                self.deployment_info = {
                    "target": target.value,
                    "timestamp": time.time(),
                    "config": kwargs
                }
                self.logger.info(f"Deployment to {target.value} successful")
            else:
                self.logger.error(f"Deployment to {target.value} failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def _deploy_edge(self, device_type: str = "raspberry_pi", **kwargs) -> bool:
        """Deploy to edge device."""
        self.logger.info(f"Deploying to edge device: {device_type}")
        
        # Edge-specific optimizations
        if device_type == "raspberry_pi":
            # Optimize for Raspberry Pi
            self.config.max_agents = min(self.config.max_agents, 20)
            self.config.message_queue_size = min(self.config.message_queue_size, 1000)
        
        # Apply edge configuration
        edge_config = kwargs.get("edge_config", self.config.edge_config)
        self.logger.info(f"Applied edge config: {edge_config}")
        
        return True
    
    async def _deploy_cloud(self, provider: str = "aws", **kwargs) -> bool:
        """Deploy to cloud infrastructure."""
        self.logger.info(f"Deploying to cloud provider: {provider}")
        
        # Cloud-specific scaling
        self.config.auto_scaling = True
        
        # Apply cloud configuration
        cloud_config = kwargs.get("cloud_config", self.config.cloud_config)
        self.logger.info(f"Applied cloud config: {cloud_config}")
        
        return True
    
    async def _deploy_hybrid(self, **kwargs) -> bool:
        """Deploy hybrid edge+cloud configuration."""
        self.logger.info("Deploying hybrid edge+cloud configuration")
        
        # Deploy core agents to edge
        await self._deploy_edge(**kwargs)
        
        # Deploy specialized agents to cloud
        await self._deploy_cloud(**kwargs)
        
        return True
    
    async def _deploy_docker(self, **kwargs) -> bool:
        """Deploy using Docker containers."""
        self.logger.info("Deploying with Docker containers")
        # Docker deployment logic would go here
        return True
    
    async def _deploy_kubernetes(self, **kwargs) -> bool:
        """Deploy to Kubernetes cluster.""" 
        self.logger.info("Deploying to Kubernetes cluster")
        # Kubernetes deployment logic would go here
        return True
    
    async def _deploy_local(self, **kwargs) -> bool:
        """Deploy locally for development."""
        self.logger.info("Deploying locally for development")
        return True
    
    async def _health_monitor(self):
        """Background health monitoring task."""
        while self.state == PlatformState.RUNNING:
            try:
                await self._update_health_status()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)  # Brief recovery pause
    
    async def _update_health_status(self):
        """Update platform and agent health status."""
        self.health_status = {
            "platform": {
                "state": self.state.value,
                "uptime": time.time() - (self.start_time or time.time()),
                "agent_count": len(self.agents),
                "message_queue_size": await self.message_bus.queue_size(),
            },
            "agents": {}
        }
        
        # Check agent health
        for agent_id, agent in self.agents.items():
            try:
                agent_status = agent.get_status()
                self.health_status["agents"][agent_id] = agent_status
            except Exception as e:
                self.health_status["agents"][agent_id] = {
                    "status": "error",
                    "error": str(e)
                }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive platform status.
        
        Returns:
            dict: Platform status information
        """
        return {
            "platform": {
                "name": self.config.name,
                "version": self.config.version,
                "state": self.state.value,
                "uptime": time.time() - (self.start_time or time.time()) if self.start_time else 0,
                "deployment": self.deployment_info,
            },
            "agents": {
                "count": len(self.agents),
                "registered": list(self.agents.keys()),
                "health": self.health_status.get("agents", {}),
            },
            "resources": {
                "message_queue_size": self.message_bus.queue_size() if hasattr(self.message_bus, 'queue_size') else 0,
                "max_agents": self.config.max_agents,
            },
            "configuration": {
                "auto_scaling": self.config.auto_scaling,
                "health_check_interval": self.config.health_check_interval,
                "deployment_target": self.config.deployment_target.value,
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get platform performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "runtime": {
                "uptime": time.time() - (self.start_time or time.time()) if self.start_time else 0,
                "state": self.state.value,
            },
            "agents": {
                "total": len(self.agents),
                "active": len([a for a in self.agents.values() if a.get_status().get("status") == "active"]),
                "by_layer": self._get_agents_by_layer(),
            },
            "messaging": {
                "queue_size": self.message_bus.queue_size() if hasattr(self.message_bus, 'queue_size') else 0,
                "messages_processed": getattr(self.message_bus, 'messages_processed', 0),
            },
            "resources": {
                "memory_usage": self._get_memory_usage(),
                "cpu_usage": self._get_cpu_usage(),
            }
        }
    
    def _get_agents_by_layer(self) -> Dict[str, int]:
        """Get agent count by NIS layer."""
        layer_count = {}
        for agent in self.agents.values():
            if hasattr(agent, 'layer'):
                layer = agent.layer.value if hasattr(agent.layer, 'value') else str(agent.layer)
                layer_count[layer] = layer_count.get(layer, 0) + 1
        return layer_count
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (mock implementation)."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (mock implementation)."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0


# Platform Factory Functions

def create_edge_platform(name: str, device_type: str = "raspberry_pi") -> NISPlatform:
    """
    Create a platform optimized for edge deployment.
    
    Args:
        name: Platform name
        device_type: Type of edge device
        
    Returns:
        NISPlatform: Configured for edge deployment
    """
    config = PlatformConfig(
        name=name,
        deployment_target=DeploymentTarget.EDGE,
        max_agents=20,
        message_queue_size=1000,
        auto_scaling=False,
        edge_config={"device_type": device_type}
    )
    return NISPlatform(config)


def create_cloud_platform(name: str, provider: str = "aws") -> NISPlatform:
    """
    Create a platform optimized for cloud deployment.
    
    Args:
        name: Platform name
        provider: Cloud provider
        
    Returns:
        NISPlatform: Configured for cloud deployment
    """
    config = PlatformConfig(
        name=name,
        deployment_target=DeploymentTarget.CLOUD,
        max_agents=1000,
        message_queue_size=100000,
        auto_scaling=True,
        cloud_config={"provider": provider}
    )
    return NISPlatform(config)


def create_development_platform(name: str = "dev-platform") -> NISPlatform:
    """
    Create a platform for local development.
    
    Args:
        name: Platform name
        
    Returns:
        NISPlatform: Configured for development
    """
    config = PlatformConfig(
        name=name,
        deployment_target=DeploymentTarget.LOCAL,
        max_agents=10,
        message_queue_size=100,
        auto_scaling=False,
        logging_level="DEBUG"
    )
    return NISPlatform(config)
