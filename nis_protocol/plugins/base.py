#!/usr/bin/env python3
"""
Base Plugin System for NIS Protocol

Provides the foundation for domain-specific plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    domain: str  # drone, auto, city, space, etc.
    description: str
    author: str = "Organica AI Solutions"
    requires: List[str] = None  # Required dependencies
    
    def __post_init__(self):
        if self.requires is None:
            self.requires = []


class BasePlugin(ABC):
    """
    Base class for all NIS Protocol plugins.
    
    Domain-specific plugins should inherit from this class and implement:
    - metadata property
    - initialize() method
    - process_request() method
    - get_capabilities() method
    
    Example::
    
        class DronePlugin(BasePlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="NIS-DRONE",
                    version="1.0.0",
                    domain="drone",
                    description="Drone/UAV integration"
                )
            
            async def initialize(self, config: dict):
                # Setup sensors, actuators, etc.
                pass
            
            async def process_request(self, request: dict) -> dict:
                # Handle drone-specific requests
                pass
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration dictionary
        """
        self.config = config or {}
        self._initialized = False
        self._capabilities = []
        self._custom_intents = []
        self._custom_tools = {}
        
        logger.info(f"🔌 Plugin created: {self.metadata.name}")
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Additional configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a domain-specific request.
        
        Args:
            request: Request dictionary with keys:
                - intent: The detected intent
                - message: Original user message
                - context: Additional context
                
        Returns:
            Result dictionary with:
                - success: bool
                - data: Any response data
                - message: Human-readable message
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this plugin provides.
        
        Returns:
            List of capability names (e.g., ['navigation', 'obstacle_avoidance'])
        """
        pass
    
    def register_custom_intent(self, intent_name: str, keywords: List[str]):
        """
        Register a custom intent for this domain.
        
        Args:
            intent_name: Name of the intent
            keywords: Keywords that trigger this intent
        """
        self._custom_intents.append({
            "name": intent_name,
            "keywords": keywords,
            "plugin": self.metadata.name
        })
        logger.info(f"📝 Registered custom intent: {intent_name}")
    
    def register_custom_tool(self, tool_name: str, tool_function: Callable):
        """
        Register a custom tool for this domain.
        
        Args:
            tool_name: Name of the tool
            tool_function: Async function to execute
        """
        self._custom_tools[tool_name] = tool_function
        logger.info(f"🔧 Registered custom tool: {tool_name}")
    
    def get_custom_intents(self) -> List[Dict[str, Any]]:
        """Get all custom intents registered by this plugin"""
        return self._custom_intents
    
    def get_custom_tools(self) -> Dict[str, Callable]:
        """Get all custom tools registered by this plugin"""
        return self._custom_tools
    
    async def shutdown(self):
        """Clean up plugin resources"""
        logger.info(f"🔌 Shutting down plugin: {self.metadata.name}")
        self._initialized = False


class PluginManager:
    """
    Manages all registered plugins for NIS Protocol.
    
    Example::
    
        manager = PluginManager()
        
        # Register plugins
        drone_plugin = DronePlugin(config={...})
        await manager.register_plugin(drone_plugin)
        
        # Process request
        result = await manager.process_request("drone", request_data)
        
        # Get all capabilities
        caps = manager.get_all_capabilities()
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.domain_to_plugin: Dict[str, str] = {}
        logger.info("🔌 Plugin Manager initialized")
    
    async def register_plugin(self, plugin: BasePlugin) -> bool:
        """
        Register a plugin with the manager.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful
        """
        try:
            # Initialize plugin
            if not plugin._initialized:
                await plugin.initialize(plugin.config)
                plugin._initialized = True
            
            # Register plugin
            plugin_name = plugin.metadata.name
            domain = plugin.metadata.domain
            
            self.plugins[plugin_name] = plugin
            self.domain_to_plugin[domain] = plugin_name
            
            logger.info(f"✅ Plugin registered: {plugin_name} (domain: {domain})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register plugin {plugin.metadata.name}: {e}")
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if unregistration successful
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            await plugin.shutdown()
            
            # Remove from registries
            del self.plugins[plugin_name]
            domain = plugin.metadata.domain
            if domain in self.domain_to_plugin:
                del self.domain_to_plugin[domain]
            
            logger.info(f"✅ Plugin unregistered: {plugin_name}")
            return True
        
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name"""
        return self.plugins.get(plugin_name)
    
    def get_plugin_by_domain(self, domain: str) -> Optional[BasePlugin]:
        """Get a plugin by domain"""
        plugin_name = self.domain_to_plugin.get(domain)
        if plugin_name:
            return self.plugins.get(plugin_name)
        return None
    
    def get_all_plugins(self) -> List[BasePlugin]:
        """Get all registered plugins"""
        return list(self.plugins.values())
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities from all plugins"""
        capabilities = {}
        for plugin_name, plugin in self.plugins.items():
            capabilities[plugin_name] = plugin.get_capabilities()
        return capabilities
    
    def get_all_custom_intents(self) -> List[Dict[str, Any]]:
        """Get all custom intents from all plugins"""
        intents = []
        for plugin in self.plugins.values():
            intents.extend(plugin.get_custom_intents())
        return intents
    
    def get_all_custom_tools(self) -> Dict[str, Callable]:
        """Get all custom tools from all plugins"""
        tools = {}
        for plugin in self.plugins.values():
            tools.update(plugin.get_custom_tools())
        return tools
    
    async def process_request(
        self,
        domain: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a request using the appropriate domain plugin.
        
        Args:
            domain: Domain name (drone, auto, city, etc.)
            request: Request data
            
        Returns:
            Response from the plugin
        """
        plugin = self.get_plugin_by_domain(domain)
        
        if not plugin:
            return {
                "success": False,
                "error": f"No plugin registered for domain: {domain}"
            }
        
        try:
            result = await plugin.process_request(request)
            return result
        except Exception as e:
            logger.error(f"❌ Plugin error ({domain}): {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def is_domain_supported(self, domain: str) -> bool:
        """Check if a domain is supported"""
        return domain in self.domain_to_plugin
    
    def get_supported_domains(self) -> List[str]:
        """Get list of all supported domains"""
        return list(self.domain_to_plugin.keys())
    
    async def shutdown_all(self):
        """Shutdown all plugins"""
        logger.info("🔌 Shutting down all plugins...")
        for plugin in self.plugins.values():
            await plugin.shutdown()
        self.plugins.clear()
        self.domain_to_plugin.clear()
        logger.info("✅ All plugins shut down")

