"""
NIS Protocol Adapter Bootstrap

This module provides functions for initializing and configuring protocol adapters.
"""

import json
import os
from typing import Dict, Any, Optional

from src.agents.coordination.coordinator_agent import CoordinatorAgent
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.acp_adapter import ACPAdapter
from src.adapters.a2a_adapter import A2AAdapter


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        The configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file is not found
        json.JSONDecodeError: If the configuration file is not valid JSON
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def initialize_adapters(
    config_path: str = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Initialize protocol adapters from configuration.
    
    Args:
        config_path: Path to the configuration file
        config_dict: Configuration dictionary (alternative to config_path)
        
    Returns:
        Dictionary of initialized adapters
        
    Raises:
        ValueError: If neither config_path nor config_dict is provided
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            # Try default location
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "protocol_routing.json"
            )
            if os.path.exists(default_path):
                config_path = default_path
            else:
                raise ValueError("Either config_path or config_dict must be provided")
        
        config = load_config(config_path)
    else:
        config = config_dict
    
    # Initialize adapters
    adapters = {}
    
    # MCP Adapter
    if "mcp" in config:
        adapters["mcp"] = MCPAdapter(config["mcp"])
    
    # ACP Adapter
    if "acp" in config:
        adapters["acp"] = ACPAdapter(config["acp"])
    
    # A2A Adapter
    if "a2a" in config:
        adapters["a2a"] = A2AAdapter(config["a2a"])
    
    return adapters


def configure_coordinator_agent(
    coordinator: CoordinatorAgent,
    config_path: str = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> None:
    """Configure a CoordinatorAgent with protocol adapters.
    
    Args:
        coordinator: The CoordinatorAgent to configure
        config_path: Path to the configuration file
        config_dict: Configuration dictionary (alternative to config_path)
        
    Raises:
        ValueError: If neither config_path nor config_dict is provided
    """
    # Initialize adapters
    adapters = initialize_adapters(config_path, config_dict)
    
    # Register adapters with the coordinator
    for protocol_name, adapter in adapters.items():
        coordinator.register_protocol_adapter(protocol_name, adapter)
    
    # Load routing rules
    if config_path:
        coordinator.load_routing_config(config_path)
    elif config_dict:
        coordinator.routing_rules = config_dict
    else:
        # Try default location
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "protocol_routing.json"
        )
        if os.path.exists(default_path):
            coordinator.load_routing_config(default_path) 