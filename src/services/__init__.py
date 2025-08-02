#!/usr/bin/env python3
"""
NIS Protocol v3.1 - Services Module
Provides consciousness, protocol bridge, and other enhanced services for universal AI coordination
"""

from .consciousness_service import (
    ConsciousnessService,
    ConsciousnessLevel,
    BiasType,
    EthicalFramework,
    ConsciousnessMetrics,
    BiasDetectionResult,
    EthicalAnalysis,
    create_consciousness_service
)

from .protocol_bridge_service import (
    ProtocolBridgeService,
    ProtocolType,
    MessageFormat,
    BridgeConfiguration,
    NISMessage,
    MCPTool,
    ATOAMessage,
    create_protocol_bridge_service
)

__all__ = [
    # Consciousness Service
    'ConsciousnessService',
    'ConsciousnessLevel',
    'BiasType',
    'EthicalFramework',
    'ConsciousnessMetrics',
    'BiasDetectionResult',
    'EthicalAnalysis',
    'create_consciousness_service',
    
    # Protocol Bridge Service
    'ProtocolBridgeService',
    'ProtocolType',
    'MessageFormat',
    'BridgeConfiguration',
    'NISMessage',
    'MCPTool',
    'ATOAMessage',
    'create_protocol_bridge_service'
]