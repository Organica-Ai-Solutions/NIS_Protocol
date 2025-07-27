"""
NIS Protocol Meta Coordinator

This module implements the MetaProtocolCoordinator, which serves as the central
hub for orchestrating communication between different AI protocols and frameworks.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
import logging
from enum import Enum

from src.core.agent import NISAgent, NISLayer
from src.adapters.base_adapter import BaseAdapter

class ProtocolPriority(Enum):
    """Priority levels for protocol message handling"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class ProtocolMetrics:
    """Metrics for protocol performance monitoring"""
    total_messages: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    average_latency: float = 0.0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None

class MetaProtocolCoordinator:
    """Central coordinator for multi-protocol orchestration.
    
    This class serves as the meta-protocol layer, enabling seamless
    communication between different AI protocols and frameworks while
    maintaining cognitive context and emotional state.
    """
    
    def __init__(self):
        self.protocol_adapters: Dict[str, BaseAdapter] = {}
        self.metrics: Dict[str, ProtocolMetrics] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.emotional_context: Dict[str, Dict[str, float]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger("meta_protocol")
    
    def register_protocol(
        self,
        protocol_name: str,
        adapter: BaseAdapter,
        priority: ProtocolPriority = ProtocolPriority.NORMAL
    ) -> None:
        """Register a new protocol adapter.
        
        Args:
            protocol_name: Name of the protocol
            adapter: Protocol adapter instance
            priority: Priority level for message handling
        """
        self.protocol_adapters[protocol_name] = adapter
        self.metrics[protocol_name] = ProtocolMetrics()
        
        self.logger.info(f"Registered protocol: {protocol_name} with priority {priority}")
    
    async def route_message(
        self,
        source_protocol: str,
        target_protocol: str,
        message: Dict[str, Any],
        conversation_id: Optional[str] = None,
        priority: ProtocolPriority = ProtocolPriority.NORMAL
    ) -> Dict[str, Any]:
        """Route a message between protocols.
        
        Args:
            source_protocol: Source protocol name
            target_protocol: Target protocol name
            message: Message to route
            conversation_id: Optional conversation ID
            priority: Message priority
            
        Returns:
            Response from target protocol
        """
        start_time = datetime.now()
        
        try:
            # Get protocol adapters
            source_adapter = self.protocol_adapters.get(source_protocol)
            target_adapter = self.protocol_adapters.get(target_protocol)
            
            if not source_adapter or not target_adapter:
                raise ValueError(f"Protocol not found: {source_protocol} or {target_protocol}")
            
            # Convert to NIS format
            nis_message = source_adapter.translate_to_nis(message)
            
            # Enrich with emotional context
            if conversation_id in self.emotional_context:
                nis_message["emotional_state"] = self.emotional_context[conversation_id]
            
            # Convert to target format
            target_message = target_adapter.translate_from_nis(nis_message)
            
            # Update metrics
            self._update_metrics(source_protocol, target_protocol, start_time)
            
            return target_message
            
        except Exception as e:
            self.logger.error(f"Error routing message: {str(e)}")
            self._update_error_metrics(source_protocol, str(e))
            raise
    
    def get_protocol_metrics(self, protocol_name: str) -> Optional[ProtocolMetrics]:
        """Get metrics for a specific protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Protocol metrics if found, None otherwise
        """
        return self.metrics.get(protocol_name)
    
    def update_emotional_context(
        self,
        conversation_id: str,
        emotional_state: Dict[str, float]
    ) -> None:
        """Update emotional context for a conversation.
        
        Args:
            conversation_id: Conversation ID
            emotional_state: Dictionary of emotional state values
        """
        self.emotional_context[conversation_id] = emotional_state
        
        # Log significant emotional changes
        if any(value > 0.8 for value in emotional_state.values()):
            self.logger.warning(f"High emotional values detected in conversation {conversation_id}")
    
    def _update_metrics(
        self,
        source_protocol: str,
        target_protocol: str,
        start_time: datetime
    ) -> None:
        """Update success metrics for protocols.
        
        Args:
            source_protocol: Source protocol name
            target_protocol: Target protocol name
            start_time: Start time of the operation
        """
        for protocol in [source_protocol, target_protocol]:
            if protocol in self.metrics:
                metrics = self.metrics[protocol]
                metrics.total_messages += 1
                metrics.successful_translations += 1
                metrics.last_success = datetime.now()
                
                # Update average latency
                latency = (datetime.now() - start_time).total_seconds()
                metrics.average_latency = (
                    (metrics.average_latency * (metrics.total_messages - 1) + latency)
                    / metrics.total_messages
                )
    
    def _update_error_metrics(self, protocol_name: str, error: str) -> None:
        """Update error metrics for a protocol.
        
        Args:
            protocol_name: Name of the protocol
            error: Error message
        """
        if protocol_name in self.metrics:
            metrics = self.metrics[protocol_name]
            metrics.total_messages += 1
            metrics.failed_translations += 1
            metrics.last_error = error 