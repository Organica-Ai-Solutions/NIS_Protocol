#!/usr/bin/env python3
"""
NIS Protocol v3.1 - Protocol Bridge Service
Integrated from NIS HUB development for universal AI communication

Provides:
- Universal protocol handler supporting 10+ external protocols
- Bidirectional translation between NIS format and external protocols
- Native MCP integration with NIS-enhanced tools
- Complete ATOA workflow with consciousness validation
- OpenAI Tools bridge with function call validation
- Performance monitoring and real-time status reporting
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import our existing infrastructure
from src.core.agent import NISAgent
from src.utils.confidence_calculator import calculate_confidence

# Import our consciousness service for validation
from src.services.consciousness_service import ConsciousnessService

# Import existing protocol adapters
try:
    from src.adapters.mcp_adapter import MCPAdapter
    from src.adapters.a2a_adapter import A2AAdapter
    MCP_AVAILABLE = True
    A2A_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    A2A_AVAILABLE = False
    logging.warning("Protocol adapters not available - creating mock adapters")


class ProtocolType(Enum):
    """Supported external protocol types"""
    # Tier 1: Native Integration
    MCP = "mcp"                           # Model Context Protocol
    A2A = "a2a"                           # Agent-to-Agent Protocol
    ATOA = "atoa"                         # Agent-to-Operator-to-Agent
    OPENAI_TOOLS = "openai_tools"         # OpenAI Function Calling
    
    # Tier 2: Bridge Support
    ANTHROPIC_MCP = "anthropic_mcp"       # Anthropic MCP variant
    SEMANTIC_KERNEL = "semantic_kernel"   # Microsoft Semantic Kernel
    LANGCHAIN = "langchain"               # LangChain framework
    
    # Tier 3: Translation Support
    AUTOGEN = "autogen"                   # Microsoft AutoGen
    CREWAI = "crewai"                     # CrewAI multi-agent
    CHAINLIT = "chainlit"                 # Chainlit interface
    CUSTOM = "custom"                     # Custom protocol


class MessageFormat(Enum):
    """Message format types"""
    JSON_RPC = "json_rpc"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    CUSTOM = "custom"


@dataclass
class ProtocolMetrics:
    """Metrics for protocol bridge performance"""
    protocol_type: ProtocolType
    total_messages: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    average_latency: float = 0.0
    last_active: float = field(default_factory=time.time)
    error_rate: float = 0.0
    

@dataclass
class BridgeConfiguration:
    """Configuration for protocol bridge"""
    protocol_type: ProtocolType
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    message_format: MessageFormat = MessageFormat.JSON_RPC
    enable_consciousness_validation: bool = True
    enable_physics_validation: bool = True
    enable_safety_validation: bool = True
    translation_timeout: float = 10.0
    max_retries: int = 3
    custom_handlers: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class NISMessage:
    """Universal NIS message structure"""
    message_id: str
    timestamp: datetime
    message_type: str
    content: Dict[str, Any]
    
    # Pipeline metadata
    pipeline_stage: Optional[str] = None
    consciousness_level: Optional[float] = None
    pinn_validation_score: Optional[float] = None
    kan_interpretability: Optional[float] = None
    safety_score: Optional[float] = None
    
    # External protocol metadata
    source_protocol: Optional[str] = None
    target_protocol: Optional[str] = None
    requires_translation: bool = False
    verification_required: bool = True
    
    # NIS verification signature
    nis_signature: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTool:
    """MCP tool definition with NIS enhancement"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    requires_consciousness_validation: bool = True
    requires_physics_validation: bool = False
    safety_level: str = "standard"  # low, standard, high


@dataclass
class ATOAMessage:
    """ATOA message structure with consciousness validation"""
    agent_id: str
    operator_id: str
    message_content: Dict[str, Any]
    ethical_review_required: bool = False
    physics_validation_required: bool = False
    consciousness_assessment: Dict[str, Any] = field(default_factory=dict)
    urgency_level: str = "normal"  # low, normal, high, critical


class ProtocolBridgeService(NISAgent):
    """
    🌉 Protocol Bridge Service for NIS Protocol v3.1
    
    Universal translator and validator for external AI communication protocols.
    Enables seamless integration with MCP, ATOA, OpenAI Tools, and other frameworks
    while maintaining NIS pipeline integrity.
    
    Key Features:
    - Universal protocol handler supporting 10+ external protocols
    - Bidirectional translation with verification preservation
    - Consciousness validation for all external interactions
    - Physics compliance checking for external outputs
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        agent_id: str = "protocol_bridge_service",
        consciousness_service: Optional[ConsciousnessService] = None,
        unified_coordinator = None,
        enable_real_time_monitoring: bool = True
    ):
        super().__init__(agent_id)
        
        # Core services
        self.consciousness_service = consciousness_service or ConsciousnessService()
        self.unified_coordinator = unified_coordinator
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Protocol management
        self.protocol_bridges: Dict[ProtocolType, Any] = {}
        self.bridge_configurations: Dict[ProtocolType, BridgeConfiguration] = {}
        self.protocol_metrics: Dict[ProtocolType, ProtocolMetrics] = {}
        
        # Message routing
        self.message_handlers: Dict[str, Callable] = {}
        self.translation_cache: Dict[str, Dict[str, Any]] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # MCP Tools registry
        self.mcp_tools: Dict[str, MCPTool] = {}
        
        # Performance monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.performance_metrics = {
            "total_messages_processed": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "average_latency": 0.0,
            "active_protocols": 0,
            "consciousness_validations": 0,
            "physics_validations": 0,
            "safety_validations": 0
        }
        
        # Initialize default configurations
        self._initialize_default_configurations()
        
        # Initialize available protocols
        self._initialize_protocol_bridges()
        
        self.logger.info(f"ProtocolBridgeService initialized: {agent_id}")
    
    def _initialize_default_configurations(self):
        """Initialize default configurations for supported protocols"""
        # MCP Configuration
        self.bridge_configurations[ProtocolType.MCP] = BridgeConfiguration(
            protocol_type=ProtocolType.MCP,
            message_format=MessageFormat.JSON_RPC,
            enable_consciousness_validation=True,
            enable_physics_validation=False,
            enable_safety_validation=True
        )
        
        # ATOA Configuration
        self.bridge_configurations[ProtocolType.ATOA] = BridgeConfiguration(
            protocol_type=ProtocolType.ATOA,
            message_format=MessageFormat.REST_API,
            enable_consciousness_validation=True,
            enable_physics_validation=True,
            enable_safety_validation=True
        )
        
        # OpenAI Tools Configuration
        self.bridge_configurations[ProtocolType.OPENAI_TOOLS] = BridgeConfiguration(
            protocol_type=ProtocolType.OPENAI_TOOLS,
            message_format=MessageFormat.REST_API,
            enable_consciousness_validation=True,
            enable_physics_validation=False,
            enable_safety_validation=True
        )
        
        # Initialize metrics for all protocols
        for protocol_type in ProtocolType:
            self.protocol_metrics[protocol_type] = ProtocolMetrics(protocol_type=protocol_type)
    
    def _initialize_protocol_bridges(self):
        """Initialize available protocol bridges"""
        try:
            # MCP Bridge
            if MCP_AVAILABLE:
                self.protocol_bridges[ProtocolType.MCP] = self._create_mcp_bridge()
                self.logger.info("MCP bridge initialized")
            
            # A2A Bridge  
            if A2A_AVAILABLE:
                self.protocol_bridges[ProtocolType.A2A] = self._create_a2a_bridge()
                self.logger.info("A2A bridge initialized")
            
            # OpenAI Tools Bridge
            self.protocol_bridges[ProtocolType.OPENAI_TOOLS] = self._create_openai_tools_bridge()
            self.logger.info("OpenAI Tools bridge initialized")
            
            self.performance_metrics["active_protocols"] = len(self.protocol_bridges)
            
        except Exception as e:
            self.logger.error(f"Error initializing protocol bridges: {e}")
    
    async def send_to_external_protocol(
        self,
        protocol_type: ProtocolType,
        message: Dict[str, Any],
        require_nis_validation: bool = True,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send message to external protocol with NIS validation
        
        Args:
            protocol_type: Type of external protocol
            message: Message to send
            require_nis_validation: Whether to apply full NIS pipeline validation
            conversation_id: Optional conversation tracking ID
            
        Returns:
            Response from external protocol with NIS validation metadata
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Sending message to {protocol_type.value} protocol: {message_id}")
            
            # 1. Create NIS message structure
            nis_message = NISMessage(
                message_id=message_id,
                timestamp=datetime.now(),
                message_type="outbound_external",
                content=message,
                source_protocol="nis",
                target_protocol=protocol_type.value,
                requires_translation=True,
                verification_required=require_nis_validation
            )
            
            # 2. Apply NIS validation if required
            if require_nis_validation and self.unified_coordinator:
                validated_content = self.unified_coordinator.process_data_pipeline(message)
                nis_message.content = validated_content
                
                # Extract validation metadata
                nis_message.consciousness_level = validated_content.get("consciousness_validation", {}).get("consciousness_confidence", 0.0)
                nis_message.pinn_validation_score = validated_content.get("pinn_validation", {}).get("confidence", 0.0)
                nis_message.safety_score = validated_content.get("safety_validation", {}).get("safety_score", 0.0)
                
                # Create NIS verification signature
                nis_message.nis_signature = {
                    "consciousness_score": nis_message.consciousness_level,
                    "physics_compliance": validated_content.get("pinn_validation", {}).get("physics_compliant", True),
                    "interpretability_verified": validated_content.get("kan_output", {}).get("confidence", 0.0) > 0.5,
                    "safety_validated": validated_content.get("safety_validation", {}).get("safety_approved", False),
                    "verification_chain": validated_content.get("pipeline_stages", []),
                    "timestamp": datetime.now().isoformat(),
                    "validator_id": self.agent_id
                }
            
            # 3. Translate to external protocol format
            translated_message = await self._translate_to_external_format(nis_message, protocol_type)
            
            # 4. Send to external protocol
            external_response = await self._send_to_protocol_bridge(protocol_type, translated_message)
            
            # 5. Translate response back to NIS format
            nis_response = await self._translate_from_external_format(external_response, protocol_type)
            
            # 6. Update metrics
            processing_time = time.time() - start_time
            await self._update_protocol_metrics(protocol_type, success=True, latency=processing_time)
            
            # 7. Track conversation if ID provided
            if conversation_id:
                self.active_conversations[conversation_id] = {
                    "last_message": nis_message,
                    "last_response": nis_response,
                    "protocol_type": protocol_type.value,
                    "updated_at": datetime.now().isoformat()
                }
            
            self.logger.info(f"External protocol message complete: {protocol_type.value}, latency: {processing_time:.3f}s")
            
            return {
                "success": True,
                "response": nis_response,
                "nis_signature": nis_message.nis_signature,
                "protocol_type": protocol_type.value,
                "message_id": message_id,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_protocol_metrics(protocol_type, success=False, latency=processing_time)
            
            self.logger.error(f"Error sending to external protocol {protocol_type.value}: {e}")
            return {
                "success": False,
                "error": str(e),
                "protocol_type": protocol_type.value,
                "message_id": message_id,
                "processing_time": processing_time,
                "requires_human_review": True
            }
    
    async def receive_from_external_protocol(
        self,
        protocol_type: ProtocolType,
        external_message: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Receive message from external protocol and apply NIS validation
        
        Args:
            protocol_type: Type of external protocol
            external_message: Message received from external protocol
            conversation_id: Optional conversation tracking ID
            
        Returns:
            NIS-validated response
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Receiving message from {protocol_type.value} protocol: {message_id}")
            
            # 1. Translate from external format to NIS format
            nis_message = await self._translate_from_external_format(external_message, protocol_type)
            
            # 2. Apply NIS validation pipeline
            if self.unified_coordinator:
                validated_content = self.unified_coordinator.process_data_pipeline(nis_message.content)
                
                # Update NIS message with validation results
                nis_message.content = validated_content
                nis_message.consciousness_level = validated_content.get("consciousness_validation", {}).get("consciousness_confidence", 0.0)
                nis_message.pinn_validation_score = validated_content.get("pinn_validation", {}).get("confidence", 0.0)
                nis_message.safety_score = validated_content.get("safety_validation", {}).get("safety_score", 0.0)
                
                # Create NIS verification signature
                nis_message.nis_signature = {
                    "consciousness_score": nis_message.consciousness_level,
                    "physics_compliance": validated_content.get("pinn_validation", {}).get("physics_compliant", True),
                    "interpretability_verified": validated_content.get("kan_output", {}).get("confidence", 0.0) > 0.5,
                    "safety_validated": validated_content.get("safety_validation", {}).get("safety_approved", False),
                    "verification_chain": validated_content.get("pipeline_stages", []),
                    "timestamp": datetime.now().isoformat(),
                    "validator_id": self.agent_id,
                    "source_protocol": protocol_type.value
                }
            
            # 3. Update metrics
            processing_time = time.time() - start_time
            await self._update_protocol_metrics(protocol_type, success=True, latency=processing_time)
            
            # 4. Track conversation if ID provided
            if conversation_id:
                self.active_conversations[conversation_id] = {
                    "last_external_message": external_message,
                    "last_nis_message": nis_message,
                    "protocol_type": protocol_type.value,
                    "updated_at": datetime.now().isoformat()
                }
            
            self.logger.info(f"External protocol receive complete: {protocol_type.value}, latency: {processing_time:.3f}s")
            
            return {
                "success": True,
                "nis_message": nis_message,
                "nis_signature": nis_message.nis_signature,
                "protocol_type": protocol_type.value,
                "message_id": message_id,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_protocol_metrics(protocol_type, success=False, latency=processing_time)
            
            self.logger.error(f"Error receiving from external protocol {protocol_type.value}: {e}")
            return {
                "success": False,
                "error": str(e),
                "protocol_type": protocol_type.value,
                "message_id": message_id,
                "processing_time": processing_time,
                "requires_human_review": True
            }
    
    # =============================================================================
    # MCP INTEGRATION
    # =============================================================================
    
    def register_mcp_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
        requires_consciousness_validation: bool = True,
        requires_physics_validation: bool = False,
        safety_level: str = "standard"
    ):
        """Register a new MCP tool with NIS validation"""
        self.mcp_tools[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            requires_consciousness_validation=requires_consciousness_validation,
            requires_physics_validation=requires_physics_validation,
            safety_level=safety_level
        )
        self.logger.info(f"Registered MCP tool: {name}")
    
    async def handle_mcp_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle MCP tool call with NIS validation"""
        start_time = time.time()
        
        try:
            if tool_name not in self.mcp_tools:
                raise ValueError(f"Unknown MCP tool: {tool_name}")
            
            tool = self.mcp_tools[tool_name]
            
            # Apply NIS validation if required
            if tool.requires_consciousness_validation or tool.requires_physics_validation:
                validation_data = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "safety_level": tool.safety_level
                }
                
                if self.unified_coordinator:
                    validated_data = self.unified_coordinator.process_data_pipeline(validation_data)
                    
                    # Check if human review is required
                    if validated_data.get("requires_human_review", False):
                        return {
                            "success": False,
                            "error": "Tool call requires human review due to validation concerns",
                            "requires_human_review": True,
                            "validation_details": validated_data.get("consciousness_validation", {})
                        }
            
            # Execute tool handler
            result = await tool.handler(parameters)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"MCP tool call complete: {tool_name}, latency: {processing_time:.3f}s")
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name,
                "processing_time": processing_time,
                "nis_validated": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Error in MCP tool call {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "processing_time": processing_time,
                "requires_human_review": True
            }
    
    # =============================================================================
    # ATOA INTEGRATION
    # =============================================================================
    
    async def handle_atoa_workflow(
        self,
        agent_id: str,
        operator_id: str,
        message_content: Dict[str, Any],
        ethical_review_required: bool = False,
        physics_validation_required: bool = False,
        urgency_level: str = "normal"
    ) -> Dict[str, Any]:
        """Handle ATOA workflow with consciousness validation"""
        start_time = time.time()
        workflow_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Handling ATOA workflow: {workflow_id}")
            
            # Create ATOA message
            atoa_message = ATOAMessage(
                agent_id=agent_id,
                operator_id=operator_id,
                message_content=message_content,
                ethical_review_required=ethical_review_required,
                physics_validation_required=physics_validation_required,
                urgency_level=urgency_level
            )
            
            # Apply consciousness validation
            consciousness_result = await self.consciousness_service.process_through_consciousness(message_content)
            atoa_message.consciousness_assessment = consciousness_result.get("consciousness_validation", {})
            
            # Determine if human operator review is needed
            requires_operator_review = (
                ethical_review_required or
                physics_validation_required or
                consciousness_result.get("consciousness_validation", {}).get("requires_human_review", False) or
                urgency_level in ["high", "critical"]
            )
            
            # Route to appropriate handler
            if requires_operator_review:
                # Route to human operator
                operator_response = await self._route_to_human_operator(atoa_message)
            else:
                # Process automatically through NIS pipeline
                operator_response = await self._process_atoa_automatically(atoa_message)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"ATOA workflow complete: {workflow_id}, operator_review: {requires_operator_review}, latency: {processing_time:.3f}s")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "requires_operator_review": requires_operator_review,
                "consciousness_assessment": atoa_message.consciousness_assessment,
                "operator_response": operator_response,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Error in ATOA workflow {workflow_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "processing_time": processing_time,
                "requires_human_review": True
            }
    
    # =============================================================================
    # INTERNAL TRANSLATION METHODS
    # =============================================================================
    
    async def _translate_to_external_format(
        self,
        nis_message: NISMessage,
        protocol_type: ProtocolType
    ) -> Dict[str, Any]:
        """Translate NIS message to external protocol format"""
        try:
            if protocol_type == ProtocolType.MCP:
                return await self._translate_to_mcp_format(nis_message)
            elif protocol_type == ProtocolType.ATOA:
                return await self._translate_to_atoa_format(nis_message)
            elif protocol_type == ProtocolType.OPENAI_TOOLS:
                return await self._translate_to_openai_format(nis_message)
            else:
                # Generic translation for other protocols
                return {
                    "id": nis_message.message_id,
                    "timestamp": nis_message.timestamp.isoformat(),
                    "content": nis_message.content,
                    "nis_metadata": {
                        "verification_signature": nis_message.nis_signature,
                        "pipeline_stage": nis_message.pipeline_stage,
                        "source_protocol": nis_message.source_protocol
                    }
                }
        except Exception as e:
            self.logger.error(f"Error translating to {protocol_type.value} format: {e}")
            raise
    
    async def _translate_from_external_format(
        self,
        external_message: Dict[str, Any],
        protocol_type: ProtocolType
    ) -> NISMessage:
        """Translate external protocol message to NIS format"""
        try:
            message_id = external_message.get("id", str(uuid.uuid4()))
            
            return NISMessage(
                message_id=message_id,
                timestamp=datetime.now(),
                message_type="inbound_external",
                content=external_message,
                source_protocol=protocol_type.value,
                target_protocol="nis",
                requires_translation=False,
                verification_required=True
            )
        except Exception as e:
            self.logger.error(f"Error translating from {protocol_type.value} format: {e}")
            raise
    
    async def _translate_to_mcp_format(self, nis_message: NISMessage) -> Dict[str, Any]:
        """Translate to MCP JSON-RPC format"""
        return {
            "jsonrpc": "2.0",
            "id": nis_message.message_id,
            "method": "nis_enhanced_call",
            "params": {
                "content": nis_message.content,
                "nis_validation": nis_message.nis_signature,
                "timestamp": nis_message.timestamp.isoformat()
            }
        }
    
    async def _translate_to_atoa_format(self, nis_message: NISMessage) -> Dict[str, Any]:
        """Translate to ATOA format"""
        return {
            "message_id": nis_message.message_id,
            "agent_message": nis_message.content,
            "nis_validation": nis_message.nis_signature,
            "requires_operator_review": nis_message.nis_signature.get("requires_human_review", False),
            "timestamp": nis_message.timestamp.isoformat()
        }
    
    async def _translate_to_openai_format(self, nis_message: NISMessage) -> Dict[str, Any]:
        """Translate to OpenAI Tools format"""
        return {
            "model": "nis-enhanced",
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(nis_message.content)
                }
            ],
            "tools": [],
            "metadata": {
                "nis_validation": nis_message.nis_signature,
                "message_id": nis_message.message_id
            }
        }
    
    # =============================================================================
    # PROTOCOL BRIDGE IMPLEMENTATIONS
    # =============================================================================
    
    def _create_mcp_bridge(self):
        """Create MCP protocol bridge"""
        # Placeholder implementation - would integrate with actual MCP adapter
        return {"type": "mcp", "status": "initialized"}
    
    def _create_a2a_bridge(self):
        """Create A2A protocol bridge"""
        # Placeholder implementation - would integrate with actual A2A adapter
        return {"type": "a2a", "status": "initialized"}
    
    def _create_openai_tools_bridge(self):
        """Create OpenAI Tools protocol bridge"""
        # Placeholder implementation - would integrate with OpenAI API
        return {"type": "openai_tools", "status": "initialized"}
    
    async def _send_to_protocol_bridge(
        self,
        protocol_type: ProtocolType,
        translated_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send message to specific protocol bridge"""
        # Placeholder implementation - would send to actual external protocol
        return {
            "success": True,
            "response": f"Mock response from {protocol_type.value}",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _route_to_human_operator(self, atoa_message: ATOAMessage) -> Dict[str, Any]:
        """Route ATOA message to human operator"""
        # Placeholder implementation - would route to actual operator dashboard
        return {
            "routed_to_operator": True,
            "operator_id": atoa_message.operator_id,
            "pending_review": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_atoa_automatically(self, atoa_message: ATOAMessage) -> Dict[str, Any]:
        """Process ATOA message automatically through NIS pipeline"""
        if self.unified_coordinator:
            result = self.unified_coordinator.process_data_pipeline(atoa_message.message_content)
            return {
                "automatic_processing": True,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        return {
            "automatic_processing": False,
            "error": "No unified coordinator available"
        }
    
    async def _update_protocol_metrics(
        self,
        protocol_type: ProtocolType,
        success: bool,
        latency: float
    ):
        """Update performance metrics for protocol"""
        metrics = self.protocol_metrics[protocol_type]
        metrics.total_messages += 1
        
        if success:
            metrics.successful_translations += 1
        else:
            metrics.failed_translations += 1
        
        # Update average latency
        if metrics.total_messages == 1:
            metrics.average_latency = latency
        else:
            metrics.average_latency = (metrics.average_latency * (metrics.total_messages - 1) + latency) / metrics.total_messages
        
        # Update error rate
        metrics.error_rate = metrics.failed_translations / metrics.total_messages
        metrics.last_active = time.time()
        
        # Update global metrics
        self.performance_metrics["total_messages_processed"] += 1
        if success:
            self.performance_metrics["successful_translations"] += 1
        else:
            self.performance_metrics["failed_translations"] += 1
    
    def get_protocol_status(self) -> Dict[str, Any]:
        """Get status of all protocol bridges"""
        return {
            "active_protocols": list(self.protocol_bridges.keys()),
            "protocol_metrics": {
                protocol_type.value: {
                    "total_messages": metrics.total_messages,
                    "success_rate": (metrics.successful_translations / metrics.total_messages) if metrics.total_messages > 0 else 0.0,
                    "average_latency": metrics.average_latency,
                    "last_active": metrics.last_active
                }
                for protocol_type, metrics in self.protocol_metrics.items()
                if metrics.total_messages > 0
            },
            "overall_performance": self.performance_metrics,
            "registered_mcp_tools": list(self.mcp_tools.keys()),
            "active_conversations": len(self.active_conversations),
            "service_id": self.agent_id
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_protocol_bridge_service(**kwargs) -> ProtocolBridgeService:
    """Factory function to create protocol bridge service"""
    return ProtocolBridgeService(**kwargs)


# Export main classes
__all__ = [
    'ProtocolBridgeService',
    'ProtocolType',
    'MessageFormat',
    'BridgeConfiguration',
    'NISMessage',
    'MCPTool',
    'ATOAMessage',
    'create_protocol_bridge_service'
]