"""
Communication Layer

This module provides agents and handlers for inter-agent communication using
various protocols including A2A (Agent-to-Agent) and ACP (Agent Computing Protocol).

Key Components:

1. Communication Agent:
   - Handles inter-agent messaging and computation requests
   - Supports multiple communication protocols
   - Maintains message history and computation tracking

2. Protocol Support:
   a. A2A Protocol:
      - Direct agent-to-agent messaging
      - Message queuing and retrieval
      - Metadata handling
   
   b. ACP Protocol:
      - Distributed computation management
      - Computation status tracking
      - Task cancellation support

3. Message Types:
   - Standard messages
   - Computation requests
   - Status updates
   - Control commands

Usage:
    from neural_hierarchy.communication import (
        CommunicationAgent,
        ProtocolType,
        ProtocolConfig,
        CommunicationMessage
    )

    # Initialize with A2A and ACP configs
    agent = CommunicationAgent(
        a2a_config=ProtocolConfig(
            protocol_type=ProtocolType.A2A,
            base_url="https://a2a.example.com",
            api_key="your_a2a_key"
        ),
        acp_config=ProtocolConfig(
            protocol_type=ProtocolType.ACP,
            base_url="https://acp.example.com",
            api_key="your_acp_key"
        )
    )
"""

from .communication_agent import (
    CommunicationAgent,
    ProtocolType,
    ProtocolConfig,
    CommunicationMessage,
    A2AProtocolHandler,
    ACPProtocolHandler
)

__all__ = [
    "CommunicationAgent",
    "ProtocolType",
    "ProtocolConfig",
    "CommunicationMessage",
    "A2AProtocolHandler",
    "ACPProtocolHandler"
] 