"""
NIS Protocol Core Messaging

This module defines the infrastructure-agnostic message types and structures
used for communication between all components of the NIS Protocol.

By defining these here, we decouple the core agent logic from any specific
messaging technology (e.g., Kafka, RabbitMQ).
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

class MessageType(Enum):
    """Defines the high-level type of a message."""
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    RESPONSE = "response"
    SYSTEM = "system"
    LOG = "log"

class MessagePriority(Enum):
    """Defines the priority of a message for queuing and processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class NISMessage:
    """A standardized message structure for the NIS Protocol."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "unknown"
    destination: str = "broadcast"
    message_type: MessageType = MessageType.EVENT
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the message to a dictionary."""
        return {
            "message_id": self.message_id,
            "source": self.source,
            "destination": self.destination,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NISMessage":
        """Deserializes a dictionary into a NISMessage object."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            source=data.get("source", "unknown"),
            destination=data.get("destination", "broadcast"),
            message_type=MessageType(data.get("message_type", "event")),
            priority=MessagePriority(data.get("priority", 2)),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        ) 