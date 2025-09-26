"""
NIS Protocol Message Bus
=======================

Message bus for communication between NIS Protocol agents.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message for inter-agent communication."""
    sender_id: str
    recipient_id: Optional[str]
    content: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    message_type: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    Message bus for communication between NIS Protocol agents.
    
    The message bus provides a pub/sub system for agents to communicate
    with each other asynchronously.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize a new message bus.
        
        Args:
            max_queue_size: Maximum number of messages in the queue
        """
        self.max_queue_size = max_queue_size
        self.queues = {}
        self.subscribers = {}
        self.messages_processed = 0
        logger.info("Message bus initialized")
    
    def register_agent(self, agent_id: str) -> bool:
        """
        Register an agent with the message bus.
        
        Args:
            agent_id: ID of the agent to register
            
        Returns:
            bool: True if registration was successful
        """
        if agent_id in self.queues:
            logger.warning(f"Agent {agent_id} already registered with message bus")
            return False
        
        self.queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
        self.subscribers[agent_id] = set()
        logger.info(f"Agent {agent_id} registered with message bus")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the message bus.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if agent_id not in self.queues:
            logger.warning(f"Agent {agent_id} not registered with message bus")
            return False
        
        # Remove agent queue
        del self.queues[agent_id]
        
        # Remove agent from subscribers lists
        for subscribers in self.subscribers.values():
            if agent_id in subscribers:
                subscribers.remove(agent_id)
        
        # Remove agent's subscribers
        if agent_id in self.subscribers:
            del self.subscribers[agent_id]
        
        logger.info(f"Agent {agent_id} unregistered from message bus")
        return True
    
    async def send(self, message: Message) -> bool:
        """
        Send a message to a recipient.
        
        Args:
            message: The message to send
            
        Returns:
            bool: True if the message was sent successfully
        """
        # Direct message
        if message.recipient_id:
            if message.recipient_id not in self.queues:
                logger.warning(f"Recipient {message.recipient_id} not registered")
                return False
            
            try:
                await self.queues[message.recipient_id].put(message)
                self.messages_processed += 1
                return True
            except asyncio.QueueFull:
                logger.error(f"Queue full for recipient {message.recipient_id}")
                return False
        
        # Broadcast message
        else:
            # Send to subscribers
            if message.sender_id in self.subscribers:
                for subscriber_id in self.subscribers[message.sender_id]:
                    if subscriber_id in self.queues:
                        try:
                            await self.queues[subscriber_id].put(message)
                            self.messages_processed += 1
                        except asyncio.QueueFull:
                            logger.error(f"Queue full for subscriber {subscriber_id}")
            
            return True
    
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message for an agent.
        
        Args:
            agent_id: ID of the agent receiving the message
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Message: The received message, or None if timed out
        """
        if agent_id not in self.queues:
            logger.warning(f"Agent {agent_id} not registered")
            return None
        
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.queues[agent_id].get(), timeout)
            else:
                return await self.queues[agent_id].get()
        except asyncio.TimeoutError:
            return None
    
    def subscribe(self, subscriber_id: str, publisher_id: str) -> bool:
        """
        Subscribe an agent to messages from another agent.
        
        Args:
            subscriber_id: ID of the subscribing agent
            publisher_id: ID of the publishing agent
            
        Returns:
            bool: True if subscription was successful
        """
        if subscriber_id not in self.queues:
            logger.warning(f"Subscriber {subscriber_id} not registered")
            return False
        
        if publisher_id not in self.subscribers:
            logger.warning(f"Publisher {publisher_id} not registered")
            return False
        
        self.subscribers[publisher_id].add(subscriber_id)
        logger.info(f"Agent {subscriber_id} subscribed to {publisher_id}")
        return True
    
    def unsubscribe(self, subscriber_id: str, publisher_id: str) -> bool:
        """
        Unsubscribe an agent from messages from another agent.
        
        Args:
            subscriber_id: ID of the subscribing agent
            publisher_id: ID of the publishing agent
            
        Returns:
            bool: True if unsubscription was successful
        """
        if publisher_id not in self.subscribers:
            logger.warning(f"Publisher {publisher_id} not registered")
            return False
        
        if subscriber_id not in self.subscribers[publisher_id]:
            logger.warning(f"Agent {subscriber_id} not subscribed to {publisher_id}")
            return False
        
        self.subscribers[publisher_id].remove(subscriber_id)
        logger.info(f"Agent {subscriber_id} unsubscribed from {publisher_id}")
        return True
    
    def queue_size(self) -> int:
        """
        Get the total number of messages in all queues.
        
        Returns:
            int: Total number of messages
        """
        return sum(q.qsize() for q in self.queues.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get message bus statistics.
        
        Returns:
            dict: Message bus statistics
        """
        return {
            "registered_agents": len(self.queues),
            "total_queue_size": self.queue_size(),
            "messages_processed": self.messages_processed,
            "max_queue_size": self.max_queue_size,
        }
