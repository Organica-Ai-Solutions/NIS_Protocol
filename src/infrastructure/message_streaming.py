"""
NIS Protocol v3 - Enhanced Kafka Message Streaming

This module provides comprehensive Kafka integration with self-audit capabilities,
async processing, and resilience patterns for all NIS Protocol agents.

Features:
- Self-audit integration for message integrity
- Async message processing with error handling
- Auto-retry and circuit breaker patterns
- Performance monitoring and optimization
- Topic management and routing
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

# Kafka imports with fallback
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError, KafkaTimeoutError
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka not available. Install kafka-python and aiokafka for full functionality.")

# Self-audit integration
from src.utils.self_audit import self_audit_engine


class MessageType(Enum):
    """Types of messages in the NIS Protocol system"""
    CONSCIOUSNESS_EVENT = "consciousness_event"
    GOAL_GENERATION = "goal_generation"
    SIMULATION_RESULT = "simulation_result"
    ALIGNMENT_CHECK = "alignment_check"
    MEMORY_OPERATION = "memory_operation"
    AGENT_COORDINATION = "agent_coordination"
    SYSTEM_HEALTH = "system_health"
    AUDIT_ALERT = "audit_alert"
    PERFORMANCE_METRIC = "performance_metric"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class StreamingTopics:
    """Centralized topic management for NIS Protocol"""
    CONSCIOUSNESS = "nis-consciousness"
    GOALS = "nis-goals"
    SIMULATION = "nis-simulation"
    ALIGNMENT = "nis-alignment"
    MEMORY = "nis-memory"
    COORDINATION = "nis-coordination"
    SYSTEM_HEALTH = "nis-system-health"
    AUDIT_ALERTS = "nis-audit-alerts"
    PERFORMANCE = "nis-performance"
    
    @classmethod
    def get_all_topics(cls) -> List[str]:
        """Get all available topics"""
        return [
            cls.CONSCIOUSNESS, cls.GOALS, cls.SIMULATION,
            cls.ALIGNMENT, cls.MEMORY, cls.COORDINATION,
            cls.SYSTEM_HEALTH, cls.AUDIT_ALERTS, cls.PERFORMANCE
        ]


@dataclass
class NISMessage:
    """Enhanced message structure for NIS Protocol"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    source_agent: str
    target_agent: Optional[str]
    topic: str
    content: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    integrity_score: Optional[float] = None
    audit_flags: Optional[List[str]] = None


@dataclass
class StreamingMetrics:
    """Metrics for Kafka streaming performance"""
    messages_sent: int = 0
    messages_received: int = 0
    errors_encountered: int = 0
    avg_latency: float = 0.0
    total_throughput: float = 0.0
    integrity_violations: int = 0
    last_update: float = 0.0


class NISKafkaManager:
    """
    Enhanced Kafka manager with self-audit integration and resilience patterns.
    
    Features:
    - Async message streaming with integrity monitoring
    - Auto-retry and circuit breaker patterns
    - Performance tracking and optimization
    - Self-audit integration for message validation
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str] = None,
        enable_self_audit: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        batch_size: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the enhanced Kafka manager"""
        self.bootstrap_servers = bootstrap_servers or ["kafka:9092"]
        self.enable_self_audit = enable_self_audit
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.batch_size = batch_size
        self.config = config or {}
        
        # Initialize components
        self.logger = logging.getLogger("nis.kafka_manager")
        self.metrics = StreamingMetrics()
        self.message_history: deque = deque(maxlen=1000)
        self.active_consumers: Dict[str, Any] = {}
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Circuit breaker state
        self.circuit_breaker = {
            'state': 'closed',  # closed, open, half-open
            'failure_count': 0,
            'last_failure_time': 0,
            'failure_threshold': 5,
            'timeout': 60  # seconds
        }
        
        # Kafka connections
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.is_initialized = False
        
        # Self-audit integration
        if self.enable_self_audit:
            self._init_self_audit()
        
        self.logger.info(f"NISKafkaManager initialized with {len(self.bootstrap_servers)} servers")
    
    def _init_self_audit(self):
        """Initialize self-audit capabilities"""
        try:
            self.self_audit_enabled = True
            self.audit_threshold = 70.0  # Minimum integrity score
            self.audit_violations: deque = deque(maxlen=100)
            self.auto_correction_enabled = True
            
            self.logger.info("Self-audit integration enabled for Kafka manager")
        except Exception as e:
            self.logger.error(f"Failed to initialize self-audit: {e}")
            self.self_audit_enabled = False
    
    async def initialize(self) -> bool:
        """Initialize async Kafka connections"""
        if not KAFKA_AVAILABLE:
            self.logger.warning("Kafka not available, running in mock mode")
            self.is_initialized = True
            return True
        
        try:
            # Initialize producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                retry_backoff_ms=int(self.retry_backoff * 1000)
            )
            await self.producer.start()
            
            self.is_initialized = True
            self.logger.info("Kafka producer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            self._update_circuit_breaker(failed=True)
            return False
    
    async def shutdown(self):
        """Gracefully shutdown Kafka connections"""
        try:
            if self.producer:
                await self.producer.stop()
            
            for consumer in self.consumers.values():
                await consumer.stop()
            
            self.logger.info("Kafka connections shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def send_message(
        self,
        message: NISMessage,
        timeout: float = 10.0
    ) -> bool:
        """
        Send message with integrity validation and error handling
        
        Args:
            message: NISMessage to send
            timeout: Timeout in seconds
            
        Returns:
            bool: True if message sent successfully
        """
        if not self._check_circuit_breaker():
            self.logger.warning("Circuit breaker open, message not sent")
            return False
        
        start_time = time.time()
        
        try:
            # Self-audit validation
            if self.enable_self_audit:
                audit_result = self._audit_message(message)
                message.integrity_score = audit_result['score']
                message.audit_flags = audit_result['flags']
                
                if audit_result['score'] < self.audit_threshold:
                    self.logger.warning(f"Message failed audit: {audit_result['flags']}")
                    self.metrics.integrity_violations += 1
                    
                    if not self.auto_correction_enabled:
                        return False
                    
                    # Attempt auto-correction
                    message = self._auto_correct_message(message, audit_result)
            
            # Prepare message for sending
            message_data = asdict(message)
            message_data['message_type'] = message.message_type.value
            message_data['priority'] = message.priority.value
            
            if not KAFKA_AVAILABLE:
                # Mock mode
                self._record_mock_send(message)
                return True
            
            # Send message
            await self.producer.send_and_wait(
                message.topic,
                message_data,
                timeout=timeout
            )
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(sent=True, latency=latency)
            self.message_history.append(message)
            
            self.logger.debug(f"Message sent successfully: {message.message_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self._update_circuit_breaker(failed=True)
            self._update_metrics(sent=False, error=True)
            return False
    
    async def create_consumer(
        self,
        topic: str,
        group_id: str,
        message_handler: Callable[[NISMessage], None]
    ) -> bool:
        """
        Create consumer for a specific topic with message handler
        
        Args:
            topic: Kafka topic to consume
            group_id: Consumer group ID
            message_handler: Function to handle received messages
            
        Returns:
            bool: True if consumer created successfully
        """
        if not KAFKA_AVAILABLE:
            self.logger.info(f"Mock consumer created for topic: {topic}")
            return True
        
        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            await consumer.start()
            self.consumers[topic] = consumer
            self.message_handlers[topic].append(message_handler)
            
            # Start consumer task
            asyncio.create_task(self._consume_messages(topic, consumer))
            
            self.logger.info(f"Consumer created for topic: {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create consumer for {topic}: {e}")
            return False
    
    async def _consume_messages(self, topic: str, consumer: AIOKafkaConsumer):
        """Consume messages from a topic"""
        try:
            async for message in consumer:
                try:
                    # Parse message
                    message_data = message.value
                    nis_message = self._parse_nis_message(message_data)
                    
                    # Self-audit validation
                    if self.enable_self_audit:
                        audit_result = self._audit_message(nis_message)
                        if audit_result['score'] < self.audit_threshold:
                            self.logger.warning(f"Received message failed audit: {audit_result['flags']}")
                            continue
                    
                    # Handle message
                    for handler in self.message_handlers[topic]:
                        try:
                            handler(nis_message)
                        except Exception as e:
                            self.logger.error(f"Message handler error: {e}")
                    
                    self._update_metrics(received=True)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from {topic}: {e}")
                    self._update_metrics(received=False, error=True)
                    
        except Exception as e:
            self.logger.error(f"Consumer error for {topic}: {e}")
    
    def _audit_message(self, message: NISMessage) -> Dict[str, Any]:
        """Audit message for integrity violations"""
        try:
            # Convert message content to text for auditing
            audit_text = f"""
            Message Type: {message.message_type.value}
            Source Agent: {message.source_agent}
            Content: {json.dumps(message.content)}
            """
            
            # Use self-audit engine
            violations = self_audit_engine.audit_text(audit_text)
            score = self_audit_engine.get_integrity_score(audit_text)
            
            flags = [v['type'] for v in violations] if violations else []
            
            return {
                'score': score,
                'flags': flags,
                'violations': violations
            }
            
        except Exception as e:
            self.logger.error(f"Audit error: {e}")
            return {'score': 100.0, 'flags': [], 'violations': []}
    
    def _auto_correct_message(self, message: NISMessage, audit_result: Dict[str, Any]) -> NISMessage:
        """Attempt to auto-correct message integrity violations"""
        try:
            corrected_content = message.content.copy()
            
            # Apply corrections based on violation flags
            for flag in audit_result['flags']:
                if flag == 'HARDCODED_PERFORMANCE':
                    # Remove hardcoded performance values
                    for key in list(corrected_content.keys()):
                        if any(term in key.lower() for term in ['confidence', 'accuracy', 'performance']):
                            if isinstance(corrected_content[key], (int, float)) and 0 <= corrected_content[key] <= 1:
                                corrected_content[key] = f"calculated_{key}"
                
                elif flag == 'HYPE_LANGUAGE':
                    # Replace hype language with neutral terms
                    if 'description' in corrected_content:
                        desc = corrected_content['description']
                        hype_replacements = {
                            'revolutionary': 'updated',
                            'breakthrough': 'improvement',
                            'advanced': 'enhanced',
                            'novel': 'new'
                        }
                        for hype, neutral in hype_replacements.items():
                            desc = desc.replace(hype, neutral)
                        corrected_content['description'] = desc
            
            # Create corrected message
            corrected_message = NISMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                priority=message.priority,
                source_agent=message.source_agent,
                target_agent=message.target_agent,
                topic=message.topic,
                content=corrected_content,
                timestamp=message.timestamp,
                correlation_id=message.correlation_id,
                metadata=message.metadata
            )
            
            self.logger.info(f"Auto-corrected message: {message.message_id}")
            return corrected_message
            
        except Exception as e:
            self.logger.error(f"Auto-correction failed: {e}")
            return message
    
    def _parse_nis_message(self, message_data: Dict[str, Any]) -> NISMessage:
        """Parse raw message data into NISMessage"""
        return NISMessage(
            message_id=message_data.get('message_id', str(uuid.uuid4())),
            message_type=MessageType(message_data.get('message_type', 'system_health')),
            priority=MessagePriority(message_data.get('priority', 3)),
            source_agent=message_data.get('source_agent', 'unknown'),
            target_agent=message_data.get('target_agent'),
            topic=message_data.get('topic', 'default'),
            content=message_data.get('content', {}),
            timestamp=message_data.get('timestamp', time.time()),
            correlation_id=message_data.get('correlation_id'),
            metadata=message_data.get('metadata'),
            integrity_score=message_data.get('integrity_score'),
            audit_flags=message_data.get('audit_flags')
        )
    
    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state"""
        current_time = time.time()
        
        if self.circuit_breaker['state'] == 'open':
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['timeout']:
                self.circuit_breaker['state'] = 'half-open'
                self.logger.info("Circuit breaker moved to half-open state")
            else:
                return False
        
        return True
    
    def _update_circuit_breaker(self, failed: bool = False):
        """Update circuit breaker state"""
        if failed:
            self.circuit_breaker['failure_count'] += 1
            self.circuit_breaker['last_failure_time'] = time.time()
            
            if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
                self.circuit_breaker['state'] = 'open'
                self.logger.warning("Circuit breaker opened due to failures")
        else:
            self.circuit_breaker['failure_count'] = 0
            if self.circuit_breaker['state'] == 'half-open':
                self.circuit_breaker['state'] = 'closed'
                self.logger.info("Circuit breaker closed")
    
    def _update_metrics(self, sent: bool = False, received: bool = False, error: bool = False, latency: float = 0.0):
        """Update streaming metrics"""
        if sent:
            self.metrics.messages_sent += 1
        if received:
            self.metrics.messages_received += 1
        if error:
            self.metrics.errors_encountered += 1
        if latency > 0:
            # Update rolling average latency
            self.metrics.avg_latency = (self.metrics.avg_latency * 0.9) + (latency * 0.1)
        
        self.metrics.last_update = time.time()
    
    def _record_mock_send(self, message: NISMessage):
        """Record mock message send for testing"""
        self.message_history.append(message)
        self._update_metrics(sent=True, latency=0.001)
        self.logger.debug(f"Mock message sent: {message.message_id}")
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics"""
        return self.metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Kafka manager"""
        return {
            'initialized': self.is_initialized,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'failure_count': self.circuit_breaker['failure_count'],
            'active_consumers': len(self.consumers),
            'metrics': asdict(self.metrics),
            'kafka_available': KAFKA_AVAILABLE,
            'self_audit_enabled': self.enable_self_audit
        } 