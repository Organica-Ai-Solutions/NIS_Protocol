# 🤖 Agent Connection & Integration Guide

## 📋 **Overview**

This comprehensive guide explains how to connect, integrate, and develop custom agents within the NIS Protocol v3.1 ecosystem. Whether you're integrating existing agents or building new ones, this guide provides everything you need to get started.

## 🎯 **Quick Start**

### **⚡ Connect to Existing Agent**
```python
from src.agents.agent_router import AgentRouter
from src.meta.unified_coordinator import UnifiedCoordinator

# Initialize coordinator and router
coordinator = UnifiedCoordinator()
router = AgentRouter(coordinator)

# Route request to appropriate agent
result = await router.route_request({
    "task_type": "physics_validation",
    "data": {"equation": "E=mc²", "context": "relativity"},
    "priority": "high"
})
```

### **🔌 Create Custom Agent**
```python
from src.core.agent import NISAgent
from typing import Dict, Any

class CustomAgent(NISAgent):
    def __init__(self, agent_id: str = "custom_agent"):
        super().__init__(agent_id)
        self.specialty = "custom_processing"
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Custom agent logic here
        return {
            "status": "success",
            "result": "processed_data",
            "agent_id": self.agent_id
        }
```

## 🏗️ **Agent Architecture Overview**

### **📊 Agent Hierarchy**
```
🎪 UnifiedCoordinator
    ↓
🗂️ Agent Router
    ↓
🤖 Specialized Agents
    ├── 📡 Signal Processing (Unified Signal Agent)
    ├── 🧮 Reasoning (Unified Reasoning Agent)  
    ├── ⚗️ Physics (Unified Physics Agent)
    ├── 😊 Emotion Agent
    ├── 🎯 Goals Agent
    ├── 🧪 Curiosity Agent
    ├── ⚖️ Ethics Agent
    ├── 💭 Memory Agent
    ├── 👁️ Vision Agent
    └── 🔧 Engineering Agents
```

### **🔄 Communication Flow**
```python
# Standard agent communication pattern
request = {
    "task_type": "reasoning",
    "data": {"input": "complex_problem"},
    "context": {"urgency": "normal", "domain": "physics"},
    "metadata": {"source": "user", "session_id": "12345"}
}

response = {
    "status": "success|error",
    "result": {"processed_data": "..."},
    "agent_id": "unified_reasoning_agent",
    "timestamp": 1640995200.0,
    "metadata": {"processing_time": 0.5, "confidence": 0.95}
}
```

## 🔌 **Integration Methods**

### **Method 1: Direct Agent Import**
```python
# Import specific unified agents
from src.agents.reasoning.unified_reasoning_agent import UnifiedReasoningAgent
from src.agents.physics.unified_physics_agent import UnifiedPhysicsAgent
from src.agents.signal_processing.unified_signal_agent import UnifiedSignalAgent

# Initialize agents
reasoning_agent = UnifiedReasoningAgent()
physics_agent = UnifiedPhysicsAgent()
signal_agent = UnifiedSignalAgent()

# Process data through mathematical pipeline
signal_result = await signal_agent.transform_signal(input_data)
reasoning_result = await reasoning_agent.process_kan_reasoning(signal_result)
physics_result = await physics_agent.validate_physics(reasoning_result)
```

### **Method 2: Router-Based Integration**
```python
from src.agents.agent_router import AgentRouter
from src.meta.unified_coordinator import create_unified_coordinator

# Initialize system
coordinator = create_unified_coordinator()
router = AgentRouter(coordinator)

# Route requests intelligently
tasks = [
    {"type": "signal_processing", "data": signal_data},
    {"type": "ethical_evaluation", "data": ethical_scenario},
    {"type": "curiosity_exploration", "data": learning_stimulus}
]

results = []
for task in tasks:
    result = await router.route_request(task)
    results.append(result)
```

### **Method 3: Coordinator Integration**
```python
from src.meta.unified_coordinator import UnifiedCoordinator

# Use high-level coordinator interface
coordinator = UnifiedCoordinator()

# Access mathematical pipeline
pipeline_result = await coordinator.process_scientific_pipeline({
    "signal_data": time_series_data,
    "analysis_mode": "frequency_domain",
    "physics_validation": True
})

# Access agent management
agent_status = coordinator.get_agent_status()
coordinator.scale_agent_instances("reasoning", target_instances=5)
```

## 🛠️ **Building Custom Agents**

### **🎯 Step 1: Agent Class Structure**
```python
from src.core.agent import NISAgent
from src.utils.confidence_calculator import calculate_confidence
from typing import Dict, Any, Optional
import asyncio
import logging

class CustomDomainAgent(NISAgent):
    """
    Custom agent for specific domain processing
    """
    
    def __init__(
        self, 
        agent_id: str = "custom_domain_agent",
        domain_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id)
        self.domain_config = domain_config or {}
        self.logger = logging.getLogger(f"nis.agents.{agent_id}")
        self.specialty = "custom_domain_processing"
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - implement your logic here"""
        try:
            # Your custom processing logic
            processed_result = await self._domain_specific_processing(data)
            
            # Calculate confidence for result
            confidence = calculate_confidence([
                processed_result.get('quality_score', 0.8),
                processed_result.get('completeness_score', 0.9)
            ])
            
            return {
                "status": "success",
                "result": processed_result,
                "confidence": confidence,
                "agent_id": self.agent_id,
                "timestamp": self._get_timestamp(),
                "metadata": {
                    "processing_time": processed_result.get('processing_time'),
                    "domain": self.specialty
                }
            }
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": self._get_timestamp()
            }
    
    async def _domain_specific_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement your domain-specific logic here"""
        # Example implementation
        input_data = data.get('input', {})
        
        # Custom processing logic
        result = {
            "processed_input": input_data,
            "domain_insights": self._extract_domain_insights(input_data),
            "quality_score": 0.85,
            "completeness_score": 0.92,
            "processing_time": 0.15
        }
        
        return result
    
    def _extract_domain_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain-specific insights"""
        return {
            "key_patterns": [],
            "recommendations": [],
            "confidence_indicators": []
        }
```

### **🎯 Step 2: Agent Registration**
```python
# Register your agent with the system
from src.agents.agent_router import AgentRouter

# Option A: Register with router
router = AgentRouter()
custom_agent = CustomDomainAgent()
router.register_agent("custom_domain", custom_agent)

# Option B: Add to coordinator
from src.meta.unified_coordinator import UnifiedCoordinator

coordinator = UnifiedCoordinator()
coordinator.register_specialized_agent("custom_domain", custom_agent)
```

### **🎯 Step 3: Agent Configuration**
```python
# Agent configuration example
custom_agent_config = {
    "agent_id": "custom_domain_agent",
    "domain_config": {
        "processing_mode": "advanced",
        "confidence_threshold": 0.8,
        "max_processing_time": 30.0,
        "enable_caching": True,
        "cache_ttl": 3600
    },
    "integration_settings": {
        "kafka_topics": ["custom_domain_input", "custom_domain_output"],
        "redis_namespace": "custom_domain",
        "monitoring_enabled": True
    }
}

# Initialize with configuration
custom_agent = CustomDomainAgent(
    agent_id=custom_agent_config["agent_id"],
    domain_config=custom_agent_config["domain_config"]
)
```

## 🔄 **Agent Communication Patterns**

### **📨 Asynchronous Messaging**
```python
from src.infrastructure.message_streaming import NISKafkaManager

class CommunicatingAgent(NISAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.kafka_manager = NISKafkaManager()
    
    async def send_message_to_agent(self, target_agent: str, message: Dict[str, Any]):
        """Send message to another agent"""
        topic = f"agent_{target_agent}_input"
        await self.kafka_manager.send_message(topic, message)
    
    async def listen_for_messages(self):
        """Listen for incoming messages"""
        topic = f"agent_{self.agent_id}_input"
        async for message in self.kafka_manager.consume_messages(topic):
            await self.handle_incoming_message(message)
    
    async def handle_incoming_message(self, message: Dict[str, Any]):
        """Handle incoming inter-agent message"""
        sender = message.get('sender_agent_id')
        data = message.get('data')
        
        # Process the message
        response = await self.process(data)
        
        # Send response back if needed
        if message.get('requires_response'):
            await self.send_response(sender, response)
```

### **🤝 Direct Agent Collaboration**
```python
class CollaborativeAgent(NISAgent):
    def __init__(self, agent_id: str, collaborator_agents: List[str] = None):
        super().__init__(agent_id)
        self.collaborators = collaborator_agents or []
        self.agent_registry = {}
    
    async def collaborate_with_agents(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with multiple agents on a complex task"""
        results = {}
        
        # Distribute sub-tasks to collaborator agents
        for agent_id in self.collaborators:
            if agent_id in self.agent_registry:
                agent = self.agent_registry[agent_id]
                sub_task = self._create_sub_task(agent_id, task_data)
                results[agent_id] = await agent.process(sub_task)
        
        # Synthesize results
        final_result = self._synthesize_results(results)
        return final_result
    
    def register_collaborator(self, agent_id: str, agent_instance: NISAgent):
        """Register a collaborator agent"""
        self.agent_registry[agent_id] = agent_instance
        if agent_id not in self.collaborators:
            self.collaborators.append(agent_id)
```

## 🧠 **Memory & State Management**

### **💾 Agent Memory Integration**
```python
from src.memory.memory_manager import MemoryManager

class MemoryAwareAgent(NISAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.memory_manager = MemoryManager()
        self.context_window = 10  # Remember last 10 interactions
    
    async def process_with_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with memory context"""
        # Retrieve relevant memories
        relevant_memories = await self.memory_manager.search_memories(
            query=data.get('input', ''),
            agent_id=self.agent_id,
            limit=self.context_window
        )
        
        # Add memory context to processing
        enhanced_data = {
            **data,
            'memory_context': relevant_memories,
            'previous_interactions': await self._get_recent_interactions()
        }
        
        # Process with enhanced context
        result = await self.process(enhanced_data)
        
        # Store new memory
        await self.memory_manager.store_memory(
            content=result,
            agent_id=self.agent_id,
            metadata={'interaction_type': 'processing', 'confidence': result.get('confidence')}
        )
        
        return result
    
    async def _get_recent_interactions(self) -> List[Dict[str, Any]]:
        """Retrieve recent agent interactions"""
        return await self.memory_manager.get_recent_memories(
            agent_id=self.agent_id,
            limit=5
        )
```

### **🔄 State Persistence**
```python
from src.infrastructure.caching_system import NISCacheManager

class StatefulAgent(NISAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.cache_manager = NISCacheManager()
        self.state_key = f"agent_state_{agent_id}"
    
    async def save_state(self, state_data: Dict[str, Any]):
        """Save agent state to cache"""
        await self.cache_manager.set(
            key=self.state_key,
            value=state_data,
            ttl=3600  # 1 hour TTL
        )
    
    async def load_state(self) -> Dict[str, Any]:
        """Load agent state from cache"""
        state = await self.cache_manager.get(self.state_key)
        return state or {}
    
    async def process_with_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with persistent state"""
        # Load current state
        current_state = await self.load_state()
        
        # Process with state context
        result = await self.process({
            **data,
            'agent_state': current_state
        })
        
        # Update and save state
        new_state = {
            **current_state,
            'last_processed': self._get_timestamp(),
            'processing_count': current_state.get('processing_count', 0) + 1,
            'last_result_confidence': result.get('confidence')
        }
        await self.save_state(new_state)
        
        return result
```

## 📊 **Monitoring & Observability**

### **📈 Agent Performance Monitoring**
```python
from src.monitoring.real_time_dashboard import RealTimeDashboard
import time

class MonitoredAgent(NISAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.dashboard = RealTimeDashboard()
        self.metrics = {
            'requests_processed': 0,
            'average_response_time': 0.0,
            'success_rate': 0.0,
            'error_count': 0
        }
    
    async def process_with_monitoring(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with comprehensive monitoring"""
        start_time = time.time()
        
        try:
            # Process the request
            result = await self.process(data)
            
            # Update success metrics
            processing_time = time.time() - start_time
            self._update_metrics(success=True, processing_time=processing_time)
            
            # Send metrics to dashboard
            await self._send_metrics_to_dashboard(result, processing_time)
            
            return result
            
        except Exception as e:
            # Update error metrics
            processing_time = time.time() - start_time
            self._update_metrics(success=False, processing_time=processing_time)
            
            # Log error to dashboard
            await self._send_error_to_dashboard(str(e), processing_time)
            
            raise e
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update internal agent metrics"""
        self.metrics['requests_processed'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        count = self.metrics['requests_processed']
        self.metrics['average_response_time'] = (
            (current_avg * (count - 1) + processing_time) / count
        )
        
        # Update success rate
        if not success:
            self.metrics['error_count'] += 1
        
        self.metrics['success_rate'] = (
            (self.metrics['requests_processed'] - self.metrics['error_count']) 
            / self.metrics['requests_processed']
        )
    
    async def _send_metrics_to_dashboard(self, result: Dict[str, Any], processing_time: float):
        """Send performance metrics to monitoring dashboard"""
        await self.dashboard.update_agent_metrics(
            agent_id=self.agent_id,
            metrics={
                **self.metrics,
                'last_processing_time': processing_time,
                'last_confidence': result.get('confidence', 0.0),
                'timestamp': self._get_timestamp()
            }
        )
```

## 🔧 **Testing & Debugging**

### **🧪 Agent Unit Testing**
```python
import pytest
from unittest.mock import AsyncMock
from src.core.agent import NISAgent

class TestCustomAgent:
    @pytest.fixture
    def custom_agent(self):
        return CustomDomainAgent("test_agent")
    
    @pytest.mark.asyncio
    async def test_agent_processing(self, custom_agent):
        """Test basic agent processing functionality"""
        test_data = {
            "input": {"test_value": 123},
            "context": {"domain": "test"}
        }
        
        result = await custom_agent.process(test_data)
        
        assert result["status"] == "success"
        assert "result" in result
        assert result["agent_id"] == "test_agent"
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, custom_agent):
        """Test agent error handling"""
        # Mock a processing error
        custom_agent._domain_specific_processing = AsyncMock(
            side_effect=ValueError("Test error")
        )
        
        result = await custom_agent.process({"input": "bad_data"})
        
        assert result["status"] == "error"
        assert "error" in result
        assert result["agent_id"] == "test_agent"
    
    @pytest.mark.asyncio  
    async def test_agent_collaboration(self, custom_agent):
        """Test inter-agent collaboration"""
        collaborator = CustomDomainAgent("collaborator_agent")
        custom_agent.register_collaborator("collaborator", collaborator)
        
        result = await custom_agent.collaborate_with_agents({
            "complex_task": "multi_agent_processing"
        })
        
        assert result is not None
        assert "collaborator" in custom_agent.agent_registry
```

### **🔍 Debugging Tools**
```python
from src.core.agent import NISAgent
import logging

class DebuggableAgent(NISAgent):
    def __init__(self, agent_id: str, debug_mode: bool = False):
        super().__init__(agent_id)
        self.debug_mode = debug_mode
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(f"DEBUG.{agent_id}")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with detailed debugging information"""
        if self.debug_mode:
            self.logger.debug(f"Processing input: {data}")
            self.logger.debug(f"Agent state: {await self._get_debug_state()}")
        
        try:
            result = await self._debug_process(data)
            
            if self.debug_mode:
                self.logger.debug(f"Processing result: {result}")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                self.logger.debug(f"Processing error: {e}", exc_info=True)
            raise e
    
    async def _debug_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with step-by-step debugging"""
        debug_steps = []
        
        # Step 1: Input validation
        debug_steps.append("Input validation")
        validated_input = self._validate_input(data)
        
        # Step 2: Data processing
        debug_steps.append("Data processing")
        processed_data = await self._process_data(validated_input)
        
        # Step 3: Result formatting
        debug_steps.append("Result formatting")
        formatted_result = self._format_result(processed_data)
        
        return {
            **formatted_result,
            "debug_info": {
                "steps_completed": debug_steps,
                "processing_path": self._get_processing_path(data),
                "performance_metrics": self._get_performance_metrics()
            } if self.debug_mode else {}
        }
    
    async def _get_debug_state(self) -> Dict[str, Any]:
        """Get current agent state for debugging"""
        return {
            "agent_id": self.agent_id,
            "memory_usage": self._get_memory_usage(),
            "active_connections": self._get_active_connections(),
            "last_activity": self._get_last_activity()
        }
```

## 📋 **Best Practices**

### **✅ Agent Development Guidelines**
1. **Inherit from NISAgent**: Always use the base NISAgent class
2. **Implement Async Methods**: Use async/await for all I/O operations
3. **Error Handling**: Comprehensive try/catch with meaningful error messages
4. **Logging**: Use structured logging with appropriate log levels
5. **Configuration**: Make agents configurable through initialization parameters
6. **Testing**: Write comprehensive unit and integration tests
7. **Documentation**: Document all public methods and agent capabilities

### **🔒 Security Considerations**
1. **Input Validation**: Validate all input data before processing
2. **Access Control**: Implement proper authentication and authorization
3. **Data Privacy**: Handle sensitive data according to privacy regulations
4. **Audit Trail**: Log all agent interactions and decisions
5. **Resource Limits**: Implement appropriate timeouts and resource constraints

### **⚡ Performance Optimization**
1. **Caching**: Implement appropriate caching strategies
2. **Connection Pooling**: Reuse database and external service connections
3. **Async Processing**: Use asynchronous programming patterns
4. **Resource Management**: Properly manage memory and computational resources
5. **Monitoring**: Implement comprehensive performance monitoring

## 📞 **Support & Resources**

### **📚 Documentation**
- [API Reference](API_Reference.md)
- [System Architecture](ARCHITECTURE.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Performance Guide](../ENHANCED_KAFKA_REDIS_INTEGRATION_GUIDE.md)

### **💬 Community**
- GitHub Issues: Report bugs and request features
- Discussions: Technical questions and community support
- Examples: Real-world integration examples in `/dev/examples/`

### **🛠️ Development Tools**
- Agent Templates: Pre-built agent templates for common use cases
- Testing Framework: Comprehensive testing utilities
- Debugging Tools: Advanced debugging and profiling utilities
- Monitoring Dashboard: Real-time agent performance monitoring

---

**Ready to build your first agent?** Start with the Quick Start section and follow the step-by-step guide to create your first custom NIS Protocol agent! 🚀
