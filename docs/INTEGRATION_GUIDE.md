# 🔗 NIS Protocol v3 - Integration Guide

> **Comprehensive guide for integrating NIS Protocol v3 into your systems**

---

## 📋 **Table of Contents**

1. [Integration Overview](#integration-overview)
2. [Quick Integration](#quick-integration)
3. [Scientific Pipeline Integration](#scientific-pipeline-integration)
4. [Consciousness System Integration](#consciousness-system-integration)
5. [Custom Agent Development](#custom-agent-development)
6. [Performance Optimization](#performance-optimization)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## 🌟 **Integration Overview**

NIS Protocol v3 provides multiple integration points depending on your use case:

### **Integration Levels**

| **Level** | **Use Case** | **Complexity** | **Features** |
|:---|:---|:---:|:---|
| **🔬 Scientific Pipeline** | Signal processing, physics validation | **Medium** | Complete Laplace→KAN→PINN→LLM pipeline |
| **🧠 Consciousness Layer** | System monitoring, introspection | **Low** | Meta-cognitive capabilities, integrity monitoring |
| **🛡️ Integrity Monitoring** | Quality assurance, validation | **Low** | Self-audit, violation detection |
| **🤖 Custom Agents** | Specialized functionality | **High** | Full agent framework with consciousness |
| **📊 Performance Monitoring** | System analytics | **Low** | Performance metrics, trend analysis |

### **System Requirements**

```yaml
# Minimum Requirements
Python: 3.8+
Memory: 8GB RAM
Storage: 2GB free space
CPU: 4+ cores recommended

# Dependencies
Core: numpy, scipy, torch, sympy, scikit-learn
Optional: kafka-python, redis, langchain, langgraph
```

---

## ⚡ **Quick Integration**

### **1. Basic Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/NIS-Protocol.git
cd NIS-Protocol

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "
import sys
sys.path.insert(0, 'src')
from meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator
print('✅ NIS Protocol v3 ready!')
"
```

### **2. Basic Signal Processing**

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer

# Initialize transformer
transformer = EnhancedLaplaceTransformer(
    agent_id="basic_integration",
    enable_self_audit=True
)

# Process a signal
t = np.linspace(0, 2, 1000)
signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-0.5*t)

result = transformer.compute_laplace_transform(signal, t)

print(f"Processing completed in {result.metrics.processing_time:.4f}s")
print(f"Reconstruction error: {result.reconstruction_error:.6f}")
print(f"Signal quality: {result.quality_assessment.value}")
```

### **3. Basic Integrity Monitoring**

```python
from utils.self_audit import self_audit_engine

# Audit any text output
text = "System analysis completed with measured performance metrics"
violations = self_audit_engine.audit_text(text)
integrity_score = self_audit_engine.get_integrity_score(text)

print(f"Integrity score: {integrity_score}/100")
print(f"Violations: {len(violations)}")

if violations:
    corrected_text, _ = self_audit_engine.auto_correct_text(text)
    print(f"Corrected: {corrected_text}")
```

---

## 🔬 **Scientific Pipeline Integration**

### **Complete Pipeline Setup**

```python
import asyncio
import numpy as np
from typing import Dict, Any

# Import pipeline components
from meta.enhanced_scientific_coordinator import (
    EnhancedScientificCoordinator, PipelineStage, ProcessingPriority
)
from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
from agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent
from agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent

class ScientificPipelineIntegration:
    """Complete scientific pipeline integration wrapper"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.coordinator = None
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """Initialize and configure the complete scientific pipeline"""
        
        # Initialize coordinator
        self.coordinator = EnhancedScientificCoordinator(
            coordinator_id="production_pipeline",
            enable_self_audit=self.config.get('enable_integrity', True),
            enable_auto_correction=self.config.get('enable_auto_correction', True)
        )
        
        # Initialize agents with configuration
        laplace_config = self.config.get('laplace', {})
        laplace_agent = EnhancedLaplaceTransformer(
            agent_id="production_laplace",
            max_frequency=laplace_config.get('max_frequency', 1000.0),
            num_points=laplace_config.get('num_points', 2048),
            enable_self_audit=True
        )
        
        kan_config = self.config.get('kan', {})
        kan_agent = EnhancedKANReasoningAgent(
            agent_id="production_kan",
            input_dim=kan_config.get('input_dim', 8),
            hidden_dims=kan_config.get('hidden_dims', [16, 12, 8]),
            output_dim=kan_config.get('output_dim', 4),
            enable_self_audit=True
        )
        
        pinn_config = self.config.get('pinn', {})
        pinn_agent = EnhancedPINNPhysicsAgent(
            agent_id="production_pinn",
            enable_self_audit=True,
            strict_mode=pinn_config.get('strict_mode', False)
        )
        
        # Register agents
        self.coordinator.register_pipeline_agent(PipelineStage.LAPLACE_TRANSFORM, laplace_agent)
        self.coordinator.register_pipeline_agent(PipelineStage.KAN_REASONING, kan_agent)
        self.coordinator.register_pipeline_agent(PipelineStage.PINN_VALIDATION, pinn_agent)
        
        print("✅ Scientific pipeline initialized and ready")
    
    async def process_signal(self, 
                           signal_data: np.ndarray, 
                           time_vector: np.ndarray,
                           description: str = "",
                           priority: ProcessingPriority = ProcessingPriority.NORMAL):
        """
        Process signal through complete scientific pipeline
        
        Args:
            signal_data: Input time-domain signal
            time_vector: Corresponding time vector
            description: Description of the signal
            priority: Processing priority
            
        Returns:
            Complete pipeline results
        """
        
        input_data = {
            'signal_data': signal_data,
            'time_vector': time_vector,
            'description': description,
            'timestamp': time.time()
        }
        
        try:
            result = await self.coordinator.execute_scientific_pipeline(
                input_data, priority=priority
            )
            
            return {
                'success': True,
                'result': result,
                'summary': {
                    'overall_accuracy': result.overall_accuracy,
                    'physics_compliance': result.physics_compliance,
                    'processing_time': result.total_processing_time,
                    'integrity_score': result.integrity_score
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback_available': True
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return self.coordinator.get_coordination_summary()

# Usage Example
async def main():
    # Configure pipeline
    config = {
        'enable_integrity': True,
        'enable_auto_correction': True,
        'laplace': {
            'max_frequency': 500.0,
            'num_points': 1024
        },
        'kan': {
            'input_dim': 8,
            'hidden_dims': [16, 8],
            'output_dim': 1
        },
        'pinn': {
            'strict_mode': False
        }
    }
    
    # Initialize pipeline
    pipeline = ScientificPipelineIntegration(config)
    
    # Create test signal
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2*np.pi*10*t) + 0.3*np.sin(2*np.pi*25*t)
    
    # Process signal
    result = await pipeline.process_signal(
        signal, t, 
        description="Multi-frequency test signal",
        priority=ProcessingPriority.HIGH
    )
    
    if result['success']:
        print("✅ Pipeline processing successful:")
        print(f"  Accuracy: {result['summary']['overall_accuracy']:.3f}")
        print(f"  Physics compliance: {result['summary']['physics_compliance']:.3f}")
        print(f"  Processing time: {result['summary']['processing_time']:.4f}s")
    else:
        print(f"❌ Pipeline processing failed: {result['error']}")
    
    return result

# Run the example
if __name__ == "__main__":
    result = asyncio.run(main())
```

---

## 🧠 **Consciousness System Integration**

### **System Monitoring Integration**

```python
from agents.consciousness.enhanced_conscious_agent import (
    EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
)

class SystemMonitoringIntegration:
    """Integration wrapper for consciousness and monitoring capabilities"""
    
    def __init__(self, monitoring_interval: float = 60.0):
        self.monitoring_interval = monitoring_interval
        self.conscious_agent = None
        self.setup_consciousness()
    
    def setup_consciousness(self):
        """Initialize consciousness monitoring system"""
        
        self.conscious_agent = EnhancedConsciousAgent(
            agent_id="system_monitor",
            reflection_interval=self.monitoring_interval,
            enable_self_audit=True,
            consciousness_level=ConsciousnessLevel.INTEGRATED
        )
        
        print("✅ Consciousness monitoring system initialized")
    
    def register_system_component(self, component_id: str, metadata: Dict[str, Any]):
        """Register a system component for monitoring"""
        self.conscious_agent.register_agent_for_monitoring(component_id, metadata)
        print(f"✅ Registered {component_id} for monitoring")
    
    def update_component_performance(self, component_id: str, performance_score: float):
        """Update performance metrics for monitored component"""
        self.conscious_agent.update_agent_performance(component_id, performance_score)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health assessment"""
        
        health_result = self.conscious_agent.perform_introspection(
            ReflectionType.SYSTEM_HEALTH_CHECK
        )
        
        integrity_result = self.conscious_agent.perform_introspection(
            ReflectionType.INTEGRITY_ASSESSMENT
        )
        
        return {
            'system_health': {
                'confidence': health_result.confidence,
                'findings': health_result.findings,
                'recommendations': health_result.recommendations
            },
            'integrity_status': {
                'score': integrity_result.integrity_score,
                'violations': len(integrity_result.integrity_violations),
                'auto_corrections': integrity_result.auto_corrections_applied
            },
            'consciousness_summary': self.conscious_agent.get_consciousness_summary()
        }
    
    def start_continuous_monitoring(self):
        """Start continuous system monitoring"""
        self.conscious_agent.start_continuous_reflection()
        print("✅ Started continuous system monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous system monitoring"""
        self.conscious_agent.stop_continuous_reflection()
        print("⏹️ Stopped continuous system monitoring")

# Usage Example
def monitoring_integration_example():
    # Initialize monitoring
    monitor = SystemMonitoringIntegration(monitoring_interval=30.0)
    
    # Register system components
    monitor.register_system_component("database", {"type": "storage", "critical": True})
    monitor.register_system_component("api_server", {"type": "service", "critical": True})
    monitor.register_system_component("ml_pipeline", {"type": "processing", "critical": False})
    
    # Simulate performance updates
    import time
    import random
    
    for i in range(5):
        # Update performance metrics
        monitor.update_component_performance("database", 0.8 + 0.1 * random.random())
        monitor.update_component_performance("api_server", 0.9 + 0.05 * random.random())
        monitor.update_component_performance("ml_pipeline", 0.7 + 0.2 * random.random())
        
        # Get health assessment
        health = monitor.get_system_health()
        
        print(f"\n📊 System Health Check {i+1}:")
        print(f"  Health confidence: {health['system_health']['confidence']:.3f}")
        print(f"  Integrity score: {health['integrity_status']['score']:.1f}/100")
        print(f"  Recommendations: {len(health['system_health']['recommendations'])}")
        
        time.sleep(1)  # Wait between updates
    
    return monitor

# Run example
if __name__ == "__main__":
    monitor = monitoring_integration_example()
```

---

## 🛠️ **Custom Agent Development**

### **Creating Custom Agents**

```python
from core.agent import NISAgent, NISLayer
from utils.integrity_metrics import calculate_confidence, create_default_confidence_factors
from utils.self_audit import self_audit_engine
from typing import Dict, Any, Optional
import time
import logging

class CustomProcessingAgent(NISAgent):
    """
    Template for creating custom agents with full NIS Protocol integration
    """
    
    def __init__(self, agent_id: str, custom_config: Dict[str, Any] = None):
        super().__init__(agent_id, NISLayer.REASONING)
        
        self.custom_config = custom_config or {}
        self.processing_history = []
        self.performance_metrics = {
            'total_processed': 0,
            'successful_processed': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        # Initialize confidence factors
        self.confidence_factors = create_default_confidence_factors()
        
        # Enable integrity monitoring
        self.enable_self_audit = self.custom_config.get('enable_integrity', True)
        
        self.logger = logging.getLogger(f"custom.{agent_id}")
        self.logger.info(f"Custom processing agent {agent_id} initialized")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with full integrity monitoring
        """
        start_time = time.time()
        
        try:
            # Extract input data
            input_data = message.get('payload', {})
            
            # Perform custom processing
            result = self._custom_processing_logic(input_data)
            
            # Calculate confidence
            confidence = self._calculate_processing_confidence(input_data, result)
            
            # Create response
            response = {
                'agent_id': self.agent_id,
                'timestamp': time.time(),
                'status': 'success',
                'payload': result,
                'confidence': confidence,
                'processing_time': time.time() - start_time
            }
            
            # Apply integrity monitoring
            if self.enable_self_audit:
                response = self._apply_integrity_monitoring(response)
            
            # Update metrics
            self._update_performance_metrics(response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return {
                'agent_id': self.agent_id,
                'timestamp': time.time(),
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _custom_processing_logic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement your custom processing logic here
        """
        # Example: Simple data transformation
        processed_data = {}
        
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                # Apply some transformation
                processed_data[f"processed_{key}"] = value * 1.1
            else:
                processed_data[f"processed_{key}"] = str(value).upper()
        
        # Add processing metadata
        processed_data['processing_metadata'] = {
            'transformation_applied': 'scale_and_uppercase',
            'processing_agent': self.agent_id,
            'timestamp': time.time()
        }
        
        return processed_data
    
    def _calculate_processing_confidence(self, 
                                       input_data: Dict[str, Any], 
                                       result: Dict[str, Any]) -> float:
        """Calculate confidence in processing results"""
        
        # Assess data quality
        data_quality = 1.0 if input_data else 0.0
        if input_data:
            # Simple quality assessment based on data completeness
            total_fields = len(input_data)
            valid_fields = sum(1 for v in input_data.values() if v is not None)
            data_quality = valid_fields / total_fields if total_fields > 0 else 0.0
        
        # Assess result complexity (simple metric)
        complexity_factor = min(1.0, len(result) / 10.0)
        
        # Validation score (placeholder - implement your validation logic)
        validation_score = 0.85  # Replace with actual validation
        
        return calculate_confidence(
            data_quality, complexity_factor, validation_score, self.confidence_factors
        )
    
    def _apply_integrity_monitoring(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply integrity monitoring to response"""
        
        # Convert response to auditable text
        audit_text = f"Agent {response['agent_id']} processed data with {response['confidence']:.3f} confidence"
        
        # Audit the text
        violations = self_audit_engine.audit_text(audit_text)
        integrity_score = self_audit_engine.get_integrity_score(audit_text)
        
        # Add integrity information to response
        response['integrity_assessment'] = {
            'integrity_score': integrity_score,
            'violations_detected': len(violations),
            'audit_timestamp': time.time()
        }
        
        # Apply auto-correction if needed
        if violations:
            corrected_text, _ = self_audit_engine.auto_correct_text(audit_text)
            response['integrity_assessment']['corrected_description'] = corrected_text
            response['integrity_assessment']['auto_corrections_applied'] = len(violations)
        
        return response
    
    def _update_performance_metrics(self, response: Dict[str, Any], processing_time: float):
        """Update agent performance metrics"""
        
        self.performance_metrics['total_processed'] += 1
        
        if response['status'] == 'success':
            self.performance_metrics['successful_processed'] += 1
            
            # Update averages
            total = self.performance_metrics['total_processed']
            
            self.performance_metrics['average_processing_time'] = (
                (self.performance_metrics['average_processing_time'] * (total - 1) + processing_time) / total
            )
            
            if 'confidence' in response:
                self.performance_metrics['average_confidence'] = (
                    (self.performance_metrics['average_confidence'] * (total - 1) + response['confidence']) / total
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        success_rate = (
            self.performance_metrics['successful_processed'] / 
            max(1, self.performance_metrics['total_processed'])
        )
        
        return {
            'agent_id': self.agent_id,
            'performance_metrics': self.performance_metrics,
            'success_rate': success_rate,
            'status': 'operational' if success_rate > 0.8 else 'degraded',
            'recommendations': self._generate_performance_recommendations(success_rate)
        }
    
    def _generate_performance_recommendations(self, success_rate: float) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if success_rate < 0.8:
            recommendations.append("Success rate below 80% - investigate error patterns")
        
        if self.performance_metrics['average_processing_time'] > 1.0:
            recommendations.append("Processing time high - consider optimization")
        
        if self.performance_metrics['average_confidence'] < 0.7:
            recommendations.append("Average confidence low - review confidence calculation")
        
        if not recommendations:
            recommendations.append("Agent performing optimally")
        
        return recommendations

# Usage Example
def custom_agent_example():
    # Create custom agent
    config = {
        'enable_integrity': True,
        'custom_parameter': 'value'
    }
    
    agent = CustomProcessingAgent("custom_processor_001", config)
    
    # Test processing
    test_data = {
        'payload': {
            'sensor_reading': 42.5,
            'device_id': 'sensor_001',
            'location': 'building_a',
            'timestamp': time.time()
        }
    }
    
    # Process data
    result = agent.process(test_data)
    
    print("📊 Custom Agent Processing Result:")
    print(f"  Status: {result['status']}")
    print(f"  Confidence: {result.get('confidence', 'N/A'):.3f}")
    print(f"  Processing time: {result['processing_time']:.4f}s")
    
    if 'integrity_assessment' in result:
        integrity = result['integrity_assessment']
        print(f"  Integrity score: {integrity['integrity_score']:.1f}/100")
        print(f"  Violations: {integrity['violations_detected']}")
    
    # Get performance summary
    summary = agent.get_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Status: {summary['status']}")
    print(f"  Recommendations: {summary['recommendations']}")
    
    return agent

# Run example
if __name__ == "__main__":
    agent = custom_agent_example()
```

---

## 📈 **Performance Optimization**

### **Memory Management**

```python
import gc
import psutil
import os

class PerformanceOptimizer:
    """Performance optimization utilities for NIS Protocol integration"""
    
    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage for large-scale processing"""
        
        # Force garbage collection
        gc.collect()
        
        # Get memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'optimization_applied': True
        }
    
    @staticmethod
    def configure_for_batch_processing(batch_size: int = 100):
        """Configure agents for optimal batch processing"""
        
        config = {
            'laplace': {
                'num_points': 1024,  # Reduced for faster processing
                'max_frequency': 100.0  # Lower for batch processing
            },
            'kan': {
                'input_dim': 4,  # Reduced dimensionality
                'hidden_dims': [8, 4],  # Smaller network
                'batch_size': batch_size
            },
            'enable_integrity': False,  # Disable for performance
            'enable_auto_correction': False
        }
        
        return config
    
    @staticmethod
    def monitor_processing_performance(processing_function, *args, **kwargs):
        """Monitor and log processing performance"""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = processing_function(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            performance_metrics = {
                'processing_time': end_time - start_time,
                'memory_delta_mb': (end_memory - start_memory) / 1024 / 1024,
                'success': True,
                'result': result
            }
            
            return performance_metrics
            
        except Exception as e:
            end_time = time.time()
            
            return {
                'processing_time': end_time - start_time,
                'success': False,
                'error': str(e)
            }

# Performance optimization example
async def optimized_batch_processing_example():
    """Example of optimized batch processing"""
    
    # Get optimized configuration
    config = PerformanceOptimizer.configure_for_batch_processing(batch_size=50)
    
    # Initialize pipeline with optimized config
    pipeline = ScientificPipelineIntegration(config)
    
    # Create batch of signals
    signals = []
    for i in range(10):
        t = np.linspace(0, 1, 500)  # Smaller signals for batch processing
        signal = np.sin(2*np.pi*(5 + i)*t)
        signals.append((signal, t))
    
    # Process batch with performance monitoring
    batch_results = []
    
    for i, (signal, t) in enumerate(signals):
        
        def process_single():
            return asyncio.run(pipeline.process_signal(
                signal, t, f"Batch signal {i}"
            ))
        
        # Monitor performance
        perf_metrics = PerformanceOptimizer.monitor_processing_performance(process_single)
        
        batch_results.append({
            'signal_index': i,
            'processing_time': perf_metrics['processing_time'],
            'memory_delta': perf_metrics.get('memory_delta_mb', 0),
            'success': perf_metrics['success']
        })
        
        # Optimize memory between batches
        if i % 5 == 0:
            PerformanceOptimizer.optimize_memory_usage()
    
    # Calculate batch statistics
    successful = [r for r in batch_results if r['success']]
    avg_time = np.mean([r['processing_time'] for r in successful]) if successful else 0
    total_memory = sum([r['memory_delta'] for r in batch_results])
    
    print(f"📊 Batch Processing Results:")
    print(f"  Total signals: {len(signals)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Average processing time: {avg_time:.4f}s")
    print(f"  Total memory usage: {total_memory:.2f}MB")
    
    return batch_results

# Run optimized example
if __name__ == "__main__":
    batch_results = asyncio.run(optimized_batch_processing_example())
```

---

## 🚀 **Production Deployment**

### **Docker Integration**

```dockerfile
# Dockerfile for NIS Protocol v3
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY docs/ ./docs/

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd -m -u 1000 nisuser
USER nisuser

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from utils.self_audit import self_audit_engine; print('OK')"

# Default command
CMD ["python", "-m", "api.main"]
```

### **Kubernetes Deployment**

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nis-protocol-v3
  labels:
    app: nis-protocol
    version: v3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nis-protocol
      version: v3
  template:
    metadata:
      labels:
        app: nis-protocol
        version: v3
    spec:
      containers:
      - name: nis-protocol
        image: nis-protocol:v3
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app/src"
        - name: NIS_CONFIG_PATH
          value: "/app/config"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: nis-protocol-config

---
apiVersion: v1
kind: Service
metadata:
  name: nis-protocol-service
spec:
  selector:
    app: nis-protocol
    version: v3
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### **Configuration Management**

```python
# config/production_config.py
import os
from typing import Dict, Any

class ProductionConfig:
    """Production configuration for NIS Protocol v3"""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        return {
            'scientific_pipeline': {
                'laplace': {
                    'max_frequency': float(os.getenv('LAPLACE_MAX_FREQ', '1000.0')),
                    'num_points': int(os.getenv('LAPLACE_NUM_POINTS', '2048')),
                    'enable_self_audit': os.getenv('ENABLE_INTEGRITY', 'true').lower() == 'true'
                },
                'kan': {
                    'input_dim': int(os.getenv('KAN_INPUT_DIM', '8')),
                    'hidden_dims': [16, 12, 8],  # Can be made configurable
                    'output_dim': int(os.getenv('KAN_OUTPUT_DIM', '4')),
                    'enable_self_audit': os.getenv('ENABLE_INTEGRITY', 'true').lower() == 'true'
                },
                'pinn': {
                    'strict_mode': os.getenv('PINN_STRICT_MODE', 'false').lower() == 'true',
                    'enable_self_audit': os.getenv('ENABLE_INTEGRITY', 'true').lower() == 'true'
                }
            },
            'consciousness': {
                'reflection_interval': float(os.getenv('REFLECTION_INTERVAL', '60.0')),
                'consciousness_level': os.getenv('CONSCIOUSNESS_LEVEL', 'ENHANCED'),
                'enable_continuous_monitoring': os.getenv('CONTINUOUS_MONITORING', 'true').lower() == 'true'
            },
            'monitoring': {
                'enable_performance_tracking': True,
                'enable_integrity_monitoring': True,
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'metrics_collection_interval': float(os.getenv('METRICS_INTERVAL', '30.0'))
            },
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', '8000')),
                'workers': int(os.getenv('API_WORKERS', '4')),
                'timeout': int(os.getenv('API_TIMEOUT', '30'))
            }
        }
```

---

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

#### **Memory Issues**

```python
# Issue: High memory usage during processing
# Solution: Implement memory management

def handle_large_signal_processing(signal_data):
    """Handle large signals with memory management"""
    
    if len(signal_data) > 10000:
        # Process in chunks
        chunk_size = 5000
        results = []
        
        for i in range(0, len(signal_data), chunk_size):
            chunk = signal_data[i:i+chunk_size]
            
            # Process chunk
            result = process_signal_chunk(chunk)
            results.append(result)
            
            # Clean up memory
            gc.collect()
        
        return combine_chunk_results(results)
    else:
        return process_signal_normally(signal_data)
```

#### **Performance Issues**

```python
# Issue: Slow processing times
# Solution: Profile and optimize

import cProfile
import pstats

def profile_processing_performance():
    """Profile processing performance to identify bottlenecks"""
    
    profiler = cProfile.Profile()
    
    # Enable profiling
    profiler.enable()
    
    # Run processing
    result = process_test_signal()
    
    # Disable profiling
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Print top 10 functions by time
    print("🔍 Performance Profile (Top 10 functions):")
    stats.print_stats(10)
    
    return result
```

#### **Integration Issues**

```python
# Issue: Agent integration failures
# Solution: Validation and fallback

def validate_agent_integration():
    """Validate agent integration and provide diagnostics"""
    
    diagnostics = {
        'agents_status': {},
        'integration_issues': [],
        'recommendations': []
    }
    
    # Test Laplace transformer
    try:
        transformer = EnhancedLaplaceTransformer("test_laplace")
        test_signal = np.sin(np.linspace(0, 1, 100))
        result = transformer.compute_laplace_transform(test_signal, np.linspace(0, 1, 100))
        diagnostics['agents_status']['laplace'] = 'operational'
    except Exception as e:
        diagnostics['agents_status']['laplace'] = f'failed: {e}'
        diagnostics['integration_issues'].append(f"Laplace transformer: {e}")
    
    # Test KAN agent
    try:
        kan_agent = EnhancedKANReasoningAgent("test_kan", 4, [8, 4], 1)
        test_data = {'test': 'data'}
        diagnostics['agents_status']['kan'] = 'operational'
    except Exception as e:
        diagnostics['agents_status']['kan'] = f'failed: {e}'
        diagnostics['integration_issues'].append(f"KAN agent: {e}")
    
    # Test PINN agent
    try:
        pinn_agent = EnhancedPINNPhysicsAgent("test_pinn")
        diagnostics['agents_status']['pinn'] = 'operational'
    except Exception as e:
        diagnostics['agents_status']['pinn'] = f'failed: {e}'
        diagnostics['integration_issues'].append(f"PINN agent: {e}")
    
    # Generate recommendations
    if diagnostics['integration_issues']:
        diagnostics['recommendations'].append("Check dependency installation")
        diagnostics['recommendations'].append("Verify configuration settings")
        diagnostics['recommendations'].append("Review error logs for details")
    else:
        diagnostics['recommendations'].append("All agents operational - integration successful")
    
    return diagnostics

# Run diagnostics
def run_integration_diagnostics():
    """Run comprehensive integration diagnostics"""
    
    print("🔍 Running NIS Protocol v3 Integration Diagnostics...")
    
    diagnostics = validate_agent_integration()
    
    print("\n📊 Agent Status:")
    for agent, status in diagnostics['agents_status'].items():
        print(f"  {agent}: {status}")
    
    if diagnostics['integration_issues']:
        print(f"\n⚠️  Issues Found ({len(diagnostics['integration_issues'])}):")
        for issue in diagnostics['integration_issues']:
            print(f"  • {issue}")
    
    print(f"\n💡 Recommendations:")
    for rec in diagnostics['recommendations']:
        print(f"  • {rec}")
    
    return diagnostics

# Run diagnostics
if __name__ == "__main__":
    diagnostics = run_integration_diagnostics()
```

---

## 📚 **Additional Resources**

### **Integration Checklist**

- [ ] **Dependencies Installed**: All required packages available
- [ ] **Configuration Set**: Production configuration applied
- [ ] **Agents Initialized**: All required agents operational
- [ ] **Integrity Monitoring**: Self-audit system enabled
- [ ] **Performance Testing**: Baseline performance established
- [ ] **Error Handling**: Graceful degradation implemented
- [ ] **Monitoring Setup**: Health checks and metrics collection
- [ ] **Documentation**: Integration documented for team

### **Support and Community**

- **📖 Documentation**: Complete docs in `/docs` directory
- **🧪 Examples**: Working examples in `/examples` directory
- **🔍 Diagnostics**: Built-in diagnostic tools available
- **📝 Issues**: GitHub issues for bug reports and questions
- **💬 Community**: Join our community for integration support

---

<div align="center">
  <h3>🔗 Complete Integration Guide for NIS Protocol v3</h3>
  <p><em>From quick start to production deployment</em></p>
  
  <p>
    <a href="../README.md">🏠 Main Documentation</a> •
    <a href="API_Reference.md">📚 API Reference</a> •
    <a href="../examples/">🧪 Examples</a> •
    <a href="GETTING_STARTED.md">🚀 Getting Started</a>
  </p>
</div> 