# NIS Protocol Error Handling & Recovery Flow

```mermaid
graph TB
    subgraph "Error Detection Layer"
        ME[Monitor Events<br/>👁️ Continuous Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)]
        HD[Health Detector<br/>❤️ Component Health]
        ED[Error Detector<br/>🚨 Exception Catching]
        PD[Performance Detector<br/>📊 Performance Degradation]
    end
    
    subgraph "Error Classification"
        EC[Error Classifier<br/>🏷️ Error Analysis]
        subgraph "Error Types"
            SYS[System Errors<br/>💻 Infrastructure Issues]
            AGT[Agent Errors<br/>🤖 Agent Failures]
            LLM[LLM Errors<br/>🧠 Provider Issues]
            MEM[Memory Errors<br/>💾 Storage Issues]
            NET[Network Errors<br/>🌐 Connectivity Issues]
        end
        
        subgraph "Severity Levels"
            LOW[Low<br/>📗 Warning Level]
            MED[Medium<br/>📙 Intervention Needed]
            HIGH[High<br/>📕 Service Degradation]
            CRIT[Critical<br/>🚨 System Failure]
        end
    end
    
    subgraph "Recovery Strategies"
        RS[Recovery Selector<br/>⚖️ Strategy Selection]
        
        subgraph "Immediate Actions"
            RT[Retry<br/>🔄 Automatic Retry]
            FB[Fallback<br/>🔃 Alternative Service]
            CL[Circuit Breaker<br/>⚡ Service Protection]
            CF[Cache Fallback<br/>💾 Cached Response]
        end
        
        subgraph "advanced Recovery"
            DR[Degraded Mode<br/>⬇️ Reduced Functionality]
            LB[Load Redistribution<br/>⚖️ Traffic Rerouting]
            FS[Failsafe Mode<br/>🛡️ Safe Operation]
            ER[Emergency Response<br/>🚨 Crisis Protocol]
        end
    end
    
    subgraph "Component-Specific Recovery"
        subgraph "Agent Recovery"
            AR[Agent Restart<br/>🔄 Component Restart]
            AS[Agent Substitution<br/>🔄 Backup Agent]
            AC[Agent Reconfiguration<br/>⚙️ Parameter Adjustment]
        end
        
        subgraph "LLM Recovery"
            LR[Provider Switch<br/>🔄 Alternative Provider]
            LF[Local Fallback<br/>💻 Local Model]
            LC[LLM Cache<br/>💾 Previous Responses]
        end
        
        subgraph "Memory Recovery"
            MR[Memory Rebuild<br/>🏗️ Index Reconstruction]
            MF[Memory Fallback<br/>💾 Backup Storage]
            MC[Memory Cleanup<br/>🧹 Garbage Collection]
        end
    end
    
    subgraph "Recovery Execution"
        RE[Recovery Executor<br/>⚡ Action Implementation]
        VM[Validation Monitor<br/>✅ Success Verification]
        RR[Recovery Reporter<br/>📊 Status Updates]
    end
    
    subgraph "Learning & Adaptation"
        LA[Learning Agent<br/>📚 Pattern Recognition]
        PS[Pattern Storage<br/>💾 Error Pattern Memory]
        PA[Preventive Actions<br/>🛡️ Proactive Measures]
        IA[Improvement Actions<br/>📈 System Enhancement]
    end
    
    subgraph "Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) & Alerting"
        RM[Recovery Monitor<br/>📊 Recovery Tracking]
        AL[Alert System<br/>📢 Notification System]
        DG[Diagnostic Generator<br/>🔍 Issue Analysis]
        RD[Recovery Dashboard<br/>📊 Visual Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)]
    end
    
    %% Detection flow
    ME --> HD
    ME --> ED
    ME --> PD
    
    %% Classification flow
    HD --> EC
    ED --> EC
    PD --> EC
    
    EC --> SYS
    EC --> AGT
    EC --> LLM
    EC --> MEM
    EC --> NET
    
    EC --> LOW
    EC --> MED
    EC --> HIGH
    EC --> CRIT
    
    %% Recovery strategy selection
    SYS --> RS
    AGT --> RS
    LLM --> RS
    MEM --> RS
    NET --> RS
    
    LOW --> RT
    MED --> FB
    HIGH --> CL
    CRIT --> CF
    
    %% advanced recovery
    RS --> DR
    RS --> LB
    RS --> FS
    RS --> ER
    
    %% Component-specific recovery
    AGT --> AR
    AGT --> AS
    AGT --> AC
    
    LLM --> LR
    LLM --> LF
    LLM --> LC
    
    MEM --> MR
    MEM --> MF
    MEM --> MC
    
    %% Recovery execution
    RT --> RE
    FB --> RE
    CL --> RE
    CF --> RE
    DR --> RE
    LB --> RE
    FS --> RE
    ER --> RE
    
    AR --> RE
    AS --> RE
    AC --> RE
    LR --> RE
    LF --> RE
    LC --> RE
    MR --> RE
    MF --> RE
    MC --> RE
    
    %% Validation and reporting
    RE --> VM
    VM --> RR
    
    %% Learning and adaptation
    RR --> LA
    LA --> PS
    PS --> PA
    PA --> IA
    
    %% Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) and alerting
    RE --> RM
    VM --> AL
    RR --> DG
    RM --> RD
    
    %% Feedback loops
    IA -.-> ME
    PA -.-> HD
    DG -.-> EC
    RD -.-> RS
    
    %% Styling
    classDef detection fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef classification fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef errortype fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef severity fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef strategy fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef immediate fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef advanced fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef recovery fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef execution fill:#e8eaf6,stroke:#283593,stroke-width:2px
    classDef learning fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) fill:#fafafa,stroke:#424242,stroke-width:2px
    
    class ME,HD,ED,PD detection
    class EC classification
    class SYS,AGT,LLM,MEM,NET errortype
    class LOW,MED,HIGH,CRIT severity
    class RS strategy
    class RT,FB,CL,CF immediate
    class DR,LB,FS,ER advanced
    class AR,AS,AC,LR,LF,LC,MR,MF,MC recovery
    class RE,VM,RR execution
    class LA,PS,PA,IA learning
    class RM,AL,DG,RD Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)
```

## Error Handling & Recovery Overview

### 🚨 **Error Detection Layer**
Continuous Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) of system health and performance:
- **Monitor Events**: Real-time event tracking across all components
- **Health Detector**: Component health assessment and vital signs
- **Error Detector**: Exception catching and error identification  
- **Performance Detector**: Degradation detection and threshold Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)

### 🏷️ **Error Classification System**

#### **Error Types**
```python
error_types = {
    "system_errors": {
        "description": "Infrastructure and platform issues",
        "examples": ["out_of_memory", "disk_full", "cpu_overload"],
        "recovery_priority": "high"
    },
    "agent_errors": {
        "description": "Individual agent failures",
        "examples": ["agent_crash", "infinite_loop", "stuck_processing"],
        "recovery_priority": "medium"
    },
    "llm_errors": {
        "description": "LLM provider issues",
        "examples": ["api_timeout", "rate_limit", "service_unavailable"],
        "recovery_priority": "high"
    },
    "memory_errors": {
        "description": "Storage and caching issues",
        "examples": ["cache_miss", "database_connection", "corruption"],
        "recovery_priority": "medium"
    },
    "network_errors": {
        "description": "Connectivity and communication issues",
        "examples": ["connection_timeout", "dns_failure", "ssl_error"],
        "recovery_priority": "high"
    }
}
```

#### **Severity Levels**
```python
severity_mapping = {
    "low": {
        "threshold": 0.2,
        "action": "log_and_monitor",
        "escalation_time": 300  # 5 minutes
    },
    "medium": {
        "threshold": 0.5,
        "action": "automatic_recovery",
        "escalation_time": 60   # 1 minute
    },
    "high": {
        "threshold": 0.8,
        "action": "immediate_intervention",
        "escalation_time": 10   # 10 seconds
    },
    "critical": {
        "threshold": 1.0,
        "action": "emergency_protocol",
        "escalation_time": 0    # Immediate
    }
}
```

## Recovery Strategies

### **Immediate Recovery Actions**

#### **Automatic Retry with Exponential Backoff**
```python
async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Intelligent retry with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate delay with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
            logger.info(f"Retrying in {delay:.2f} seconds...")
            
            await asyncio.sleep(delay)
    
    raise Exception(f"Max retries ({max_retries}) exceeded")
```

#### **Circuit Breaker Pattern**
```python
class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: Attempting recovery (HALF_OPEN)")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func()
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker: Recovery successful (CLOSED)")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker: OPEN due to {self.failure_count} failures")
            
            raise e
```

### **Component-Specific Recovery**

#### **Agent Recovery**
```python
class AgentRecoveryManager:
    """Manages agent recovery and substitution"""
    
    async def recover_agent(self, agent_name, error_info):
        """Multi-strategy agent recovery"""
        
        recovery_strategies = [
            self.restart_agent,
            self.substitute_agent,
            self.reconfigure_agent
        ]
        
        for strategy in recovery_strategies:
            try:
                result = await strategy(agent_name, error_info)
                if result.success:
                    logger.info(f"Agent {agent_name} recovered using {strategy.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
        
        # If all strategies fail, escalate
        await self.escalate_agent_failure(agent_name, error_info)
    
    async def restart_agent(self, agent_name, error_info):
        """Restart the failed agent"""
        logger.info(f"Restarting agent: {agent_name}")
        
        # Graceful shutdown
        await self.shutdown_agent(agent_name)
        
        # Wait for cleanup
        await asyncio.sleep(2)
        
        # Restart with fresh configuration
        new_agent = await self.create_agent(agent_name)
        
        return RecoveryResult(success=True, agent=new_agent)
    
    async def substitute_agent(self, agent_name, error_info):
        """Substitute with backup agent"""
        logger.info(f"Substituting agent: {agent_name}")
        
        backup_agent = await self.get_backup_agent(agent_name)
        if backup_agent:
            await self.activate_backup_agent(backup_agent)
            return RecoveryResult(success=True, agent=backup_agent)
        
        raise Exception(f"No backup agent available for {agent_name}")
```

#### **LLM Provider Recovery**
```python
class LLMRecoveryManager:
    """Manages LLM provider failover and recovery"""
    
    def __init__(self):
        self.provider_health = {}
        self.fallback_order = [
            "openai",
            "anthropic", 
            "google",
            "local_ollama"
        ]
    
    async def recover_llm_request(self, request, failed_provider):
        """Recover failed LLM request"""
        
        # Mark provider as unhealthy
        self.provider_health[failed_provider] = "unhealthy"
        
        # Try fallback providers
        for provider in self.fallback_order:
            if provider == failed_provider:
                continue
                
            if self.provider_health.get(provider, "healthy") == "healthy":
                try:
                    logger.info(f"Attempting LLM recovery with provider: {provider}")
                    response = await self.call_provider(provider, request)
                    
                    logger.info(f"LLM recovery successful with {provider}")
                    return response
                    
                except Exception as e:
                    logger.warning(f"LLM provider {provider} also failed: {e}")
                    self.provider_health[provider] = "unhealthy"
        
        # If all providers fail, use cached response
        logger.warning("All LLM providers failed, attempting cache fallback")
        return await self.get_cached_response(request)
    
    async def health_check_providers(self):
        """Periodic health check for LLM providers"""
        
        for provider in self.fallback_order:
            try:
                # Simple health check request
                await self.call_provider(provider, {"text": "health check"})
                self.provider_health[provider] = "healthy"
            except Exception:
                self.provider_health[provider] = "unhealthy"
```

### **Memory System Recovery**
```python
class MemoryRecoveryManager:
    """Manages memory system recovery"""
    
    async def recover_memory_system(self, error_type, error_info):
        """Comprehensive memory system recovery"""
        
        if error_type == "redis_connection":
            return await self.recover_redis_connection()
        elif error_type == "vector_corruption":
            return await self.recover_vector_store()
        elif error_type == "memory_leak":
            return await self.cleanup_memory_leak()
        else:
            return await self.full_memory_recovery()
    
    async def recover_redis_connection(self):
        """Recover Redis connection issues"""
        
        # Try to reconnect
        try:
            await self.redis_client.ping()
            logger.info("Redis connection recovered")
            return RecoveryResult(success=True)
        except Exception:
            logger.warning("Redis still unavailable, switching to backup")
            
            # Switch to backup Redis instance
            backup_redis = await self.get_backup_redis()
            if backup_redis:
                self.redis_client = backup_redis
                return RecoveryResult(success=True)
            
            # If no backup, use in-memory fallback
            logger.warning("No Redis backup, using in-memory cache")
            self.use_memory_fallback()
            return RecoveryResult(success=True, degraded=True)
    
    async def recover_vector_store(self):
        """Recover vector store corruption"""
        
        logger.info("Recovering vector store from corruption")
        
        # Backup current corrupted store
        await self.backup_corrupted_store()
        
        # Restore from last known good backup
        restore_success = await self.restore_from_backup()
        
        if restore_success:
            logger.info("Vector store restored from backup")
            return RecoveryResult(success=True)
        
        # If restore fails, rebuild from source
        logger.warning("Backup restore failed, rebuilding vector store")
        await self.rebuild_vector_store()
        return RecoveryResult(success=True, degraded=True)
```

## Crisis Detection & Response

### **Confidence Crisis Detection**
```python
class CrisisDetector:
    """Detects system-wide crisis situations"""
    
    def __init__(self):
        self.crisis_thresholds = {
            "confidence_drop": 0.3,      # Overall confidence below 30%
            "error_rate_spike": 0.15,    # Error rate above 15%
            "response_time_spike": 5.0,  # Response time above 5 seconds
            "agent_failure_rate": 0.25   # 25% of agents failing
        }
    
    async def detect_crisis(self):
        """Continuous crisis detection"""
        
        metrics = await self.collect_system_metrics()
        
        crisis_indicators = {
            "confidence_drop": metrics["avg_confidence"] < self.crisis_thresholds["confidence_drop"],
            "error_rate_spike": metrics["error_rate"] > self.crisis_thresholds["error_rate_spike"],
            "response_time_spike": metrics["avg_response_time"] > self.crisis_thresholds["response_time_spike"],
            "agent_failure_rate": metrics["failed_agents"] / metrics["total_agents"] > self.crisis_thresholds["agent_failure_rate"]
        }
        
        active_indicators = [k for k, v in crisis_indicators.items() if v]
        
        if len(active_indicators) >= 2:  # Multiple indicators = crisis
            await self.trigger_crisis_response(active_indicators, metrics)
            return True
        
        return False
    
    async def trigger_crisis_response(self, indicators, metrics):
        """Comprehensive crisis response protocol"""
        
        logger.error(f"🚨 SYSTEM CRISIS DETECTED: {indicators}")
        
        # Immediate protective measures
        await self.enable_conservative_mode()
        await self.increase_redundancy()
        await self.defer_non_critical_operations()
        
        # Emergency notifications
        await self.notify_administrators(indicators, metrics)
        await self.log_crisis_event(indicators, metrics)
        
        # Adaptive responses based on crisis type
        if "confidence_drop" in indicators:
            await self.enable_ensemble_methods()
            await self.increase_verification_requirements()
        
        if "error_rate_spike" in indicators:
            await self.enable_circuit_breakers()
        
        if "response_time_spike" in indicators:
            await self.scale_up_resources()
            await self.optimize_processing_pipeline()
        
        if "agent_failure_rate" in indicators:
            await self.activate_backup_agents()
            await self.redistribute_workload()
```

This error handling and recovery system ensures:
- ✅ **Proactive Detection**: Continuous Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) before issues become critical
- ✅ **Intelligent Classification**: Smart error categorization and severity assessment
- ✅ **Multi-Strategy Recovery**: Multiple recovery approaches for each error type
- ✅ **Crisis Management**: System-wide crisis detection and emergency protocols
- ✅ **Learning Integration**: Pattern recognition to prevent future issues

high-quality for production deployment and AWS MAP program requirements! 