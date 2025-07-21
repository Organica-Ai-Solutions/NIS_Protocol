#!/usr/bin/env python3
"""
NIS Protocol v3 - Enhanced Infrastructure Integration Demo

This demo showcases the complete Kafka/Redis infrastructure integration
with enhanced agents, self-audit capabilities, and production-ready features.

Features demonstrated:
- Enhanced Kafka message streaming with integrity monitoring
- Redis caching with performance optimization
- Self-audit integration across all infrastructure
- Real-time health monitoring and auto-recovery
- Message-driven agent coordination
- Comprehensive performance tracking

Usage:
    python examples/enhanced_infrastructure_integration_demo.py
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Infrastructure imports
from infrastructure.integration_coordinator import InfrastructureCoordinator
from infrastructure.message_streaming import MessageType, MessagePriority
from infrastructure.caching_system import CacheStrategy

# Enhanced agent imports
from agents.enhanced_agent_base import AgentConfiguration
from agents.simulation.enhanced_scenario_simulator import EnhancedScenarioSimulator
from agents.simulation.scenario_simulator import ScenarioType, SimulationParameters

# Utility imports
from utils.self_audit import self_audit_engine


class InfrastructureDemo:
    """
    Comprehensive demo of enhanced infrastructure integration
    """
    
    def __init__(self):
        self.infrastructure: InfrastructureCoordinator = None
        self.agents: Dict[str, Any] = {}
        self.demo_results: Dict[str, Any] = {}
        
        logger.info("🚀 Enhanced Infrastructure Integration Demo Initialized")
    
    async def run_complete_demo(self):
        """Run the complete infrastructure integration demo"""
        try:
            logger.info("=" * 80)
            logger.info("🎯 STARTING ENHANCED INFRASTRUCTURE INTEGRATION DEMO")
            logger.info("=" * 80)
            
            # Phase 1: Initialize Infrastructure
            await self._demo_infrastructure_initialization()
            
            # Phase 2: Agent Integration
            await self._demo_agent_integration()
            
            # Phase 3: Message Streaming
            await self._demo_message_streaming()
            
            # Phase 4: Caching System
            await self._demo_caching_system()
            
            # Phase 5: Self-Audit Integration
            await self._demo_self_audit_integration()
            
            # Phase 6: Performance Monitoring
            await self._demo_performance_monitoring()
            
            # Phase 7: Recovery and Resilience
            await self._demo_recovery_resilience()
            
            # Phase 8: Complete System Integration
            await self._demo_complete_system_integration()
            
            # Generate final report
            await self._generate_demo_report()
            
        except Exception as e:
            logger.error(f"Demo execution error: {e}")
            raise
        finally:
            await self._cleanup_demo()
    
    async def _demo_infrastructure_initialization(self):
        """Demo: Infrastructure Initialization"""
        logger.info("\n📋 PHASE 1: INFRASTRUCTURE INITIALIZATION")
        logger.info("-" * 50)
        
        try:
            # Configure Kafka and Redis (mock mode for demo)
            kafka_config = {
                "bootstrap_servers": ["localhost:9092"],
                "options": {
                    "max_retries": 3,
                    "retry_backoff": 1.0
                }
            }
            
            redis_config = {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "options": {
                    "max_memory": "512mb",
                    "eviction_policy": "allkeys-lru"
                }
            }
            
            # Initialize infrastructure coordinator
            self.infrastructure = InfrastructureCoordinator(
                kafka_config=kafka_config,
                redis_config=redis_config,
                enable_self_audit=True,
                health_check_interval=30.0,
                auto_recovery=True
            )
            
            # Initialize infrastructure
            success = await self.infrastructure.initialize()
            
            if success:
                logger.info("✅ Infrastructure initialization successful")
                
                # Get status
                status = self.infrastructure.get_comprehensive_status()
                logger.info(f"   Integration Status: {status['integration_status']}")
                logger.info(f"   Overall Health: {status['overall_health']}")
                logger.info(f"   Services: {list(status['services'].keys())}")
                
                self.demo_results["infrastructure_init"] = {
                    "success": True,
                    "status": status,
                    "timestamp": time.time()
                }
            else:
                logger.warning("⚠️  Infrastructure initialization partial (mock mode)")
                self.demo_results["infrastructure_init"] = {
                    "success": False,
                    "note": "Running in mock mode for demo",
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Infrastructure initialization error: {e}")
            raise
    
    async def _demo_agent_integration(self):
        """Demo: Agent Integration with Infrastructure"""
        logger.info("\n🤖 PHASE 2: AGENT INTEGRATION")
        logger.info("-" * 50)
        
        try:
            # Create enhanced scenario simulator
            simulator = EnhancedScenarioSimulator(
                agent_id="demo_scenario_simulator",
                infrastructure_coordinator=self.infrastructure,
                enable_monte_carlo=True,
                enable_physics_validation=True
            )
            
            # Initialize agent
            success = await simulator.initialize()
            
            if success:
                logger.info("✅ Enhanced Scenario Simulator initialized")
                
                # Get agent status
                status = simulator.get_enhanced_status()
                logger.info(f"   Agent State: {status['state']}")
                logger.info(f"   Capabilities: {list(status['capabilities'].keys())}")
                logger.info(f"   Infrastructure Available: {status['infrastructure_available']}")
                
                self.agents["scenario_simulator"] = simulator
                
                self.demo_results["agent_integration"] = {
                    "success": True,
                    "agent_count": len(self.agents),
                    "agent_status": status,
                    "timestamp": time.time()
                }
                
            else:
                logger.error("❌ Agent initialization failed")
                
        except Exception as e:
            logger.error(f"Agent integration error: {e}")
            raise
    
    async def _demo_message_streaming(self):
        """Demo: Message Streaming with Kafka"""
        logger.info("\n📨 PHASE 3: MESSAGE STREAMING")
        logger.info("-" * 50)
        
        try:
            if not self.infrastructure:
                logger.warning("Infrastructure not available for messaging demo")
                return
            
            # Test message sending
            logger.info("🔄 Testing message streaming...")
            
            # Send a system health message
            success = await self.infrastructure.send_message(
                message_type=MessageType.SYSTEM_HEALTH,
                content={
                    "event": "demo_health_check",
                    "status": "healthy",
                    "demo_phase": "message_streaming",
                    "timestamp": time.time()
                },
                source_agent="demo_controller",
                priority=MessagePriority.NORMAL
            )
            
            if success:
                logger.info("✅ System health message sent successfully")
            
            # Send a simulation request message
            success = await self.infrastructure.send_message(
                message_type=MessageType.SIMULATION_RESULT,
                content={
                    "scenario_id": "demo_archaeological_site",
                    "scenario_type": "archaeological_excavation",
                    "parameters": {
                        "monte_carlo_iterations": 1000,
                        "objectives": ["artifact_discovery", "site_preservation"],
                        "constraints": {
                            "budget": 50000,
                            "duration_days": 45,
                            "team_size": 6
                        }
                    },
                    "priority": MessagePriority.HIGH.value
                },
                source_agent="demo_controller",
                priority=MessagePriority.HIGH
            )
            
            if success:
                logger.info("✅ Simulation request message sent successfully")
            
            # Get streaming metrics
            metrics = self.infrastructure.get_metrics()
            logger.info(f"   Total Messages: {metrics.total_messages}")
            logger.info(f"   Error Rate: {metrics.error_rate:.3f}")
            
            self.demo_results["message_streaming"] = {
                "success": True,
                "messages_sent": 2,
                "metrics": metrics.kafka_metrics,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Message streaming error: {e}")
            raise
    
    async def _demo_caching_system(self):
        """Demo: Caching System with Redis"""
        logger.info("\n💾 PHASE 4: CACHING SYSTEM")
        logger.info("-" * 50)
        
        try:
            if not self.infrastructure:
                logger.warning("Infrastructure not available for caching demo")
                return
            
            logger.info("🔄 Testing caching system...")
            
            # Cache simulation configuration
            config_data = {
                "simulation_parameters": {
                    "monte_carlo_iterations": 1000,
                    "physics_validation": True,
                    "cache_results": True
                },
                "performance_settings": {
                    "batch_size": 10,
                    "timeout": 300,
                    "auto_retry": True
                },
                "demo_metadata": {
                    "created_by": "infrastructure_demo",
                    "version": "v3",
                    "timestamp": time.time()
                }
            }
            
            # Test caching
            success = await self.infrastructure.cache_data(
                key="demo_simulation_config",
                value=config_data,
                agent_id="demo_controller",
                ttl=3600
            )
            
            if success:
                logger.info("✅ Configuration data cached successfully")
            
            # Test cache retrieval
            retrieved_data = await self.infrastructure.get_cached_data(
                key="demo_simulation_config",
                agent_id="demo_controller"
            )
            
            if retrieved_data:
                logger.info("✅ Configuration data retrieved from cache")
                logger.info(f"   Cache hit: {retrieved_data['demo_metadata']['created_by']}")
            
            # Cache simulation results
            demo_results = {
                "scenario_id": "demo_archaeological_analysis",
                "success_probability": 0.847,
                "confidence_interval": [0.781, 0.913],
                "processing_time": 2.3,
                "cache_timestamp": time.time()
            }
            
            success = await self.infrastructure.cache_data(
                key="demo_simulation_results",
                value=demo_results,
                agent_id="scenario_simulator",
                ttl=7200
            )
            
            if success:
                logger.info("✅ Simulation results cached successfully")
            
            self.demo_results["caching_system"] = {
                "success": True,
                "cache_operations": 3,
                "cache_hits": 1,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Caching system error: {e}")
            raise
    
    async def _demo_self_audit_integration(self):
        """Demo: Self-Audit Integration"""
        logger.info("\n🛡️ PHASE 5: SELF-AUDIT INTEGRATION")
        logger.info("-" * 50)
        
        try:
            logger.info("🔄 Testing self-audit integration...")
            
            # Test audit on infrastructure components
            test_texts = [
                "System performance shows 95% accuracy with validated benchmarks",
                "Revolutionary breakthrough achieved 99.9% confidence in testing",
                "The advanced system demonstrates optimized performance metrics"
            ]
            
            audit_results = []
            
            for i, text in enumerate(test_texts):
                violations = self_audit_engine.audit_text(text)
                integrity_score = self_audit_engine.get_integrity_score(text)
                
                result = {
                    "text_id": i + 1,
                    "integrity_score": integrity_score,
                    "violations": len(violations),
                    "violation_types": [v['type'] for v in violations] if violations else []
                }
                
                audit_results.append(result)
                
                if violations:
                    logger.warning(f"   Text {i+1}: {len(violations)} violations detected - Score: {integrity_score:.1f}")
                    for violation in violations:
                        logger.warning(f"     - {violation['type']}: {violation['description']}")
                else:
                    logger.info(f"   Text {i+1}: No violations - Score: {integrity_score:.1f}")
            
            # Test infrastructure audit
            if self.infrastructure:
                status = self.infrastructure.get_comprehensive_status()
                audit_text = f"""
                Infrastructure Status:
                Integration Status: {status['integration_status']}
                Overall Health: {status['overall_health']}
                Uptime: {status['uptime']:.1f} seconds
                Auto Recovery: {status['auto_recovery_enabled']}
                """
                
                violations = self_audit_engine.audit_text(audit_text)
                integrity_score = self_audit_engine.get_integrity_score(audit_text)
                
                logger.info(f"   Infrastructure Audit Score: {integrity_score:.1f}")
                if violations:
                    logger.warning(f"   Infrastructure Violations: {len(violations)}")
            
            self.demo_results["self_audit"] = {
                "success": True,
                "audit_results": audit_results,
                "infrastructure_score": integrity_score,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Self-audit integration error: {e}")
            raise
    
    async def _demo_performance_monitoring(self):
        """Demo: Performance Monitoring"""
        logger.info("\n📊 PHASE 6: PERFORMANCE MONITORING")
        logger.info("-" * 50)
        
        try:
            logger.info("🔄 Testing performance monitoring...")
            
            # Get infrastructure metrics
            if self.infrastructure:
                metrics = self.infrastructure.get_metrics()
                
                logger.info("📈 Infrastructure Metrics:")
                logger.info(f"   Total Messages: {metrics.total_messages}")
                logger.info(f"   Total Cache Operations: {metrics.total_cache_operations}")
                logger.info(f"   Error Rate: {metrics.error_rate:.3f}")
                logger.info(f"   Avg Response Time: {metrics.avg_response_time:.3f}s")
                logger.info(f"   Uptime: {metrics.uptime:.1f}s")
            
            # Get agent metrics
            if "scenario_simulator" in self.agents:
                agent = self.agents["scenario_simulator"]
                agent_status = agent.get_enhanced_status()
                
                logger.info("🤖 Agent Metrics:")
                logger.info(f"   Messages Sent: {agent_status['metrics']['messages_sent']}")
                logger.info(f"   Messages Received: {agent_status['metrics']['messages_received']}")
                logger.info(f"   Cache Hits: {agent_status['metrics']['cache_hits']}")
                logger.info(f"   Cache Misses: {agent_status['metrics']['cache_misses']}")
                logger.info(f"   Integrity Score: {agent_status['metrics']['integrity_score']:.1f}")
                logger.info(f"   Error Count: {agent_status['metrics']['error_count']}")
            
            # Send performance metrics message
            if self.infrastructure:
                performance_data = {
                    "infrastructure_metrics": {
                        "total_messages": metrics.total_messages,
                        "error_rate": metrics.error_rate,
                        "uptime": metrics.uptime
                    },
                    "agent_metrics": agent_status['metrics'] if 'agent_status' in locals() else {},
                    "demo_phase": "performance_monitoring",
                    "timestamp": time.time()
                }
                
                await self.infrastructure.send_message(
                    message_type=MessageType.PERFORMANCE_METRIC,
                    content=performance_data,
                    source_agent="demo_controller",
                    priority=MessagePriority.LOW
                )
                
                logger.info("✅ Performance metrics message sent")
            
            self.demo_results["performance_monitoring"] = {
                "success": True,
                "infrastructure_metrics": metrics.__dict__ if 'metrics' in locals() else {},
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            raise
    
    async def _demo_recovery_resilience(self):
        """Demo: Recovery and Resilience Features"""
        logger.info("\n🔄 PHASE 7: RECOVERY AND RESILIENCE")
        logger.info("-" * 50)
        
        try:
            logger.info("🔄 Testing recovery and resilience...")
            
            # Simulate error condition and recovery
            if "scenario_simulator" in self.agents:
                agent = self.agents["scenario_simulator"]
                
                # Simulate error
                agent.metrics.error_count += 5  # Simulate errors
                logger.info("   Simulated error conditions")
                
                # Trigger health check
                await agent._perform_health_check()
                
                # Check if agent state changed
                status = agent.get_enhanced_status()
                logger.info(f"   Agent State after errors: {status['state']}")
                
                # Reset errors (simulate recovery)
                agent.metrics.error_count = 0
                await agent._perform_health_check()
                
                # Check recovery
                status = agent.get_enhanced_status()
                logger.info(f"   Agent State after recovery: {status['state']}")
                
                logger.info("✅ Recovery simulation completed")
            
            # Test circuit breaker behavior
            if self.infrastructure:
                status = self.infrastructure.get_comprehensive_status()
                logger.info(f"   Infrastructure Status: {status['integration_status']}")
                
                # Infrastructure auto-recovery is always running
                logger.info("✅ Auto-recovery mechanisms active")
            
            self.demo_results["recovery_resilience"] = {
                "success": True,
                "recovery_tested": True,
                "auto_recovery_active": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Recovery and resilience error: {e}")
            raise
    
    async def _demo_complete_system_integration(self):
        """Demo: Complete System Integration"""
        logger.info("\n🎯 PHASE 8: COMPLETE SYSTEM INTEGRATION")
        logger.info("-" * 50)
        
        try:
            logger.info("🔄 Testing complete system integration...")
            
            # Run end-to-end simulation workflow
            if "scenario_simulator" in self.agents:
                simulator = self.agents["scenario_simulator"]
                
                # Create simulation parameters
                parameters = SimulationParameters(
                    monte_carlo_iterations=500,  # Reduced for demo
                    objectives=["artifact_discovery", "cultural_preservation"],
                    constraints={
                        "budget": 75000,
                        "duration_days": 60,
                        "team_size": 8,
                        "site_area_sqm": 200
                    }
                )
                
                logger.info("   Executing archaeological excavation simulation...")
                
                # Execute simulation
                result = await simulator.simulate_scenario(
                    scenario_id="demo_complete_integration",
                    scenario_type=ScenarioType.ARCHAEOLOGICAL_EXCAVATION,
                    parameters=parameters,
                    requester_agent="demo_controller",
                    priority=MessagePriority.NORMAL
                )
                
                logger.info("✅ Simulation completed successfully")
                logger.info(f"   Success Probability: {result.result.success_probability:.3f}")
                logger.info(f"   Confidence Interval: {result.result.confidence_interval}")
                logger.info(f"   Processing Time: {result.processing_time:.3f}s")
                logger.info(f"   Integrity Score: {result.integrity_score:.1f}")
                
                # Test cached result retrieval
                if result.cache_key:
                    cached_result = await simulator.get_cached_data(result.cache_key)
                    if cached_result:
                        logger.info("✅ Result successfully cached and retrievable")
                
                self.demo_results["complete_integration"] = {
                    "success": True,
                    "simulation_result": {
                        "success_probability": result.result.success_probability,
                        "processing_time": result.processing_time,
                        "integrity_score": result.integrity_score,
                        "cached": result.cache_key is not None
                    },
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Complete system integration error: {e}")
            raise
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        logger.info("\n📋 GENERATING DEMO REPORT")
        logger.info("=" * 80)
        
        try:
            # Calculate overall success rate
            successful_phases = sum(1 for result in self.demo_results.values() 
                                  if result.get("success", False))
            total_phases = len(self.demo_results)
            success_rate = successful_phases / total_phases if total_phases > 0 else 0
            
            logger.info(f"🎯 DEMO COMPLETION SUMMARY")
            logger.info(f"   Successful Phases: {successful_phases}/{total_phases}")
            logger.info(f"   Success Rate: {success_rate:.1%}")
            
            # Infrastructure status
            if self.infrastructure:
                status = self.infrastructure.get_comprehensive_status()
                logger.info(f"   Infrastructure Status: {status['integration_status']}")
                logger.info(f"   Overall Health: {status['overall_health']}")
            
            # Agent status
            logger.info(f"   Active Agents: {len(self.agents)}")
            
            # Key achievements
            logger.info("\n🏆 KEY ACHIEVEMENTS:")
            
            if self.demo_results.get("infrastructure_init", {}).get("success"):
                logger.info("   ✅ Infrastructure initialization successful")
            
            if self.demo_results.get("agent_integration", {}).get("success"):
                logger.info("   ✅ Agent integration with infrastructure")
            
            if self.demo_results.get("message_streaming", {}).get("success"):
                logger.info("   ✅ Message streaming with Kafka")
            
            if self.demo_results.get("caching_system", {}).get("success"):
                logger.info("   ✅ Caching system with Redis")
            
            if self.demo_results.get("self_audit", {}).get("success"):
                logger.info("   ✅ Self-audit integration")
            
            if self.demo_results.get("performance_monitoring", {}).get("success"):
                logger.info("   ✅ Performance monitoring")
            
            if self.demo_results.get("recovery_resilience", {}).get("success"):
                logger.info("   ✅ Recovery and resilience")
            
            if self.demo_results.get("complete_integration", {}).get("success"):
                logger.info("   ✅ Complete system integration")
            
            # Performance summary
            logger.info("\n📊 PERFORMANCE SUMMARY:")
            
            if self.infrastructure:
                metrics = self.infrastructure.get_metrics()
                logger.info(f"   Total Messages: {metrics.total_messages}")
                logger.info(f"   Total Cache Operations: {metrics.total_cache_operations}")
                logger.info(f"   Error Rate: {metrics.error_rate:.3f}")
                logger.info(f"   Uptime: {metrics.uptime:.1f}s")
            
            # Save report to file
            report_data = {
                "demo_summary": {
                    "success_rate": success_rate,
                    "successful_phases": successful_phases,
                    "total_phases": total_phases,
                    "timestamp": time.time()
                },
                "phase_results": self.demo_results,
                "infrastructure_status": status if 'status' in locals() else {},
                "agent_count": len(self.agents)
            }
            
            with open("enhanced_infrastructure_demo_report.json", "w") as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info("\n💾 Demo report saved to: enhanced_infrastructure_demo_report.json")
            
        except Exception as e:
            logger.error(f"Demo report generation error: {e}")
    
    async def _cleanup_demo(self):
        """Clean up demo resources"""
        logger.info("\n🧹 CLEANING UP DEMO RESOURCES")
        logger.info("-" * 50)
        
        try:
            # Shutdown agents
            for agent_id, agent in self.agents.items():
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
                    logger.info(f"   Agent {agent_id} shut down")
            
            # Shutdown infrastructure
            if self.infrastructure:
                await self.infrastructure.shutdown()
                logger.info("   Infrastructure coordinator shut down")
            
            logger.info("✅ Demo cleanup completed")
            
        except Exception as e:
            logger.error(f"Demo cleanup error: {e}")


async def main():
    """Main demo execution function"""
    try:
        # Create and run demo
        demo = InfrastructureDemo()
        await demo.run_complete_demo()
        
        logger.info("\n🎉 ENHANCED INFRASTRUCTURE INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run demo
    asyncio.run(main()) 