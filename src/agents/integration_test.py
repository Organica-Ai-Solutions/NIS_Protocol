"""
Integration Test for Hybrid Agent Architecture

This module tests the complete hybrid agent system including:
- LLM + Scientific pipeline integration
- Agent routing and context sharing  
- Physics validation and signal processing
- End-to-end task processing workflow

Tests the full Laplace → KAN → PINN → LLM pipeline.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any

# Import all components
from .hybrid_agent_core import (
    MetaCognitiveProcessor, CuriosityEngine, ValidationAgent,
    LLMProvider, ProcessingLayer
)
from .agent_router import (
    AgentRouter, NISContextBus, TaskRequest, TaskType, 
    AgentCapabilities, RoutingStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSystemIntegrationTest:
    """Complete integration test for the hybrid agent system."""
    
    def __init__(self):
        self.context_bus = NISContextBus()
        self.router = AgentRouter(self.context_bus)
        self.agents = {}
        
        # Test data sets
        self.test_signals = {
            "simple_sine": np.sin(np.linspace(0, 4*np.pi, 100)).tolist(),
            "noisy_signal": (np.sin(np.linspace(0, 2*np.pi, 50)) + 0.1*np.random.randn(50)).tolist(),
            "complex_pattern": [1.2, 2.1, 1.8, 2.3, 1.9, 2.4, 1.7, 2.2] * 8
        }
        
        self.test_results = {}
    
    async def setup_agents(self):
        """Set up all hybrid agents and register them."""
        logger.info("🔧 Setting up hybrid agents...")
        
        # Create agents
        self.agents["metacognitive"] = MetaCognitiveProcessor("metacog_test_001")
        self.agents["curiosity"] = CuriosityEngine("curiosity_test_001") 
        self.agents["validator"] = ValidationAgent("validator_test_001")
        
        # Register with router
        await self._register_all_agents()
        
        logger.info(f"✅ Set up {len(self.agents)} hybrid agents")
    
    async def _register_all_agents(self):
        """Register all agents with their capabilities."""
        
        # MetaCognitive Agent
        self.router.register_agent(
            self.agents["metacognitive"],
            AgentCapabilities(
                agent_id="metacog_test_001",
                agent_type="metacognitive",
                supported_tasks={TaskType.ANALYSIS, TaskType.OPTIMIZATION, TaskType.REASONING},
                llm_provider=LLMProvider.GPT4,
                processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN],
                specializations=["optimization", "self-assessment", "system-analysis"]
            )
        )
        
        # Curiosity Agent
        self.router.register_agent(
            self.agents["curiosity"],
            AgentCapabilities(
                agent_id="curiosity_test_001",
                agent_type="curiosity",
                supported_tasks={TaskType.EXPLORATION, TaskType.ANALYSIS},
                llm_provider=LLMProvider.GEMINI,
                processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN],
                specializations=["novelty-detection", "exploration", "pattern-discovery"]
            )
        )
        
        # Validation Agent
        self.router.register_agent(
            self.agents["validator"],
            AgentCapabilities(
                agent_id="validator_test_001",
                agent_type="validation",
                supported_tasks={TaskType.VALIDATION, TaskType.REASONING, TaskType.PHYSICS_SIMULATION},
                llm_provider=LLMProvider.CLAUDE4,
                processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN, ProcessingLayer.PINN],
                specializations=["physics-validation", "integrity-checking", "constraint-verification"]
            )
        )
    
    async def test_individual_agents(self):
        """Test each agent individually."""
        logger.info("🧪 Testing individual agents...")
        
        test_data = self.test_signals["simple_sine"]
        
        for agent_name, agent in self.agents.items():
            try:
                response = await agent.process_hybrid_request(
                    test_data,
                    f"Test {agent_name} processing of sine wave signal"
                )
                
                # Validate response structure
                assert "scientific_validation" in response
                assert "llm_response" in response
                assert "integrity_score" in response["scientific_validation"]
                
                integrity = response["scientific_validation"]["integrity_score"]
                logger.info(f"   {agent_name}: Integrity score {integrity:.3f}")
                
                self.test_results[f"{agent_name}_individual"] = {
                    "success": True,
                    "integrity_score": integrity,
                    "response_time": response.get("hybrid_metadata", {}).get("total_processing_time", 0)
                }
                
            except Exception as e:
                logger.error(f"   {agent_name} failed: {e}")
                self.test_results[f"{agent_name}_individual"] = {"success": False, "error": str(e)}
        
        logger.info("✅ Individual agent tests completed")
    
    async def test_routing_system(self):
        """Test the routing system with different task types."""
        logger.info("🔀 Testing routing system...")
        
        test_tasks = [
            TaskRequest(
                task_id="route_test_001",
                task_type=TaskType.OPTIMIZATION,
                input_data=self.test_signals["noisy_signal"],
                description="Optimize signal processing parameters for noisy data",
                priority=8
            ),
            TaskRequest(
                task_id="route_test_002",
                task_type=TaskType.VALIDATION,
                input_data=self.test_signals["complex_pattern"],
                description="Validate physics compliance of complex pattern data",
                preferred_llm=LLMProvider.CLAUDE4,
                processing_layers=[ProcessingLayer.PINN]
            ),
            TaskRequest(
                task_id="route_test_003", 
                task_type=TaskType.EXPLORATION,
                input_data=self.test_signals["simple_sine"],
                description="Explore novel features in sine wave for pattern discovery"
            )
        ]
        
        routing_results = []
        
        for task in test_tasks:
            try:
                agent, decision = await self.router.route_task(task)
                
                if agent:
                    logger.info(f"   {task.task_id} → {agent.agent_id} (confidence: {decision.confidence:.3f})")
                    
                    # Process the task
                    response = await agent.process_hybrid_request(task.input_data, task.description)
                    
                    routing_results.append({
                        "task_id": task.task_id,
                        "routed_to": agent.agent_id,
                        "confidence": decision.confidence,
                        "success": "error" not in response,
                        "integrity_score": response.get("scientific_validation", {}).get("integrity_score", 0)
                    })
                else:
                    logger.warning(f"   {task.task_id} → No suitable agent found")
                    routing_results.append({
                        "task_id": task.task_id,
                        "routed_to": None,
                        "success": False
                    })
                    
            except Exception as e:
                logger.error(f"   Routing test {task.task_id} failed: {e}")
                routing_results.append({
                    "task_id": task.task_id,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["routing_system"] = routing_results
        logger.info("✅ Routing system tests completed")
    
    async def test_context_sharing(self):
        """Test context bus and shared memory."""
        logger.info("🧠 Testing context sharing...")
        
        # Store test data in context
        self.context_bus.set_global_context("test_signal", self.test_signals["simple_sine"])
        self.context_bus.store_memory("test_memory_001", {
            "signal_type": "sine_wave",
            "frequency": 1.0,
            "amplitude": 1.0,
            "analysis_notes": "Clean sine wave for testing"
        })
        
        # Update agent states
        for agent_name, agent in self.agents.items():
            self.context_bus.update_agent_state(agent.agent_id, {
                "last_test": "context_sharing",
                "status": "active",
                "processing_capabilities": agent_name
            })
        
        # Verify context retrieval
        retrieved_signal = self.context_bus.get_global_context("test_signal")
        retrieved_memory = self.context_bus.retrieve_memory("test_memory_001")
        
        context_test_results = {
            "signal_stored_correctly": retrieved_signal == self.test_signals["simple_sine"],
            "memory_stored_correctly": retrieved_memory is not None,
            "agent_states_updated": len(self.context_bus.agent_states) == len(self.agents),
            "context_summary": self.context_bus.get_context_summary()
        }
        
        self.test_results["context_sharing"] = context_test_results
        logger.info("✅ Context sharing tests completed")
    
    async def test_scientific_pipeline(self):
        """Test the complete scientific processing pipeline."""
        logger.info("🔬 Testing scientific pipeline...")
        
        # Test with validation agent (has full pipeline)
        validator = self.agents["validator"]
        
        pipeline_tests = []
        
        for signal_name, signal_data in self.test_signals.items():
            try:
                response = await validator.process_hybrid_request(
                    signal_data,
                    f"Complete scientific analysis of {signal_name}"
                )
                
                sci_validation = response.get("scientific_validation", {})
                
                pipeline_test = {
                    "signal_name": signal_name,
                    "laplace_processed": sci_validation.get("laplace_status") == "processed",
                    "kan_patterns": sci_validation.get("kan_patterns", 0),
                    "physics_valid": sci_validation.get("physics_valid", False),
                    "integrity_score": sci_validation.get("integrity_score", 0),
                    "processing_time": sci_validation.get("processing_time", 0)
                }
                
                pipeline_tests.append(pipeline_test)
                logger.info(f"   {signal_name}: Integrity {pipeline_test['integrity_score']:.3f}, "
                           f"KAN patterns: {pipeline_test['kan_patterns']}")
                
            except Exception as e:
                logger.error(f"   Pipeline test {signal_name} failed: {e}")
                pipeline_tests.append({
                    "signal_name": signal_name,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["scientific_pipeline"] = pipeline_tests
        logger.info("✅ Scientific pipeline tests completed")
    
    async def test_load_balancing(self):
        """Test load balancing and performance monitoring."""
        logger.info("⚖️ Testing load balancing...")
        
        # Simulate multiple concurrent requests
        concurrent_tasks = []
        
        for i in range(6):  # More tasks than agents to test load balancing
            task = TaskRequest(
                task_id=f"load_test_{i:03d}",
                task_type=TaskType.ANALYSIS,
                input_data=self.test_signals["simple_sine"],
                description=f"Load test analysis request {i}"
            )
            concurrent_tasks.append(task)
        
        # Process tasks concurrently
        async def process_task(task):
            agent, decision = await self.router.route_task(task, RoutingStrategy.LOAD_BALANCED)
            if agent:
                response = await agent.process_hybrid_request(task.input_data, task.description)
                return {
                    "task_id": task.task_id,
                    "agent_id": agent.agent_id,
                    "success": "error" not in response,
                    "processing_time": response.get("hybrid_metadata", {}).get("total_processing_time", 0)
                }
            return {"task_id": task.task_id, "success": False}
        
        # Run concurrent tasks
        results = await asyncio.gather(*[process_task(task) for task in concurrent_tasks])
        
        # Analyze load distribution
        agent_task_counts = {}
        for result in results:
            if result.get("agent_id"):
                agent_task_counts[result["agent_id"]] = agent_task_counts.get(result["agent_id"], 0) + 1
        
        load_test_results = {
            "total_tasks": len(concurrent_tasks),
            "successful_tasks": sum(1 for r in results if r.get("success")),
            "agent_distribution": agent_task_counts,
            "load_balanced": len(set(agent_task_counts.keys())) > 1,  # Tasks distributed across multiple agents
            "router_status": self.router.get_router_status()
        }
        
        self.test_results["load_balancing"] = load_test_results
        logger.info(f"   Tasks distributed: {agent_task_counts}")
        logger.info("✅ Load balancing tests completed")
    
    async def run_complete_test(self):
        """Run the complete integration test suite."""
        logger.info("🚀 Starting Hybrid Agent Integration Test")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_agents()
            
            # Run all test suites
            await self.test_individual_agents()
            await self.test_routing_system()
            await self.test_context_sharing()
            await self.test_scientific_pipeline()
            await self.test_load_balancing()
            
            # Generate summary report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        
        return True
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 HYBRID AGENT INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, results in self.test_results.items():
            logger.info(f"\n📊 {test_name.upper()}:")
            
            if isinstance(results, list):
                for result in results:
                    total_tests += 1
                    if result.get("success", True):
                        passed_tests += 1
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    logger.info(f"   {result.get('task_id', result.get('signal_name', 'test'))}: {status}")
            
            elif isinstance(results, dict):
                if "success" in results:
                    total_tests += 1
                    if results["success"]:
                        passed_tests += 1
                        logger.info("   ✅ PASS")
                    else:
                        logger.info(f"   ❌ FAIL: {results.get('error', 'Unknown error')}")
                else:
                    # Complex results - analyze sub-components
                    for key, value in results.items():
                        if isinstance(value, bool):
                            total_tests += 1
                            if value:
                                passed_tests += 1
                                logger.info(f"   {key}: ✅ PASS")
                            else:
                                logger.info(f"   {key}: ❌ FAIL")
                        else:
                            logger.info(f"   {key}: {value}")
        
        # Final summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\n🎯 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {total_tests - passed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("   🎉 INTEGRATION TEST: PASSED")
        else:
            logger.info("   ⚠️  INTEGRATION TEST: NEEDS ATTENTION")
        
        # System status
        router_status = self.router.get_router_status()
        logger.info(f"\n📈 SYSTEM STATUS:")
        logger.info(f"   Active Agents: {router_status['active_agents']}")
        logger.info(f"   Routing Requests: {router_status['routing_stats']['total_requests']}")
        logger.info(f"   Context Bus: {router_status['context_summary']}")

# Main test runner
async def run_integration_test():
    """Run the complete hybrid agent integration test."""
    test_suite = HybridSystemIntegrationTest()
    success = await test_suite.run_complete_test()
    return success

if __name__ == "__main__":
    asyncio.run(run_integration_test()) 