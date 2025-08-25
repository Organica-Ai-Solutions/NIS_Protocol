#!/usr/bin/env python3
"""
🧪 Comprehensive Agent Test Suite for NIS Protocol
Advanced testing framework for multi-agent coordination and deep agent integration

Features:
- Individual agent functionality testing
- Cross-agent communication validation
- LangChain integration testing
- Deep agent workflow testing
- Performance and load testing
- Integration with master orchestrator
- Real-world scenario simulations
"""

import asyncio
import logging
import time
import json
import unittest
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from pathlib import Path
import sys
import traceback

# Core testing framework
import pytest
from unittest.mock import Mock, AsyncMock, patch

# NIS components
try:
    from .master_agent_orchestrator import MasterAgentOrchestrator, TaskPriority, TaskRequest
    from .enhanced_langchain_integration import EnhancedLangChainIntegration, LangChainAgentConfig
    from .deep.planner import DeepAgentPlanner
except ImportError as e:
    print(f"Warning: Some agent components not available: {e}")

# Individual agents for testing
agent_test_imports = {}
try:
    from .memory.enhanced_memory_agent import EnhancedMemoryAgent
    agent_test_imports['memory'] = EnhancedMemoryAgent
except ImportError:
    agent_test_imports['memory'] = None

try:
    from .reasoning.unified_reasoning_agent import UnifiedReasoningAgent
    agent_test_imports['reasoning'] = UnifiedReasoningAgent
except ImportError:
    agent_test_imports['reasoning'] = None

try:
    from .physics.unified_physics_agent import UnifiedPhysicsAgent
    agent_test_imports['physics'] = UnifiedPhysicsAgent
except ImportError:
    agent_test_imports['physics'] = None

try:
    from .consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
    agent_test_imports['consciousness'] = EnhancedConsciousAgent
except ImportError:
    agent_test_imports['consciousness'] = None

try:
    from .research.web_search_agent import WebSearchAgent
    agent_test_imports['web_search'] = WebSearchAgent
except ImportError:
    agent_test_imports['web_search'] = None


class TestCategory(Enum):
    """Categories of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    LOAD = "load"
    SCENARIO = "scenario"
    LANGCHAIN = "langchain"
    DEEP_AGENT = "deep_agent"


@dataclass
class TestResult:
    """Result from a test execution"""
    test_name: str
    category: TestCategory
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuiteConfig:
    """Configuration for test suite execution"""
    categories: List[TestCategory] = None
    agents_to_test: List[str] = None
    include_performance: bool = True
    include_load_tests: bool = False
    max_concurrent_tests: int = 5
    test_timeout: float = 300.0
    enable_logging: bool = True
    log_level: str = "INFO"


class AgentTestFramework:
    """
    🧪 Advanced testing framework for NIS agents
    """
    
    def __init__(self, config: TestSuiteConfig = None):
        self.config = config or TestSuiteConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.enable_logging:
            logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Test results storage
        self.test_results: List[TestResult] = []
        self.test_metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "total_duration": 0.0
        }
        
        # Test environment
        self.test_orchestrator: Optional[MasterAgentOrchestrator] = None
        self.test_langchain_integration: Optional[EnhancedLangChainIntegration] = None
        self.test_agents: Dict[str, Any] = {}
        
        # Initialize test environment
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup the test environment"""
        try:
            self.logger.info("🔧 Setting up test environment")
            
            # Create test orchestrator
            self.test_orchestrator = MasterAgentOrchestrator(
                enable_deep_agent=True,
                enable_auto_discovery=False  # We'll manually add agents for testing
            )
            
            # Create test LangChain integration
            self.test_langchain_integration = EnhancedLangChainIntegration(
                enable_auto_wrap=False  # Manual wrapping for controlled testing
            )
            
            # Initialize test agents
            self._initialize_test_agents()
            
            self.logger.info("✅ Test environment setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup test environment: {e}")
            raise
    
    def _initialize_test_agents(self):
        """Initialize agents for testing"""
        
        for agent_type, agent_class in agent_test_imports.items():
            if agent_class is None:
                self.logger.debug(f"⚠️ Agent type '{agent_type}' not available for testing")
                continue
            
            try:
                # Create test instance
                agent_id = f"test_{agent_type}_agent"
                agent_instance = agent_class(agent_id=agent_id)
                
                self.test_agents[agent_type] = agent_instance
                
                # Register with orchestrator
                capabilities = self._determine_test_capabilities(agent_type)
                self.test_orchestrator.register_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    instance=agent_instance,
                    capabilities=capabilities
                )
                
                # Wrap for LangChain
                langchain_config = LangChainAgentConfig(
                    agent_id=agent_id,
                    agent_type=agent_type
                )
                self.test_langchain_integration.wrap_agent(agent_instance, langchain_config)
                
                self.logger.info(f"✅ Initialized test agent: {agent_type}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize test agent {agent_type}: {e}")
    
    def _determine_test_capabilities(self, agent_type: str) -> List[str]:
        """Determine capabilities for test agent"""
        capability_map = {
            'memory': ['memory_storage', 'memory_retrieval', 'temporal_modeling'],
            'reasoning': ['logical_reasoning', 'inference', 'problem_solving'],
            'physics': ['physics_validation', 'conservation_laws', 'simulation'],
            'consciousness': ['meta_cognition', 'self_reflection', 'awareness'],
            'web_search': ['web_research', 'information_gathering', 'fact_checking']
        }
        return capability_map.get(agent_type, ['general_processing'])
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        
        self.logger.info("🚀 Starting comprehensive agent test suite")
        start_time = time.time()
        
        # Determine which test categories to run
        categories = self.config.categories or list(TestCategory)
        
        # Run tests by category
        for category in categories:
            await self._run_category_tests(category)
        
        # Calculate final metrics
        total_duration = time.time() - start_time
        self.test_metrics["total_duration"] = total_duration
        
        # Generate report
        report = self._generate_test_report()
        
        self.logger.info(f"✅ Test suite completed in {total_duration:.2f}s")
        self.logger.info(f"📊 Results: {self.test_metrics['passed_tests']}/{self.test_metrics['total_tests']} passed")
        
        return report
    
    async def _run_category_tests(self, category: TestCategory):
        """Run tests for a specific category"""
        
        self.logger.info(f"🧪 Running {category.value} tests")
        
        if category == TestCategory.UNIT:
            await self._run_unit_tests()
        elif category == TestCategory.INTEGRATION:
            await self._run_integration_tests()
        elif category == TestCategory.PERFORMANCE:
            await self._run_performance_tests()
        elif category == TestCategory.LOAD:
            await self._run_load_tests()
        elif category == TestCategory.SCENARIO:
            await self._run_scenario_tests()
        elif category == TestCategory.LANGCHAIN:
            await self._run_langchain_tests()
        elif category == TestCategory.DEEP_AGENT:
            await self._run_deep_agent_tests()
    
    async def _run_unit_tests(self):
        """Run unit tests for individual agents"""
        
        for agent_type, agent_instance in self.test_agents.items():
            
            # Test basic agent initialization
            await self._run_test(
                f"unit_init_{agent_type}",
                TestCategory.UNIT,
                self._test_agent_initialization,
                agent_type, agent_instance
            )
            
            # Test agent capabilities
            await self._run_test(
                f"unit_capabilities_{agent_type}",
                TestCategory.UNIT,
                self._test_agent_capabilities,
                agent_type, agent_instance
            )
            
            # Test agent processing
            await self._run_test(
                f"unit_processing_{agent_type}",
                TestCategory.UNIT,
                self._test_agent_processing,
                agent_type, agent_instance
            )
    
    async def _run_integration_tests(self):
        """Run integration tests between agents"""
        
        # Test orchestrator integration
        await self._run_test(
            "integration_orchestrator",
            TestCategory.INTEGRATION,
            self._test_orchestrator_integration
        )
        
        # Test cross-agent communication
        await self._run_test(
            "integration_cross_agent_communication",
            TestCategory.INTEGRATION,
            self._test_cross_agent_communication
        )
        
        # Test memory sharing
        await self._run_test(
            "integration_memory_sharing",
            TestCategory.INTEGRATION,
            self._test_memory_sharing
        )
    
    async def _run_performance_tests(self):
        """Run performance tests"""
        
        # Test agent response times
        await self._run_test(
            "performance_response_times",
            TestCategory.PERFORMANCE,
            self._test_agent_response_times
        )
        
        # Test throughput
        await self._run_test(
            "performance_throughput",
            TestCategory.PERFORMANCE,
            self._test_agent_throughput
        )
        
        # Test memory usage
        await self._run_test(
            "performance_memory_usage",
            TestCategory.PERFORMANCE,
            self._test_memory_usage
        )
    
    async def _run_load_tests(self):
        """Run load tests"""
        
        if not self.config.include_load_tests:
            self.logger.info("⚠️ Load tests skipped (disabled in config)")
            return
        
        # Test concurrent agent execution
        await self._run_test(
            "load_concurrent_agents",
            TestCategory.LOAD,
            self._test_concurrent_agent_execution
        )
        
        # Test high-volume task processing
        await self._run_test(
            "load_high_volume_tasks",
            TestCategory.LOAD,
            self._test_high_volume_task_processing
        )
    
    async def _run_scenario_tests(self):
        """Run real-world scenario tests"""
        
        # Scenario 1: Research and Analysis Pipeline
        await self._run_test(
            "scenario_research_analysis",
            TestCategory.SCENARIO,
            self._test_research_analysis_scenario
        )
        
        # Scenario 2: Physics Problem Solving
        await self._run_test(
            "scenario_physics_problem",
            TestCategory.SCENARIO,
            self._test_physics_problem_scenario
        )
        
        # Scenario 3: Memory and Reasoning Chain
        await self._run_test(
            "scenario_memory_reasoning",
            TestCategory.SCENARIO,
            self._test_memory_reasoning_scenario
        )
    
    async def _run_langchain_tests(self):
        """Run LangChain integration tests"""
        
        # Test tool creation
        await self._run_test(
            "langchain_tool_creation",
            TestCategory.LANGCHAIN,
            self._test_langchain_tool_creation
        )
        
        # Test workflow execution
        await self._run_test(
            "langchain_workflow_execution",
            TestCategory.LANGCHAIN,
            self._test_langchain_workflow_execution
        )
        
        # Test agent interoperability
        await self._run_test(
            "langchain_agent_interoperability",
            TestCategory.LANGCHAIN,
            self._test_langchain_agent_interoperability
        )
    
    async def _run_deep_agent_tests(self):
        """Run Deep Agent tests"""
        
        # Test plan creation
        await self._run_test(
            "deep_agent_plan_creation",
            TestCategory.DEEP_AGENT,
            self._test_deep_agent_plan_creation
        )
        
        # Test plan execution
        await self._run_test(
            "deep_agent_plan_execution",
            TestCategory.DEEP_AGENT,
            self._test_deep_agent_plan_execution
        )
        
        # Test skill coordination
        await self._run_test(
            "deep_agent_skill_coordination",
            TestCategory.DEEP_AGENT,
            self._test_deep_agent_skill_coordination
        )
    
    async def _run_test(
        self,
        test_name: str,
        category: TestCategory,
        test_function: callable,
        *args,
        **kwargs
    ):
        """Run a single test with error handling and timing"""
        
        start_time = time.time()
        
        try:
            # Run the test with timeout
            result = await asyncio.wait_for(
                test_function(*args, **kwargs),
                timeout=self.config.test_timeout
            )
            
            duration = time.time() - start_time
            
            # Determine status based on result
            if isinstance(result, dict):
                status = result.get("status", "passed")
                details = result
            else:
                status = "passed" if result else "failed"
                details = {"result": result}
            
            test_result = TestResult(
                test_name=test_name,
                category=category,
                status=status,
                duration=duration,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                category=category,
                status="failed",
                duration=duration,
                details={},
                error="Test timeout"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                category=category,
                status="error",
                duration=duration,
                details={},
                error=str(e),
                traceback=traceback.format_exc()
            )
        
        # Store result and update metrics
        self.test_results.append(test_result)
        self._update_test_metrics(test_result)
        
        # Log result
        status_emoji = {
            "passed": "✅",
            "failed": "❌",
            "skipped": "⚠️",
            "error": "💥"
        }
        
        self.logger.info(
            f"{status_emoji.get(test_result.status, '❓')} "
            f"{test_name}: {test_result.status} ({test_result.duration:.2f}s)"
        )
        
        if test_result.error:
            self.logger.debug(f"Error details: {test_result.error}")
    
    def _update_test_metrics(self, test_result: TestResult):
        """Update test metrics"""
        self.test_metrics["total_tests"] += 1
        
        if test_result.status == "passed":
            self.test_metrics["passed_tests"] += 1
        elif test_result.status == "failed":
            self.test_metrics["failed_tests"] += 1
        elif test_result.status == "skipped":
            self.test_metrics["skipped_tests"] += 1
        elif test_result.status == "error":
            self.test_metrics["error_tests"] += 1
    
    # ============================================================================
    # Individual Test Functions
    # ============================================================================
    
    async def _test_agent_initialization(self, agent_type: str, agent_instance: Any) -> Dict[str, Any]:
        """Test agent initialization"""
        
        checks = {
            "has_agent_id": hasattr(agent_instance, 'agent_id'),
            "agent_id_valid": getattr(agent_instance, 'agent_id', None) is not None,
            "has_process_method": hasattr(agent_instance, 'process') or hasattr(agent_instance, 'execute'),
            "initialization_complete": True
        }
        
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "agent_type": agent_type
        }
    
    async def _test_agent_capabilities(self, agent_type: str, agent_instance: Any) -> Dict[str, Any]:
        """Test agent capabilities"""
        
        capabilities = []
        
        # Check for common capabilities
        if hasattr(agent_instance, 'get_capabilities'):
            try:
                capabilities = agent_instance.get_capabilities()
            except Exception:
                pass
        
        checks = {
            "has_capabilities": len(capabilities) > 0,
            "capabilities_list": capabilities,
            "agent_type_specific": agent_type in str(capabilities).lower() if capabilities else False
        }
        
        return {
            "status": "passed" if checks["has_capabilities"] else "skipped",
            "checks": checks,
            "capabilities": capabilities
        }
    
    async def _test_agent_processing(self, agent_type: str, agent_instance: Any) -> Dict[str, Any]:
        """Test basic agent processing"""
        
        test_input = {
            "input": f"Test processing for {agent_type} agent",
            "context": {"test": True}
        }
        
        try:
            if hasattr(agent_instance, 'process'):
                result = await agent_instance.process(test_input)
            elif hasattr(agent_instance, 'execute'):
                result = await agent_instance.execute(test_input)
            else:
                return {
                    "status": "skipped",
                    "reason": "No process or execute method available"
                }
            
            checks = {
                "processing_successful": result is not None,
                "result_type": type(result).__name__,
                "has_output": bool(result)
            }
            
            return {
                "status": "passed" if checks["processing_successful"] else "failed",
                "checks": checks,
                "result_summary": str(result)[:200] if result else None
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"processing_successful": False}
            }
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration"""
        
        if not self.test_orchestrator:
            return {"status": "skipped", "reason": "Orchestrator not available"}
        
        try:
            # Submit a test task
            task_id = await self.test_orchestrator.submit_task(
                task_type="test_task",
                data={"input": "Test orchestrator integration"},
                priority=TaskPriority.NORMAL
            )
            
            # Wait for result
            result = await self.test_orchestrator.get_task_result(task_id, wait=True, timeout=30.0)
            
            checks = {
                "task_submitted": task_id is not None,
                "task_executed": result is not None,
                "result_status": result.status if result else "none",
                "orchestrator_responsive": True
            }
            
            return {
                "status": "passed" if all([checks["task_submitted"], checks["task_executed"]]) else "failed",
                "checks": checks,
                "task_id": task_id
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"orchestrator_responsive": False}
            }
    
    async def _test_cross_agent_communication(self) -> Dict[str, Any]:
        """Test cross-agent communication"""
        
        if len(self.test_agents) < 2:
            return {"status": "skipped", "reason": "Need at least 2 agents for communication test"}
        
        agent_types = list(self.test_agents.keys())[:2]
        
        try:
            # Send message between agents using orchestrator
            success = self.test_langchain_integration.send_cross_agent_message(
                from_agent=f"test_{agent_types[0]}_agent",
                to_agent=f"test_{agent_types[1]}_agent",
                message={"test": "cross-agent communication", "timestamp": time.time()}
            )
            
            # Check if message was received
            messages = self.test_langchain_integration.get_agent_messages(f"test_{agent_types[1]}_agent")
            
            checks = {
                "message_sent": success,
                "message_received": len(messages) > 0,
                "message_content_valid": any("cross-agent communication" in str(msg) for msg in messages)
            }
            
            return {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "message_count": len(messages)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"communication_error": True}
            }
    
    async def _test_memory_sharing(self) -> Dict[str, Any]:
        """Test memory sharing between agents"""
        
        # This is a placeholder test - would need actual memory sharing implementation
        return {
            "status": "skipped",
            "reason": "Memory sharing test not implemented yet",
            "checks": {"memory_sharing_available": False}
        }
    
    async def _test_agent_response_times(self) -> Dict[str, Any]:
        """Test agent response times"""
        
        response_times = {}
        
        for agent_type, agent_instance in self.test_agents.items():
            try:
                start_time = time.time()
                
                # Simple processing test
                test_input = {"input": "Quick response test"}
                
                if hasattr(agent_instance, 'process'):
                    await agent_instance.process(test_input)
                elif hasattr(agent_instance, 'execute'):
                    await agent_instance.execute(test_input)
                
                response_time = time.time() - start_time
                response_times[agent_type] = response_time
                
            except Exception as e:
                response_times[agent_type] = None
        
        avg_response_time = sum(t for t in response_times.values() if t is not None) / len([t for t in response_times.values() if t is not None])
        
        checks = {
            "all_agents_responsive": all(t is not None for t in response_times.values()),
            "average_response_time": avg_response_time,
            "response_times_acceptable": avg_response_time < 5.0  # 5 second threshold
        }
        
        return {
            "status": "passed" if checks["response_times_acceptable"] else "failed",
            "checks": checks,
            "response_times": response_times
        }
    
    async def _test_agent_throughput(self) -> Dict[str, Any]:
        """Test agent throughput"""
        
        # Simple throughput test - process multiple tasks quickly
        num_tasks = 10
        
        try:
            if not self.test_orchestrator:
                return {"status": "skipped", "reason": "Orchestrator not available"}
            
            start_time = time.time()
            
            # Submit multiple tasks
            task_ids = []
            for i in range(num_tasks):
                task_id = await self.test_orchestrator.submit_task(
                    task_type="throughput_test",
                    data={"input": f"Throughput test task {i}"},
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            # Wait for all results
            completed_tasks = 0
            for task_id in task_ids:
                result = await self.test_orchestrator.get_task_result(task_id, wait=True, timeout=10.0)
                if result and result.status == "completed":
                    completed_tasks += 1
            
            total_time = time.time() - start_time
            throughput = completed_tasks / total_time if total_time > 0 else 0
            
            checks = {
                "tasks_submitted": len(task_ids),
                "tasks_completed": completed_tasks,
                "completion_rate": completed_tasks / num_tasks,
                "throughput_tasks_per_second": throughput,
                "throughput_acceptable": throughput > 1.0  # At least 1 task per second
            }
            
            return {
                "status": "passed" if checks["throughput_acceptable"] else "failed",
                "checks": checks
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"throughput_test_error": True}
            }
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        
        # Basic memory usage test
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform some operations with agents
        for agent_type, agent_instance in self.test_agents.items():
            try:
                test_input = {"input": "Memory usage test"}
                if hasattr(agent_instance, 'process'):
                    await agent_instance.process(test_input)
            except Exception:
                pass
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        checks = {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_increase_mb": memory_increase,
            "memory_increase_acceptable": memory_increase < 100  # Less than 100MB increase
        }
        
        return {
            "status": "passed" if checks["memory_increase_acceptable"] else "failed",
            "checks": checks
        }
    
    async def _test_concurrent_agent_execution(self) -> Dict[str, Any]:
        """Test concurrent agent execution"""
        
        if len(self.test_agents) < 2:
            return {"status": "skipped", "reason": "Need at least 2 agents for concurrent test"}
        
        try:
            # Run multiple agents concurrently
            tasks = []
            for agent_type, agent_instance in self.test_agents.items():
                if hasattr(agent_instance, 'process'):
                    task = agent_instance.process({"input": f"Concurrent test for {agent_type}"})
                    tasks.append(task)
            
            # Wait for all to complete
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            successful_results = sum(1 for r in results if not isinstance(r, Exception))
            
            checks = {
                "concurrent_agents": len(tasks),
                "successful_executions": successful_results,
                "execution_time": execution_time,
                "concurrency_successful": successful_results == len(tasks),
                "performance_acceptable": execution_time < 30.0  # 30 second threshold
            }
            
            return {
                "status": "passed" if checks["concurrency_successful"] and checks["performance_acceptable"] else "failed",
                "checks": checks
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"concurrent_execution_error": True}
            }
    
    async def _test_high_volume_task_processing(self) -> Dict[str, Any]:
        """Test high volume task processing"""
        
        # This would be a more intensive test - placeholder for now
        return {
            "status": "skipped",
            "reason": "High volume test skipped (would be resource intensive)",
            "checks": {"high_volume_capable": "unknown"}
        }
    
    async def _test_research_analysis_scenario(self) -> Dict[str, Any]:
        """Test research and analysis scenario"""
        
        # Simulate a research pipeline: web search → reasoning → memory storage
        
        if 'web_search' not in self.test_agents or 'reasoning' not in self.test_agents:
            return {"status": "skipped", "reason": "Required agents not available"}
        
        try:
            # Step 1: Web search
            search_result = None
            if hasattr(self.test_agents['web_search'], 'process'):
                search_result = await self.test_agents['web_search'].process({
                    "query": "artificial intelligence trends 2024"
                })
            
            # Step 2: Reasoning analysis
            reasoning_result = None
            if search_result and hasattr(self.test_agents['reasoning'], 'process'):
                reasoning_result = await self.test_agents['reasoning'].process({
                    "input": str(search_result),
                    "task": "analyze trends"
                })
            
            checks = {
                "search_completed": search_result is not None,
                "reasoning_completed": reasoning_result is not None,
                "pipeline_successful": search_result is not None and reasoning_result is not None
            }
            
            return {
                "status": "passed" if checks["pipeline_successful"] else "failed",
                "checks": checks,
                "scenario": "research_analysis"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"scenario_error": True}
            }
    
    async def _test_physics_problem_scenario(self) -> Dict[str, Any]:
        """Test physics problem solving scenario"""
        
        if 'physics' not in self.test_agents:
            return {"status": "skipped", "reason": "Physics agent not available"}
        
        try:
            physics_agent = self.test_agents['physics']
            
            # Test physics problem
            if hasattr(physics_agent, 'process'):
                result = await physics_agent.process({
                    "problem": "Calculate the trajectory of a projectile launched at 45 degrees with initial velocity 20 m/s",
                    "domain": "mechanics"
                })
            
            checks = {
                "physics_processing": result is not None,
                "result_contains_physics": "trajectory" in str(result).lower() if result else False
            }
            
            return {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "scenario": "physics_problem"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"scenario_error": True}
            }
    
    async def _test_memory_reasoning_scenario(self) -> Dict[str, Any]:
        """Test memory and reasoning chain scenario"""
        
        if 'memory' not in self.test_agents or 'reasoning' not in self.test_agents:
            return {"status": "skipped", "reason": "Required agents not available"}
        
        try:
            memory_agent = self.test_agents['memory']
            reasoning_agent = self.test_agents['reasoning']
            
            # Store some information
            store_result = None
            if hasattr(memory_agent, 'process'):
                store_result = await memory_agent.process({
                    "action": "store",
                    "content": "The capital of France is Paris",
                    "type": "fact"
                })
            
            # Retrieve and reason about it
            retrieve_result = None
            if hasattr(memory_agent, 'process'):
                retrieve_result = await memory_agent.process({
                    "action": "retrieve",
                    "query": "capital of France"
                })
            
            # Use reasoning on retrieved information
            reasoning_result = None
            if retrieve_result and hasattr(reasoning_agent, 'process'):
                reasoning_result = await reasoning_agent.process({
                    "input": str(retrieve_result),
                    "task": "explain geographical significance"
                })
            
            checks = {
                "memory_store": store_result is not None,
                "memory_retrieve": retrieve_result is not None,
                "reasoning_applied": reasoning_result is not None,
                "chain_successful": all([store_result, retrieve_result, reasoning_result])
            }
            
            return {
                "status": "passed" if checks["chain_successful"] else "failed",
                "checks": checks,
                "scenario": "memory_reasoning_chain"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"scenario_error": True}
            }
    
    async def _test_langchain_tool_creation(self) -> Dict[str, Any]:
        """Test LangChain tool creation"""
        
        if not self.test_langchain_integration:
            return {"status": "skipped", "reason": "LangChain integration not available"}
        
        try:
            # Get all tools
            all_tools = self.test_langchain_integration.get_all_tools()
            
            # Check tool creation
            tools_by_agent = {}
            for agent_id, wrapper in self.test_langchain_integration.wrapped_agents.items():
                tools_by_agent[agent_id] = len(wrapper.get_tools())
            
            checks = {
                "tools_created": len(all_tools) > 0,
                "tools_per_agent": tools_by_agent,
                "total_tools": len(all_tools),
                "tool_creation_successful": len(all_tools) >= len(self.test_agents)
            }
            
            return {
                "status": "passed" if checks["tool_creation_successful"] else "failed",
                "checks": checks
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"tool_creation_error": True}
            }
    
    async def _test_langchain_workflow_execution(self) -> Dict[str, Any]:
        """Test LangChain workflow execution"""
        
        if not self.test_langchain_integration:
            return {"status": "skipped", "reason": "LangChain integration not available"}
        
        try:
            # Create a simple workflow
            agent_ids = list(self.test_langchain_integration.wrapped_agents.keys())[:2]
            
            if len(agent_ids) < 2:
                return {"status": "skipped", "reason": "Need at least 2 wrapped agents"}
            
            workflow = self.test_langchain_integration.create_multi_agent_workflow(
                "test_workflow",
                agent_ids
            )
            
            if not workflow:
                return {"status": "failed", "checks": {"workflow_creation": False}}
            
            # Execute workflow
            result = await self.test_langchain_integration.execute_workflow(
                "test_workflow",
                "Test workflow execution",
                max_iterations=3
            )
            
            checks = {
                "workflow_created": workflow is not None,
                "workflow_executed": result is not None,
                "execution_successful": result.get("status") == "completed" if result else False
            }
            
            return {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "result_summary": result
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"workflow_execution_error": True}
            }
    
    async def _test_langchain_agent_interoperability(self) -> Dict[str, Any]:
        """Test LangChain agent interoperability"""
        
        # Test cross-agent tool usage
        if not self.test_langchain_integration:
            return {"status": "skipped", "reason": "LangChain integration not available"}
        
        try:
            status = self.test_langchain_integration.get_integration_status()
            
            checks = {
                "langchain_core_available": status.get("langchain_core_available", False),
                "langgraph_available": status.get("langgraph_available", False),
                "wrapped_agents_count": len(status.get("wrapped_agents", [])),
                "total_tools": status.get("total_tools", 0),
                "interoperability_ready": status.get("total_tools", 0) > 0
            }
            
            return {
                "status": "passed" if checks["interoperability_ready"] else "failed",
                "checks": checks,
                "integration_status": status
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"interoperability_error": True}
            }
    
    async def _test_deep_agent_plan_creation(self) -> Dict[str, Any]:
        """Test Deep Agent plan creation"""
        
        if not self.test_orchestrator or not self.test_orchestrator.deep_planner:
            return {"status": "skipped", "reason": "Deep Agent planner not available"}
        
        try:
            planner = self.test_orchestrator.deep_planner
            
            # Create a test plan
            plan = await planner.create_plan(
                goal="Test deep agent planning",
                context={"test": True, "agents_available": list(self.test_agents.keys())}
            )
            
            checks = {
                "plan_created": plan is not None,
                "plan_has_steps": len(plan.steps) > 0 if plan else False,
                "plan_id_valid": plan.id is not None if plan else False
            }
            
            return {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "plan_steps": len(plan.steps) if plan else 0
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"plan_creation_error": True}
            }
    
    async def _test_deep_agent_plan_execution(self) -> Dict[str, Any]:
        """Test Deep Agent plan execution"""
        
        if not self.test_orchestrator or not self.test_orchestrator.deep_planner:
            return {"status": "skipped", "reason": "Deep Agent planner not available"}
        
        try:
            planner = self.test_orchestrator.deep_planner
            
            # Create and execute a simple plan
            plan = await planner.create_plan(
                goal="Execute simple test plan",
                context={"test": True}
            )
            
            if not plan:
                return {"status": "failed", "checks": {"plan_creation": False}}
            
            # Execute the plan
            execution_result = await planner.execute_plan(plan.id)
            
            checks = {
                "plan_created": plan is not None,
                "plan_executed": execution_result is not None,
                "execution_status": execution_result.get("status") if execution_result else "unknown",
                "execution_successful": execution_result.get("status") in ["completed", "partial_success"] if execution_result else False
            }
            
            return {
                "status": "passed" if checks["execution_successful"] else "failed",
                "checks": checks,
                "execution_result": execution_result
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"plan_execution_error": True}
            }
    
    async def _test_deep_agent_skill_coordination(self) -> Dict[str, Any]:
        """Test Deep Agent skill coordination"""
        
        if not self.test_orchestrator or not self.test_orchestrator.deep_planner:
            return {"status": "skipped", "reason": "Deep Agent planner not available"}
        
        try:
            planner = self.test_orchestrator.deep_planner
            skills = planner.skills
            
            checks = {
                "skills_registered": len(skills) > 0,
                "skill_names": list(skills.keys()),
                "skills_functional": all(hasattr(skill, 'execute') for skill in skills.values())
            }
            
            return {
                "status": "passed" if checks["skills_registered"] and checks["skills_functional"] else "failed",
                "checks": checks,
                "skill_count": len(skills)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "checks": {"skill_coordination_error": True}
            }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Organize results by category
        results_by_category = {}
        for result in self.test_results:
            category = result.category.value
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)
        
        # Calculate success rates by category
        category_stats = {}
        for category, results in results_by_category.items():
            total = len(results)
            passed = len([r for r in results if r.status == "passed"])
            failed = len([r for r in results if r.status == "failed"])
            skipped = len([r for r in results if r.status == "skipped"])
            errors = len([r for r in results if r.status == "error"])
            
            category_stats[category] = {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "success_rate": passed / total if total > 0 else 0,
                "avg_duration": sum(r.duration for r in results) / total if total > 0 else 0
            }
        
        # Failed test details
        failed_tests = [r for r in self.test_results if r.status in ["failed", "error"]]
        
        # Performance summary
        performance_summary = {
            "fastest_test": min(self.test_results, key=lambda x: x.duration) if self.test_results else None,
            "slowest_test": max(self.test_results, key=lambda x: x.duration) if self.test_results else None,
            "average_test_duration": sum(r.duration for r in self.test_results) / len(self.test_results) if self.test_results else 0
        }
        
        # Overall assessment
        overall_success_rate = self.test_metrics["passed_tests"] / max(self.test_metrics["total_tests"], 1)
        
        if overall_success_rate >= 0.95:
            assessment = "EXCELLENT"
        elif overall_success_rate >= 0.85:
            assessment = "GOOD"
        elif overall_success_rate >= 0.70:
            assessment = "ACCEPTABLE"
        elif overall_success_rate >= 0.50:
            assessment = "NEEDS_IMPROVEMENT"
        else:
            assessment = "CRITICAL_ISSUES"
        
        return {
            "test_suite_summary": {
                "total_tests": self.test_metrics["total_tests"],
                "passed": self.test_metrics["passed_tests"],
                "failed": self.test_metrics["failed_tests"],
                "skipped": self.test_metrics["skipped_tests"],
                "errors": self.test_metrics["error_tests"],
                "success_rate": overall_success_rate,
                "total_duration": self.test_metrics["total_duration"],
                "assessment": assessment
            },
            "category_breakdown": category_stats,
            "performance_summary": performance_summary,
            "failed_tests": [
                {
                    "name": t.test_name,
                    "category": t.category.value,
                    "error": t.error,
                    "duration": t.duration
                }
                for t in failed_tests
            ],
            "agent_coverage": {
                "total_agents_available": len(agent_test_imports),
                "agents_tested": len(self.test_agents),
                "agents_in_orchestrator": len(self.test_orchestrator.agents) if self.test_orchestrator else 0,
                "agents_in_langchain": len(self.test_langchain_integration.wrapped_agents) if self.test_langchain_integration else 0
            },
            "integration_status": {
                "orchestrator_ready": self.test_orchestrator is not None,
                "langchain_ready": self.test_langchain_integration is not None,
                "deep_agent_ready": self.test_orchestrator.enable_deep_agent if self.test_orchestrator else False
            },
            "timestamp": time.time(),
            "config_used": {
                "categories": [c.value for c in self.config.categories] if self.config.categories else "all",
                "include_performance": self.config.include_performance,
                "include_load_tests": self.config.include_load_tests,
                "test_timeout": self.config.test_timeout
            }
        }


# Convenience functions for running tests
async def run_full_test_suite(config: TestSuiteConfig = None) -> Dict[str, Any]:
    """Run the full test suite"""
    framework = AgentTestFramework(config)
    return await framework.run_all_tests()

async def run_quick_test(agent_types: List[str] = None) -> Dict[str, Any]:
    """Run a quick test of basic functionality"""
    config = TestSuiteConfig(
        categories=[TestCategory.UNIT, TestCategory.INTEGRATION],
        agents_to_test=agent_types,
        include_performance=False,
        include_load_tests=False,
        test_timeout=60.0
    )
    return await run_full_test_suite(config)

async def run_performance_test() -> Dict[str, Any]:
    """Run performance-focused tests"""
    config = TestSuiteConfig(
        categories=[TestCategory.PERFORMANCE],
        include_performance=True,
        include_load_tests=False
    )
    return await run_full_test_suite(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        print("🧪 Running NIS Protocol Agent Test Suite")
        
        # Run quick test
        print("\n🚀 Running quick test...")
        quick_result = await run_quick_test()
        
        print(f"\n📊 Quick Test Results:")
        print(f"Success Rate: {quick_result['test_suite_summary']['success_rate']:.1%}")
        print(f"Assessment: {quick_result['test_suite_summary']['assessment']}")
        
        # Run performance test if quick test passes
        if quick_result['test_suite_summary']['success_rate'] > 0.8:
            print("\n⚡ Running performance test...")
            perf_result = await run_performance_test()
            print(f"Performance Test Success Rate: {perf_result['test_suite_summary']['success_rate']:.1%}")
        
        print("\n✅ Test suite execution complete!")
    
    asyncio.run(main())
