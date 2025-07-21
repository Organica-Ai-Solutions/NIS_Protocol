"""
Consciousness Agent Benchmarks

Production-grade benchmarks for validating consciousness agent performance
claims and ensuring system reliability under various operational conditions.

Benchmark Categories:
1. Load Testing - High volume operation validation
2. Stress Testing - Performance under resource constraints  
3. Accuracy Testing - Decision quality validation
4. Scalability Testing - Multi-agent coordination performance
5. Memory Testing - Long-term operation stability
"""

import time
import json
import psutil
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

# Import consciousness agents
from src.agents.consciousness.conscious_agent import ConsciousAgent
from src.agents.consciousness.introspection_manager import IntrospectionManager
from src.agents.consciousness.meta_cognitive_processor import MetaCognitiveProcessor


class ConsciousnessBenchmarkSuite:
    """Comprehensive benchmark suite for consciousness agents."""
    
    def __init__(self):
        self.results = {}
        self.start_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def run_load_test(self, agent_type: str, operations_per_second: int, duration_seconds: int):
        """Run load test for specified agent type."""
        print(f"Running load test: {agent_type} - {operations_per_second} ops/sec for {duration_seconds}s")
        
        # Initialize agent
        if agent_type == "conscious_agent":
            agent = ConsciousAgent()
            test_message = {
                "operation": "introspect",
                "agent_data": {
                    "performance_metrics": {"success_rate": 0.9, "response_time": 0.1},
                    "goals": [{"id": "test_goal", "target_metrics": {"accuracy": 0.9}}]
                }
            }
            operation = lambda: agent.process(test_message)
        elif agent_type == "introspection_manager":
            agent = IntrospectionManager()
            operation = lambda: agent.analyze_system_performance()
        elif agent_type == "meta_processor":
            agent = MetaCognitiveProcessor()
            operation = lambda: agent.get_meta_insights()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Load test execution
        start_time = time.time()
        end_time = start_time + duration_seconds
        completed_operations = 0
        failed_operations = 0
        response_times = []
        memory_samples = []
        cpu_samples = []
        
        interval = 1.0 / operations_per_second
        
        while time.time() < end_time:
            operation_start = time.time()
            
            try:
                result = operation()
                operation_end = time.time()
                response_time = operation_end - operation_start
                response_times.append(response_time)
                completed_operations += 1
            except Exception as e:
                failed_operations += 1
                print(f"Operation failed: {e}")
            
            # Sample system metrics every 10 operations
            if completed_operations % 10 == 0:
                memory_samples.append(self._get_memory_usage())
                cpu_samples.append(self._get_cpu_usage())
            
            # Rate limiting
            elapsed = time.time() - operation_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        # Calculate results
        actual_duration = time.time() - start_time
        actual_ops_per_second = completed_operations / actual_duration
        success_rate = completed_operations / (completed_operations + failed_operations)
        
        results = {
            "agent_type": agent_type,
            "target_ops_per_second": operations_per_second,
            "actual_ops_per_second": actual_ops_per_second,
            "duration": actual_duration,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "success_rate": success_rate,
            "mean_response_time": np.mean(response_times) if response_times else 0,
            "p95_response_time": np.percentile(response_times, 95) if response_times else 0,
            "p99_response_time": np.percentile(response_times, 99) if response_times else 0,
            "peak_memory_mb": max(memory_samples) if memory_samples else 0,
            "mean_cpu_percent": np.mean(cpu_samples) if cpu_samples else 0,
            "performance_target_met": actual_ops_per_second >= operations_per_second * 0.9
        }
        
        self.results[f"load_test_{agent_type}"] = results
        return results
    
    def run_stress_test(self, agent_type: str, max_concurrent: int, total_operations: int):
        """Run stress test with high concurrency."""
        print(f"Running stress test: {agent_type} - {max_concurrent} concurrent, {total_operations} total ops")
        
        # Initialize agent
        if agent_type == "conscious_agent":
            agent = ConsciousAgent()
            test_message = {
                "operation": "evaluate_decision",
                "decision_data": {
                    "reasoning_steps": [{"claims": ["test"], "conclusions": ["result"]}],
                    "emotional_factors": {"arousal": 0.5, "valence": 0.0}
                }
            }
            operation = lambda: agent.process(test_message)
        elif agent_type == "introspection_manager":
            agent = IntrospectionManager()
            # Setup test introspections
            from src.agents.consciousness.introspection_manager import AgentIntrospection, PerformanceStatus
            test_introspections = {}
            for i in range(10):
                agent_id = f"stress_test_agent_{i}"
                introspection = AgentIntrospection(
                    agent_id=agent_id,
                    status=PerformanceStatus.GOOD,
                    confidence=0.85,
                    performance_metrics={"success_rate": 0.9, "response_time": 0.1},
                    behavioral_patterns={"consistency": 0.8},
                    cultural_neutrality_score=0.9,
                    mathematical_validation={},
                    last_evaluation=time.time()
                )
                test_introspections[agent_id] = introspection
            agent.agent_introspections = test_introspections
            operation = lambda: agent.get_agent_status_summary()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Stress test execution
        start_time = time.time()
        completed_operations = 0
        failed_operations = 0
        response_times = []
        
        def execute_operation():
            try:
                op_start = time.time()
                result = operation()
                op_end = time.time()
                return op_end - op_start, None
            except Exception as e:
                return None, str(e)
        
        # Execute operations with controlled concurrency
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(execute_operation) for _ in range(total_operations)]
            
            for future in as_completed(futures):
                response_time, error = future.result()
                if error:
                    failed_operations += 1
                else:
                    completed_operations += 1
                    response_times.append(response_time)
        
        # Calculate results
        total_duration = time.time() - start_time
        success_rate = completed_operations / total_operations
        throughput = completed_operations / total_duration
        
        results = {
            "agent_type": agent_type,
            "max_concurrent": max_concurrent,
            "total_operations": total_operations,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "success_rate": success_rate,
            "throughput_ops_per_sec": throughput,
            "total_duration": total_duration,
            "mean_response_time": np.mean(response_times) if response_times else 0,
            "p95_response_time": np.percentile(response_times, 95) if response_times else 0,
            "p99_response_time": np.percentile(response_times, 99) if response_times else 0,
            "stress_test_passed": success_rate >= 0.95 and throughput >= 10
        }
        
        self.results[f"stress_test_{agent_type}"] = results
        return results
    
    def run_accuracy_benchmark(self):
        """Run accuracy benchmark for decision quality analysis."""
        print("Running accuracy benchmark for decision quality analysis")
        
        agent = ConsciousAgent()
        
        # Test cases with known expected quality ranges
        test_cases = [
            {
                "name": "high_quality_decision",
                "decision_data": {
                    "reasoning_steps": [
                        {"claims": ["valid_premise"], "conclusions": ["logical_step_1"]},
                        {"premises": ["logical_step_1"], "conclusions": ["sound_conclusion"]}
                    ],
                    "premises": [{"claims": ["valid_premise"]}],
                    "conclusion": {"claims": ["sound_conclusion"]},
                    "emotional_factors": {"arousal": 0.3, "valence": 0.6, "regulation_quality": 0.9},
                    "context": {"type": "routine"},
                    "current_goals": [{"success_metrics": {"accuracy": 0.9}, "priority": 0.8}],
                    "expected_outcomes": [{"metrics": {"accuracy": 0.95}}],
                    "risk_factors": [{"severity": 0.2, "probability": 0.1}]
                },
                "expected_quality": (0.8, 1.0)
            },
            {
                "name": "medium_quality_decision",
                "decision_data": {
                    "reasoning_steps": [{"claims": ["premise"], "conclusions": ["conclusion"]}],
                    "emotional_factors": {"arousal": 0.5, "valence": 0.0},
                    "context": {"type": "routine"},
                    "current_goals": [{"success_metrics": {"accuracy": 0.8}}],
                    "risk_factors": [{"severity": 0.4, "probability": 0.3}]
                },
                "expected_quality": (0.5, 0.8)
            },
            {
                "name": "low_quality_decision",
                "decision_data": {
                    "reasoning_steps": [],
                    "emotional_factors": {"arousal": 0.9, "valence": -0.8},
                    "context": {"type": "crisis"},
                    "current_goals": [],
                    "risk_factors": [{"severity": 0.9, "probability": 0.8}]
                },
                "expected_quality": (0.0, 0.5)
            }
        ]
        
        correct_assessments = 0
        total_assessments = len(test_cases)
        assessment_times = []
        
        for test_case in test_cases:
            message = {
                "operation": "evaluate_decision",
                "decision_data": test_case["decision_data"]
            }
            
            start_time = time.time()
            result = agent.process(message)
            assessment_time = time.time() - start_time
            assessment_times.append(assessment_time)
            
            # Extract quality score
            decision_result = result.get("result", {}).get("decision_result", {})
            overall_quality = decision_result.get("overall_quality", 0.0)
            
            # Check if assessment is within expected range
            expected_min, expected_max = test_case["expected_quality"]
            if expected_min <= overall_quality <= expected_max:
                correct_assessments += 1
                status = "CORRECT"
            else:
                status = "INCORRECT"
            
            print(f"  {test_case['name']}: quality={overall_quality:.3f}, expected={test_case['expected_quality']}, {status}")
        
        accuracy = correct_assessments / total_assessments
        mean_assessment_time = np.mean(assessment_times)
        
        results = {
            "total_test_cases": total_assessments,
            "correct_assessments": correct_assessments,
            "accuracy": accuracy,
            "mean_assessment_time": mean_assessment_time,
            "accuracy_target_met": accuracy >= 0.8,
            "performance_target_met": mean_assessment_time <= 0.2
        }
        
        self.results["accuracy_benchmark"] = results
        return results
    
    def run_memory_leak_test(self, duration_minutes: int = 10):
        """Run long-term memory usage test to detect leaks."""
        print(f"Running memory leak test for {duration_minutes} minutes")
        
        # Initialize agents
        conscious_agent = ConsciousAgent()
        introspection_manager = IntrospectionManager()
        meta_processor = MetaCognitiveProcessor()
        
        # Test messages
        test_messages = [
            {"operation": "introspect", "agent_data": {"performance_metrics": {"success_rate": 0.9}}},
            {"operation": "evaluate_decision", "decision_data": {"reasoning_steps": [], "emotional_factors": {}}}
        ]
        
        # Memory tracking
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        memory_samples = []
        operation_count = 0
        
        while time.time() < end_time:
            # Perform operations
            for message in test_messages:
                conscious_agent.process(message)
                operation_count += 1
            
            introspection_manager.analyze_system_performance()
            meta_processor.get_meta_insights()
            operation_count += 2
            
            # Sample memory every 30 seconds
            if operation_count % 100 == 0:
                current_memory = self._get_memory_usage()
                memory_samples.append({
                    "time": time.time() - start_time,
                    "memory_mb": current_memory,
                    "operations": operation_count
                })
                print(f"  Memory: {current_memory:.1f}MB, Operations: {operation_count}")
            
            time.sleep(0.1)  # Brief pause between operations
        
        # Analyze memory usage trend
        if len(memory_samples) >= 3:
            times = [s["time"] for s in memory_samples]
            memories = [s["memory_mb"] for s in memory_samples]
            
            # Calculate memory growth rate
            memory_slope = np.polyfit(times, memories, 1)[0]  # MB per second
            memory_growth_per_hour = memory_slope * 3600
            
            # Memory leak detection
            memory_leak_detected = memory_growth_per_hour > 10  # More than 10MB/hour growth
            peak_memory = max(memories)
            memory_efficiency = peak_memory < 200  # Less than 200MB peak usage
        else:
            memory_leak_detected = False
            peak_memory = 0
            memory_efficiency = True
            memory_growth_per_hour = 0
        
        results = {
            "test_duration_minutes": duration_minutes,
            "total_operations": operation_count,
            "memory_samples": len(memory_samples),
            "peak_memory_mb": peak_memory,
            "memory_growth_per_hour_mb": memory_growth_per_hour,
            "memory_leak_detected": memory_leak_detected,
            "memory_efficiency_target_met": memory_efficiency,
            "memory_timeline": memory_samples
        }
        
        self.results["memory_leak_test"] = results
        return results
    
    def run_full_benchmark_suite(self):
        """Run complete benchmark suite and generate report."""
        print("=== NIS Protocol Consciousness Agents Benchmark Suite ===")
        print(f"Start time: {time.ctime()}")
        print(f"Initial memory usage: {self.start_memory:.1f}MB")
        print()
        
        # Load tests
        print("1. LOAD TESTING")
        self.run_load_test("conscious_agent", 50, 30)
        self.run_load_test("introspection_manager", 20, 30)
        self.run_load_test("meta_processor", 30, 30)
        print()
        
        # Stress tests
        print("2. STRESS TESTING")
        self.run_stress_test("conscious_agent", 10, 100)
        self.run_stress_test("introspection_manager", 5, 50)
        print()
        
        # Accuracy testing
        print("3. ACCURACY TESTING")
        self.run_accuracy_benchmark()
        print()
        
        # Memory testing
        print("4. MEMORY LEAK TESTING")
        self.run_memory_leak_test(5)  # 5 minute test for demo
        print()
        
        # Generate comprehensive report
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("=== BENCHMARK RESULTS SUMMARY ===")
        
        # Overall status
        all_targets_met = True
        failed_tests = []
        
        for test_name, results in self.results.items():
            target_met = False
            
            if "load_test" in test_name:
                target_met = results.get("performance_target_met", False)
                status = "✅ PASS" if target_met else "❌ FAIL"
                print(f"{test_name}: {status} - {results['actual_ops_per_second']:.1f} ops/sec (target: {results['target_ops_per_second']})")
                
            elif "stress_test" in test_name:
                target_met = results.get("stress_test_passed", False)
                status = "✅ PASS" if target_met else "❌ FAIL"
                print(f"{test_name}: {status} - {results['success_rate']:.3f} success rate, {results['throughput_ops_per_sec']:.1f} ops/sec")
                
            elif "accuracy_benchmark" in test_name:
                target_met = results.get("accuracy_target_met", False) and results.get("performance_target_met", False)
                status = "✅ PASS" if target_met else "❌ FAIL"
                print(f"{test_name}: {status} - {results['accuracy']:.3f} accuracy, {results['mean_assessment_time']:.3f}s avg time")
                
            elif "memory_leak_test" in test_name:
                target_met = not results.get("memory_leak_detected", True) and results.get("memory_efficiency_target_met", False)
                status = "✅ PASS" if target_met else "❌ FAIL"
                print(f"{test_name}: {status} - {results['peak_memory_mb']:.1f}MB peak, {results['memory_growth_per_hour_mb']:.2f}MB/hr growth")
            
            if not target_met:
                all_targets_met = False
                failed_tests.append(test_name)
        
        print()
        print("=== OVERALL ASSESSMENT ===")
        overall_status = "✅ ALL TARGETS MET" if all_targets_met else f"❌ {len(failed_tests)} TESTS FAILED"
        print(f"Status: {overall_status}")
        
        if failed_tests:
            print("Failed tests:")
            for test in failed_tests:
                print(f"  - {test}")
        
        # Performance summary
        print(f"Final memory usage: {self._get_memory_usage():.1f}MB")
        print(f"Memory increase: {self._get_memory_usage() - self.start_memory:.1f}MB")
        
        # Save detailed results
        report = {
            "benchmark_timestamp": time.time(),
            "overall_status": "PASS" if all_targets_met else "FAIL",
            "tests_passed": len(self.results) - len(failed_tests),
            "tests_failed": len(failed_tests),
            "failed_tests": failed_tests,
            "detailed_results": self.results,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
            }
        }
        
        with open("benchmarks/consciousness_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results saved to: benchmarks/consciousness_benchmark_report.json")
        return report


if __name__ == "__main__":
    # Run full benchmark suite
    benchmark = ConsciousnessBenchmarkSuite()
    report = benchmark.run_full_benchmark_suite()
    
    print(f"\nBenchmark complete. Overall status: {report['overall_status']}")
    print(f"Results: {report['tests_passed']} passed, {report['tests_failed']} failed") 