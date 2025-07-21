"""
Performance Validation Tests for Consciousness Agents

This module validates all performance claims made by consciousness agents
with actual benchmark measurements and statistical validation.

Validates:
- Response time claims (< 200ms for standard operations)
- Accuracy claims (> 85% for decision quality assessments)
- Memory efficiency claims (< 100MB per agent instance)
- Throughput claims (> 50 operations per second)
- Error analysis effectiveness claims
- Pattern learning efficiency claims
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock

# Import agents to test
from ..conscious_agent import ConsciousAgent
from ..introspection_manager import IntrospectionManager
from ..meta_cognitive_processor import MetaCognitiveProcessor


class TestPerformanceClaims:
    """Validate specific performance claims made in the consciousness agents."""
    
    def test_conscious_agent_response_time_claim(self):
        """Validate ConsciousAgent claim: response time < 200ms for standard operations."""
        agent = ConsciousAgent()
        
        test_message = {
            "operation": "introspect",
            "agent_data": {
                "performance_metrics": {"success_rate": 0.9, "response_time": 0.1},
                "goals": [{"id": "test_goal", "target_metrics": {"accuracy": 0.9}}]
            }
        }
        
        # Measure response time over multiple iterations
        response_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            result = agent.process(test_message)
            end_time = time.perf_counter()
            response_times.append(end_time - start_time)
        
        mean_response_time = np.mean(response_times)
        percentile_95 = np.percentile(response_times, 95)
        
        # Validate claim
        assert mean_response_time < 0.2, f"Mean response time {mean_response_time:.3f}s exceeds 200ms claim"
        assert percentile_95 < 0.5, f"95th percentile {percentile_95:.3f}s exceeds reasonable limit"
    
    def test_decision_quality_accuracy_claim(self):
        """Validate ConsciousAgent claim: > 85% accuracy for decision quality assessments."""
        agent = ConsciousAgent()
        
        # Test cases with known quality levels
        test_cases = [
            {
                "decision_data": {
                    "reasoning_steps": [{"claims": ["valid"], "conclusions": ["sound"]}],
                    "emotional_factors": {"arousal": 0.3, "valence": 0.6},
                    "current_goals": [{"success_metrics": {"accuracy": 0.9}}]
                },
                "expected_high_quality": True
            },
            {
                "decision_data": {
                    "reasoning_steps": [],
                    "emotional_factors": {"arousal": 0.9, "valence": -0.8},
                    "current_goals": []
                },
                "expected_high_quality": False
            }
        ]
        
        correct_assessments = 0
        for test_case in test_cases:
            message = {"operation": "evaluate_decision", "decision_data": test_case["decision_data"]}
            result = agent.process(message)
            
            decision_result = result.get("result", {}).get("decision_result", {})
            quality = decision_result.get("overall_quality", 0.0)
            
            # Check if assessment matches expectation
            if test_case["expected_high_quality"] and quality > 0.7:
                correct_assessments += 1
            elif not test_case["expected_high_quality"] and quality <= 0.5:
                correct_assessments += 1
        
        accuracy = correct_assessments / len(test_cases)
        assert accuracy >= 0.85, f"Decision quality accuracy {accuracy:.2f} below 85% claim"
    
    def test_introspection_manager_throughput_claim(self):
        """Validate IntrospectionManager claim: > 50 operations per second."""
        manager = IntrospectionManager()
        
        # Setup test data
        from ..introspection_manager import AgentIntrospection, PerformanceStatus
        test_introspections = {}
        for i in range(5):
            agent_id = f"test_agent_{i}"
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
        
        manager.agent_introspections = test_introspections
        
        # Measure throughput
        operation_count = 100
        start_time = time.perf_counter()
        
        for _ in range(operation_count):
            result = manager.analyze_system_performance()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = operation_count / duration
        
        assert throughput >= 50, f"Throughput {throughput:.1f} ops/sec below 50 ops/sec claim"
    
    def test_meta_processor_pattern_learning_claim(self):
        """Validate MetaCognitiveProcessor claim: efficient pattern learning < 50ms per update."""
        processor = MetaCognitiveProcessor()
        
        # Create test cognitive analysis
        from ..meta_cognitive_processor import CognitiveAnalysis, CognitiveProcess
        
        analysis = CognitiveAnalysis(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.85,
            quality_metrics={"accuracy": 0.9},
            bias_assessment={"confirmation_bias": 0.2},
            processing_time=0.1,
            confidence_score=0.9
        )
        
        # Measure pattern learning time
        learning_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            processor._update_cognitive_patterns(analysis)
            end_time = time.perf_counter()
            learning_times.append(end_time - start_time)
        
        mean_learning_time = np.mean(learning_times)
        assert mean_learning_time < 0.05, f"Pattern learning time {mean_learning_time:.4f}s exceeds 50ms claim"
    
    def test_memory_efficiency_claim(self):
        """Validate memory efficiency claim: < 100MB per agent instance."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create agent instances
        conscious_agent = ConsciousAgent()
        introspection_manager = IntrospectionManager()
        meta_processor = MetaCognitiveProcessor()
        
        # Perform some operations to initialize internal state
        test_message = {"operation": "introspect", "agent_data": {"performance_metrics": {}}}
        conscious_agent.process(test_message)
        introspection_manager.analyze_system_performance()
        meta_processor.get_meta_insights()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Allow some overhead for test infrastructure
        assert memory_usage < 100, f"Memory usage {memory_usage:.1f}MB exceeds 100MB claim"
    
    def test_error_analysis_effectiveness_claim(self):
        """Validate error analysis effectiveness claim: pattern detection and severity assessment."""
        agent = ConsciousAgent()
        
        # Create test scenario with known error patterns
        test_errors = [
            {"type": "timeout", "severity": "medium", "timestamp": time.time() - 3600},
            {"type": "timeout", "severity": "high", "timestamp": time.time() - 3500},
            {"type": "validation_error", "severity": "low", "timestamp": time.time() - 1800}
        ]
        
        agent_data = {"errors": test_errors}
        
        # Test error analysis
        start_time = time.perf_counter()
        error_analysis = agent._analyze_errors(agent_data)
        analysis_time = time.perf_counter() - start_time
        
        # Validate effectiveness claims
        assert analysis_time < 0.1, f"Error analysis time {analysis_time:.3f}s should be fast"
        assert error_analysis["severity"] in ["high", "medium", "low", "minimal"], "Should categorize severity"
        assert "timeout" in str(error_analysis.get("error_patterns", [])), "Should detect timeout pattern"
        assert error_analysis["confidence"] > 0.5, "Should have reasonable confidence"
        assert error_analysis["total_errors"] == 3, "Should count all errors correctly"


def validate_all_performance_claims():
    """Run all performance validation tests and generate report."""
    import subprocess
    import sys
    
    # Run tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], capture_output=True, text=True)
    
    # Generate validation report
    validation_report = {
        "validation_timestamp": time.time(),
        "all_claims_validated": result.returncode == 0,
        "tests_run": result.stdout.count("PASSED") + result.stdout.count("FAILED"),
        "tests_passed": result.stdout.count("PASSED"),
        "tests_failed": result.stdout.count("FAILED"),
        "performance_claims_status": "VALIDATED" if result.returncode == 0 else "VALIDATION_FAILED"
    }
    
    return validation_report


if __name__ == "__main__":
    # Run validation when executed directly
    report = validate_all_performance_claims()
    print(f"Performance claims validation: {report['performance_claims_status']}")
    print(f"Tests: {report['tests_passed']}/{report['tests_run']} passed") 