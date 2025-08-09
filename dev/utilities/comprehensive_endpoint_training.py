#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing & Training Script
Running systematic tests on all NIS Protocol endpoints with proper parameter formats
"""

import requests
import json
import time
from datetime import datetime

class NISEndpointTrainer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "working": [],
            "partial": [],
            "errors": [],
            "parameter_fixes": [],
            "performance_metrics": {}
        }
        
    def test_endpoint(self, method, endpoint, data=None, timeout=15, description=""):
        """Test an endpoint and capture detailed metrics"""
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
                
            elapsed = time.time() - start_time
            
            result = {
                "endpoint": f"{method.upper()} {endpoint}",
                "description": description,
                "status_code": response.status_code,
                "response_time": round(elapsed, 4),
                "timestamp": datetime.now().isoformat()
            }
            
            if response.status_code == 200:
                try:
                    result["response"] = response.json()
                    result["response_type"] = "json"
                except:
                    result["response"] = response.text[:500] + "..." if len(response.text) > 500 else response.text
                    result["response_type"] = "text"
                    
                if elapsed < 1.0:
                    result["performance"] = "excellent"
                elif elapsed < 5.0:
                    result["performance"] = "good"
                else:
                    result["performance"] = "slow"
                    
                self.results["working"].append(result)
                
            elif response.status_code == 422:
                # Parameter validation error - try to fix
                error_detail = response.json().get('detail', [])
                result["error"] = error_detail
                result["fix_needed"] = "parameter_format"
                self.results["parameter_fixes"].append(result)
                
            else:
                result["error"] = response.text[:200]
                self.results["errors"].append(result)
                
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "endpoint": f"{method.upper()} {endpoint}",
                "description": description,
                "error": str(e),
                "response_time": round(elapsed, 4),
                "timestamp": datetime.now().isoformat()
            }
            self.results["errors"].append(result)
            return result
    
    def run_comprehensive_tests(self):
        """Run comprehensive endpoint testing"""
        print("🧪 STARTING COMPREHENSIVE NIS PROTOCOL ENDPOINT TRAINING")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Base URL: {self.base_url}")
        print()
        
        # Test endpoints with corrected parameters based on previous failures
        test_cases = [
            # Visualization with correct format
            {
                "method": "POST",
                "endpoint": "/visualization/create",
                "data": {
                    "data": {"values": [1, 2, 3, 4, 5], "labels": ["A", "B", "C", "D", "E"]},
                    "chart_type": "bar",
                    "title": "Training Data Visualization"
                },
                "description": "Scientific data visualization"
            },
            
            # Ethics evaluation with action field
            {
                "method": "POST", 
                "endpoint": "/agents/alignment/evaluate_ethics",
                "data": {
                    "action": "provide medical diagnosis recommendation",
                    "scenario": "An AI is asked to help with medical diagnosis",
                    "ethical_frameworks": ["utilitarian", "deontological"]
                },
                "description": "AI ethics evaluation"
            },
            
            # Curiosity with proper stimulus format
            {
                "method": "POST",
                "endpoint": "/agents/curiosity/process_stimulus", 
                "data": {
                    "stimulus": {
                        "type": "information",
                        "content": "A new scientific discovery about quantum computing",
                        "source": "research_paper",
                        "complexity": "high"
                    }
                },
                "description": "Curiosity-driven learning stimulus"
            },
            
            # Learning with operation field
            {
                "method": "POST",
                "endpoint": "/agents/learning/process",
                "data": {
                    "operation": "learn_from_data",
                    "learning_data": "Example machine learning dataset with features and labels",
                    "learning_type": "supervised",
                    "algorithm": "neural_network"
                },
                "description": "Supervised learning processing"
            },
            
            # Simulation with scenario_id and proper structure
            {
                "method": "POST",
                "endpoint": "/agents/simulation/run",
                "data": {
                    "scenario_id": "physics_simulation_001", 
                    "scenario": {
                        "type": "physics",
                        "parameters": {"gravity": 9.8, "time": 10, "mass": 1.0},
                        "environment": "vacuum"
                    }
                },
                "description": "Physics simulation execution"
            },
            
            # Main simulation with concept field
            {
                "method": "POST",
                "endpoint": "/simulation/run",
                "data": {
                    "concept": "quantum_entanglement_experiment",
                    "simulation_config": {
                        "type": "quantum_physics",
                        "duration": 10,
                        "physics_laws": ["quantum_mechanics", "conservation_energy"]
                    }
                },
                "description": "Quantum physics simulation"
            },
            
            # NVIDIA processing with prompt
            {
                "method": "POST",
                "endpoint": "/nvidia/process", 
                "data": {
                    "prompt": "Analyze this physics data using NVIDIA acceleration",
                    "input_data": "sample tensor data for GPU processing",
                    "processing_type": "inference",
                    "model_type": "physics_informed"
                },
                "description": "NVIDIA-accelerated physics processing"
            },
            
            # General processing with text and context
            {
                "method": "POST",
                "endpoint": "/process",
                "data": {
                    "text": "Process this scientific text for key insights",
                    "context": "physics research analysis",
                    "processing_mode": "comprehensive",
                    "extract_insights": True
                },
                "description": "General text processing"
            },
            
            # Agent behavior modification
            {
                "method": "POST", 
                "endpoint": "/agent/behavior/test_agent_001",
                "data": {
                    "behavior_changes": {
                        "learning_rate": 0.01,
                        "exploration_factor": 0.2,
                        "collaboration_mode": "active"
                    },
                    "modification_type": "parameter_tuning"
                },
                "description": "Agent behavior modification"
            },
            
            # BitNet training force
            {
                "method": "POST",
                "endpoint": "/training/bitnet/force",
                "data": {
                    "training_config": {
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    },
                    "force_restart": True
                },
                "description": "Force BitNet model training"
            }
        ]
        
        # Run all test cases
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['description']}")
            print(f"Endpoint: {test_case['method']} {test_case['endpoint']}")
            
            result = self.test_endpoint(
                method=test_case["method"],
                endpoint=test_case["endpoint"], 
                data=test_case.get("data"),
                description=test_case["description"]
            )
            
            if result["status_code"] == 200:
                print(f"✅ SUCCESS - {result['response_time']}s - {result['performance']}")
            elif result.get("fix_needed"):
                print(f"⚠️  PARAMETER FIX NEEDED - {result['status_code']}")
            else:
                print(f"❌ ERROR - {result.get('status_code', 'TIMEOUT')} - {result.get('error', 'Unknown')[:50]}")
                
            time.sleep(0.5)  # Brief pause between requests
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        total_tests = len(self.results["working"]) + len(self.results["partial"]) + len(self.results["errors"]) + len(self.results["parameter_fixes"])
        
        print(f"\n\n🎯 COMPREHENSIVE ENDPOINT TRAINING REPORT")
        print("=" * 80)
        print(f"Total Endpoints Tested: {total_tests}")
        print(f"✅ Fully Working: {len(self.results['working'])}")
        print(f"⚠️  Parameter Fixes Needed: {len(self.results['parameter_fixes'])}")
        print(f"❌ Errors: {len(self.results['errors'])}")
        
        if self.results["working"]:
            avg_response_time = sum(r["response_time"] for r in self.results["working"]) / len(self.results["working"])
            print(f"⚡ Average Response Time: {avg_response_time:.3f}s")
            
            excellent_performance = sum(1 for r in self.results["working"] if r.get("performance") == "excellent")
            print(f"🚀 Excellent Performance (< 1s): {excellent_performance}/{len(self.results['working'])}")
        
        print(f"\n📊 DETAILED RESULTS:")
        
        if self.results["working"]:
            print(f"\n✅ WORKING ENDPOINTS ({len(self.results['working'])}):")
            for result in self.results["working"]:
                print(f"  • {result['endpoint']} - {result['response_time']}s - {result['description']}")
        
        if self.results["parameter_fixes"]:
            print(f"\n⚠️  PARAMETER FIXES NEEDED ({len(self.results['parameter_fixes'])}):")
            for result in self.results["parameter_fixes"]:
                print(f"  • {result['endpoint']} - {result['description']}")
                if isinstance(result.get('error'), list) and result['error']:
                    for error in result['error'][:2]:  # Show first 2 errors
                        missing_field = error.get('loc', [])[-1] if error.get('loc') else 'unknown'
                        print(f"    - Missing/Invalid: {missing_field}")
        
        if self.results["errors"]:
            print(f"\n❌ ERRORS ({len(self.results['errors'])}):")
            for result in self.results["errors"]:
                print(f"  • {result['endpoint']} - {result['description']} - {result.get('error', 'Unknown')[:50]}")
        
        # Save detailed report
        report_file = f"nis_endpoint_training_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print(f"\n🌙 Training session complete! System analysis ready for tomorrow.")
        print("=" * 80)

if __name__ == "__main__":
    trainer = NISEndpointTrainer()
    trainer.run_comprehensive_tests()
    trainer.generate_training_report()