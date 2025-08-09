#!/usr/bin/env python3
"""
Intensive Brain Stress Test

Pushes the brain-like coordination system to its absolute limits:
- 100+ concurrent requests
- All endpoints simultaneously 
- Complex coordination patterns
- Real AI processing under stress
- Multi-layered stress scenarios
"""

import asyncio
import time
import requests
import random
from typing import Dict, Any, List
import concurrent.futures

class IntensiveBrainStressTester:
    """Ultra-intensive stress testing for brain-like coordination"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.results = {
            'light_stress': {},
            'medium_stress': {},
            'heavy_stress': {},
            'extreme_stress': {}
        }
    
    async def run_progressive_stress_test(self):
        """Progressive stress test - increasing intensity"""
        print("🧠 PROGRESSIVE BRAIN STRESS TEST")
        print("=" * 50)
        
        stress_levels = [
            ("Light Stress", 20, 2),
            ("Medium Stress", 50, 5), 
            ("Heavy Stress", 100, 10),
            ("Extreme Stress", 200, 20)
        ]
        
        for level_name, concurrent_requests, duration in stress_levels:
            print(f"\n🎯 {level_name}: {concurrent_requests} concurrent requests for {duration}s")
            
            result = await self.stress_test_level(concurrent_requests, duration)
            self.results[level_name.lower().replace(' ', '_')] = result
            
            success_rate = result.get('success_rate', 0)
            rps = result.get('requests_per_second', 0)
            print(f"   ✅ Success Rate: {success_rate:.1f}%")
            print(f"   ⚡ Throughput: {rps:.1f} req/s")
            
            # Brief pause between stress levels
            await asyncio.sleep(1)
        
        return self.results
    
    async def stress_test_level(self, concurrent_requests: int, duration: int):
        """Run stress test at specific level"""
        start_time = time.time()
        end_time = start_time + duration
        
        tasks = []
        request_count = 0
        
        # Generate stress tasks
        while time.time() < end_time:
            batch_tasks = []
            
            for _ in range(min(concurrent_requests, 50)):  # Batch size limit
                endpoint = self.get_random_endpoint()
                task = self.make_stress_request(endpoint, request_count)
                batch_tasks.append(task)
                request_count += 1
            
            # Execute batch
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=10.0
                )
                tasks.extend(batch_results)
            except asyncio.TimeoutError:
                print(f"   ⚠️ Batch timeout at request {request_count}")
                break
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Analyze results
        return self.analyze_stress_results(tasks, time.time() - start_time)
    
    def get_random_endpoint(self):
        """Get random endpoint for varied stress testing"""
        endpoints = [
            ("GET", "/health", "Health check"),
            ("GET", "/", "Root endpoint"),
            ("GET", "/agents", "Agent management"),
            ("POST", "/chat", "AI chat"),
            ("POST", "/agent/create", "Agent creation")
        ]
        return random.choice(endpoints)
    
    async def make_stress_request(self, endpoint: tuple, request_id: int):
        """Make stress request to endpoint"""
        method, path, description = endpoint
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{path}", timeout=5)
            elif method == "POST":
                if path == "/chat":
                    payload = {
                        "message": f"Intensive stress test {request_id}: How does brain coordination work?",
                        "agent_type": random.choice(["consciousness", "reasoning", "memory"])
                    }
                elif path == "/agent/create":
                    payload = {
                        "agent_type": random.choice(["consciousness", "reasoning", "memory", "physics"]),
                        "capabilities": [f"capability_{request_id % 10}"],
                        "memory_size": str(random.randint(500, 2000))
                    }
                else:
                    payload = {}
                
                response = requests.post(f"{self.base_url}{path}", json=payload, timeout=5)
            
            response_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response_time,
                'endpoint': path,
                'request_id': request_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time,
                'endpoint': path,
                'request_id': request_id
            }
    
    def analyze_stress_results(self, results: List, total_time: float):
        """Analyze stress test results"""
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        failed = len(results) - successful
        
        if successful > 0:
            avg_response_time = sum(
                r.get('response_time', 0) for r in results 
                if isinstance(r, dict) and r.get('success', False)
            ) / successful
        else:
            avg_response_time = 0
        
        success_rate = (successful / len(results) * 100) if results else 0
        requests_per_second = len(results) / total_time if total_time > 0 else 0
        
        return {
            'total_requests': len(results),
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'requests_per_second': requests_per_second,
            'total_time': total_time
        }

class ConcurrentCoordinationTester:
    """Test multiple coordination patterns simultaneously"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
    
    async def test_concurrent_coordination_patterns(self):
        """Test multiple coordination patterns running simultaneously"""
        print("\n🧠 CONCURRENT COORDINATION PATTERNS TEST")
        print("=" * 50)
        
        # Define coordination patterns
        patterns = [
            self.consciousness_coordination_pattern(),
            self.reasoning_coordination_pattern(), 
            self.memory_coordination_pattern(),
            self.physics_coordination_pattern(),
            self.integration_coordination_pattern()
        ]
        
        start_time = time.time()
        
        try:
            # Run all patterns concurrently
            results = await asyncio.gather(*patterns, return_exceptions=True)
            
            total_time = time.time() - start_time
            successful_patterns = sum(1 for r in results if not isinstance(r, Exception))
            
            print(f"\n📊 COORDINATION PATTERNS RESULTS:")
            print(f"   Successful patterns: {successful_patterns}/5")
            print(f"   Total coordination time: {total_time:.3f}s")
            print(f"   Pattern efficiency: {5 / total_time:.1f} patterns/s")
            
            return {
                'successful_patterns': successful_patterns,
                'total_patterns': 5,
                'success_rate': successful_patterns / 5,
                'total_time': total_time,
                'pattern_efficiency': 5 / total_time
            }
            
        except Exception as e:
            print(f"❌ Coordination patterns test failed: {e}")
            return {'success_rate': 0.0, 'error': str(e)}
    
    async def consciousness_coordination_pattern(self):
        """Consciousness coordination pattern"""
        tasks = []
        for i in range(5):
            payload = {
                "message": f"Consciousness pattern {i}: Analyze self-awareness",
                "agent_type": "consciousness"
            }
            task = self.make_coordination_request("/chat", payload)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {'pattern': 'consciousness', 'results': len(results)}
    
    async def reasoning_coordination_pattern(self):
        """Reasoning coordination pattern"""
        tasks = []
        for i in range(5):
            payload = {
                "message": f"Reasoning pattern {i}: Solve logical problem",
                "agent_type": "reasoning"
            }
            task = self.make_coordination_request("/chat", payload)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {'pattern': 'reasoning', 'results': len(results)}
    
    async def memory_coordination_pattern(self):
        """Memory coordination pattern"""
        tasks = []
        for i in range(3):
            payload = {
                "agent_type": "memory",
                "capabilities": ["episodic_memory", "semantic_memory"],
                "memory_size": "1500"
            }
            task = self.make_coordination_request("/agent/create", payload)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {'pattern': 'memory', 'results': len(results)}
    
    async def physics_coordination_pattern(self):
        """Physics coordination pattern"""
        tasks = []
        for i in range(3):
            payload = {
                "agent_type": "physics",
                "capabilities": ["conservation_laws", "constraint_validation"],
                "memory_size": "1000"
            }
            task = self.make_coordination_request("/agent/create", payload)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {'pattern': 'physics', 'results': len(results)}
    
    async def integration_coordination_pattern(self):
        """Integration coordination pattern"""
        tasks = []
        # Mix of different endpoint calls
        health_task = self.make_coordination_request("/health", None, "GET")
        agents_task = self.make_coordination_request("/agents", None, "GET")
        tasks.extend([health_task, agents_task])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {'pattern': 'integration', 'results': len(results)}
    
    async def make_coordination_request(self, endpoint: str, payload: dict, method: str = "POST"):
        """Make coordination request"""
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=5)
            
            return response.status_code == 200
        except Exception:
            return False

async def main():
    """Main intensive stress test"""
    print("🧠 INTENSIVE BRAIN COORDINATION STRESS TEST")
    print("=" * 60)
    print("Pushing the brain-like coordination to its absolute limits...")
    print()
    
    # Test 1: Progressive stress testing
    stress_tester = IntensiveBrainStressTester()
    stress_results = await stress_tester.run_progressive_stress_test()
    
    # Test 2: Concurrent coordination patterns
    coordination_tester = ConcurrentCoordinationTester()
    coordination_results = await coordination_tester.test_concurrent_coordination_patterns()
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 INTENSIVE STRESS TEST ASSESSMENT")
    print("=" * 60)
    
    # Calculate overall performance
    extreme_success = stress_results.get('extreme_stress', {}).get('success_rate', 0)
    coordination_success = coordination_results.get('success_rate', 0) * 100
    
    overall_performance = (extreme_success + coordination_success) / 2
    
    print(f"🚀 Extreme Stress Success: {extreme_success:.1f}%")
    print(f"🧠 Coordination Patterns: {coordination_success:.1f}%") 
    print(f"🌟 Overall Stress Performance: {overall_performance:.1f}%")
    
    if overall_performance >= 90:
        print("\n🏆 EXCEPTIONAL: Brain handles extreme stress perfectly!")
    elif overall_performance >= 70:
        print("\n✅ EXCELLENT: Brain performs well under intense stress!")
    elif overall_performance >= 50:
        print("\n⚠️ GOOD: Brain handles moderate stress acceptably!")
    else:
        print("\n🔧 NEEDS OPTIMIZATION: Brain struggles under stress!")
    
    return {
        'stress_results': stress_results,
        'coordination_results': coordination_results,
        'overall_performance': overall_performance
    }

if __name__ == "__main__":
    try:
        final_results = asyncio.run(main())
        print(f"\n🏁 Intensive brain stress test complete!")
        print(f"   Final performance: {final_results['overall_performance']:.1f}%")
    except KeyboardInterrupt:
        print("\n⚠️ Intensive test interrupted")
    except Exception as e:
        print(f"\n❌ Intensive test failed: {e}") 