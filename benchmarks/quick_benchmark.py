#!/usr/bin/env python3
"""
NIS Protocol Quick Benchmark
Runs benchmarks with better timeout handling and parallel execution.
"""

import asyncio
import aiohttp
import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Reduced test sets for faster execution
QUICK_CODE_TESTS = [
    {"id": "CODE_1", "prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.", 
     "test": "assert add(2, 3) == 5", "func": "add"},
    {"id": "CODE_2", "prompt": "Write a Python function `factorial(n)` that returns n factorial.", 
     "test": "assert factorial(5) == 120", "func": "factorial"},
    {"id": "CODE_3", "prompt": "Write a Python function `is_prime(n)` that returns True if n is prime.", 
     "test": "assert is_prime(7) == True and is_prime(4) == False", "func": "is_prime"},
]

QUICK_MATH_TESTS = [
    {"id": "MATH_1", "q": "What is 15 * 8?", "a": 120},
    {"id": "MATH_2", "q": "A car travels 60 mph for 2.5 hours. How many miles?", "a": 150},
    {"id": "MATH_3", "q": "What is 25% of 80?", "a": 20},
]

QUICK_PHYSICS_TESTS = [
    {"id": "PHYS_1", "q": "F=ma. If m=5kg and a=3m/s², what is F in Newtons?", "a": 15, "tol": 1},
    {"id": "PHYS_2", "q": "KE = 0.5*m*v². If m=2kg and v=4m/s, what is KE in Joules?", "a": 16, "tol": 1},
]


@dataclass
class TestResult:
    id: str
    category: str
    passed: bool
    latency_ms: float
    details: str = ""


class QuickBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    async def check_server(self) -> bool:
        """Check if server is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def call_chat(self, message: str, timeout: int = 45) -> Tuple[Optional[str], float]:
        """Call chat endpoint with timeout"""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json={"message": message, "conversation_id": f"bench_{int(time.time())}", "user_id": "benchmark"},
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    latency = (time.time() - start) * 1000
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", data.get("message", "")), latency
                    return None, latency
        except asyncio.TimeoutError:
            return None, (time.time() - start) * 1000
        except Exception as e:
            return None, (time.time() - start) * 1000
    
    def extract_code(self, response: str) -> str:
        """Extract code from response"""
        if "```python" in response:
            match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return match.group(1)
        if "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return match.group(1)
        if "def " in response:
            start = response.find("def ")
            return response[start:]
        return response
    
    def test_code(self, code: str, test: str, func_name: str) -> Tuple[bool, str]:
        """Test generated code"""
        try:
            namespace = {}
            exec(code, namespace)
            if func_name not in namespace:
                return False, f"Function {func_name} not found"
            exec(test, namespace)
            return True, "OK"
        except AssertionError:
            return False, "Assertion failed"
        except Exception as e:
            return False, str(e)[:50]
    
    def extract_number(self, response: str) -> Optional[float]:
        """Extract number from response"""
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return float(numbers[-1])
        return None
    
    async def run_code_tests(self) -> List[TestResult]:
        """Run code generation tests"""
        print("\n📝 CODE GENERATION")
        print("-" * 40)
        results = []
        
        for test in QUICK_CODE_TESTS:
            response, latency = await self.call_chat(
                f"{test['prompt']} Return only the function code."
            )
            
            if response:
                code = self.extract_code(response)
                passed, detail = self.test_code(code, test['test'], test['func'])
            else:
                passed, detail = False, "Timeout/Error"
            
            result = TestResult(test['id'], 'code', passed, latency, detail)
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"  {status} {test['id']}: {latency:.0f}ms - {detail}")
        
        return results
    
    async def run_math_tests(self) -> List[TestResult]:
        """Run math reasoning tests"""
        print("\n🔢 MATH REASONING")
        print("-" * 40)
        results = []
        
        for test in QUICK_MATH_TESTS:
            response, latency = await self.call_chat(
                f"{test['q']} Answer with just the number."
            )
            
            if response:
                answer = self.extract_number(response)
                passed = answer is not None and abs(answer - test['a']) < 0.1
                detail = f"expected={test['a']}, got={answer}"
            else:
                passed, detail = False, "Timeout/Error"
            
            result = TestResult(test['id'], 'math', passed, latency, detail)
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"  {status} {test['id']}: {latency:.0f}ms - {detail}")
        
        return results
    
    async def run_physics_tests(self) -> List[TestResult]:
        """Run physics tests"""
        print("\n⚛️ PHYSICS")
        print("-" * 40)
        results = []
        
        for test in QUICK_PHYSICS_TESTS:
            response, latency = await self.call_chat(
                f"{test['q']} Answer with just the number."
            )
            
            if response:
                answer = self.extract_number(response)
                passed = answer is not None and abs(answer - test['a']) <= test['tol']
                detail = f"expected={test['a']}±{test['tol']}, got={answer}"
            else:
                passed, detail = False, "Timeout/Error"
            
            result = TestResult(test['id'], 'physics', passed, latency, detail)
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"  {status} {test['id']}: {latency:.0f}ms - {detail}")
        
        return results
    
    async def run_latency_test(self) -> List[TestResult]:
        """Run latency tests"""
        print("\n⏱️ LATENCY")
        print("-" * 40)
        results = []
        
        prompts = [("LAT_1", "Say hello"), ("LAT_2", "What is 2+2?"), ("LAT_3", "Hi")]
        
        for test_id, prompt in prompts:
            response, latency = await self.call_chat(prompt, timeout=30)
            passed = response is not None and latency < 30000
            detail = f"{latency:.0f}ms"
            
            result = TestResult(test_id, 'latency', passed, latency, detail)
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"  {status} {test_id}: {detail}")
        
        return results
    
    async def run_all(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("NIS PROTOCOL QUICK BENCHMARK")
        print("=" * 60)
        
        # Check server
        if not await self.check_server():
            print("\n❌ Server not running at", self.base_url)
            print("   Start with: python -m uvicorn main:app --port 8000")
            return
        
        print("✅ Server connected")
        
        # Run tests sequentially to avoid overwhelming server
        all_results = []
        
        all_results.extend(await self.run_code_tests())
        all_results.extend(await self.run_math_tests())
        all_results.extend(await self.run_physics_tests())
        all_results.extend(await self.run_latency_test())
        
        # Summary
        self._print_summary(all_results)
        self._save_results(all_results)
    
    def _print_summary(self, results: List[TestResult]):
        """Print summary"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_latency = sum(r.latency_ms for r in results)
        
        print(f"\n📊 Score: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"⏱️ Total time: {total_latency/1000:.1f}s")
        print(f"📈 Avg latency: {total_latency/total:.0f}ms")
        
        # By category
        print("\n📋 By Category:")
        categories = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "total": 0}
            categories[r.category]["total"] += 1
            if r.passed:
                categories[r.category]["passed"] += 1
        
        for cat, stats in categories.items():
            pct = stats["passed"] / stats["total"] * 100
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
        
        # Verdict
        print("\n" + "=" * 60)
        score = passed / total * 100
        if score >= 80:
            print("🏆 EXCELLENT")
        elif score >= 60:
            print("✅ GOOD")
        elif score >= 40:
            print("⚠️ NEEDS IMPROVEMENT")
        else:
            print("❌ NEEDS WORK")
        print("=" * 60)
    
    def _save_results(self, results: List[TestResult]):
        """Save results to file"""
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"quick_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in results],
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "avg_latency_ms": sum(r.latency_ms for r in results) / len(results)
            }
        }
        
        with open(output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📁 Saved: benchmarks/results/{filename}")


async def main():
    benchmark = QuickBenchmark()
    await benchmark.run_all()


if __name__ == "__main__":
    asyncio.run(main())
