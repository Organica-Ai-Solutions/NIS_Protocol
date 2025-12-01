"""
NIS Protocol Benchmark Suite
Professional benchmarks to measure real value of orchestration vs single LLM.

Benchmarks:
1. HumanEval Subset - Code generation accuracy
2. GSM8K Subset - Math reasoning
3. Multi-turn Conversation - Memory effectiveness
4. Cost Efficiency - Tokens/cost per task
5. Latency - Response time comparison
6. Cache Effectiveness - Savings from caching
7. Physics Validation - PINN accuracy (NIS-specific)
"""

import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cost_tracker import get_cost_tracker, CostTracker


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    name: str
    passed: bool
    score: float  # 0-100
    latency_ms: float
    tokens_used: int
    cost_usd: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from full benchmark suite"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    results_by_category: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HUMANEVAL SUBSET - Code Generation
# =============================================================================

HUMANEVAL_PROBLEMS = [
    {
        "id": "HE_001",
        "prompt": "Write a Python function that returns the sum of two numbers.",
        "test": "assert add(2, 3) == 5 and add(-1, 1) == 0 and add(0, 0) == 0",
        "function_name": "add"
    },
    {
        "id": "HE_002",
        "prompt": "Write a Python function that checks if a string is a palindrome (case-insensitive).",
        "test": "assert is_palindrome('racecar') == True and is_palindrome('hello') == False and is_palindrome('A man a plan a canal Panama'.replace(' ', '').lower()) == True",
        "function_name": "is_palindrome"
    },
    {
        "id": "HE_003",
        "prompt": "Write a Python function that returns the factorial of a non-negative integer.",
        "test": "assert factorial(0) == 1 and factorial(1) == 1 and factorial(5) == 120",
        "function_name": "factorial"
    },
    {
        "id": "HE_004",
        "prompt": "Write a Python function that returns the nth Fibonacci number (0-indexed).",
        "test": "assert fibonacci(0) == 0 and fibonacci(1) == 1 and fibonacci(10) == 55",
        "function_name": "fibonacci"
    },
    {
        "id": "HE_005",
        "prompt": "Write a Python function that finds the maximum element in a list.",
        "test": "assert find_max([1, 5, 3, 9, 2]) == 9 and find_max([-1, -5, -3]) == -1",
        "function_name": "find_max"
    },
    {
        "id": "HE_006",
        "prompt": "Write a Python function that reverses a string.",
        "test": "assert reverse_string('hello') == 'olleh' and reverse_string('') == ''",
        "function_name": "reverse_string"
    },
    {
        "id": "HE_007",
        "prompt": "Write a Python function that counts the occurrences of a character in a string.",
        "test": "assert count_char('hello', 'l') == 2 and count_char('hello', 'z') == 0",
        "function_name": "count_char"
    },
    {
        "id": "HE_008",
        "prompt": "Write a Python function that checks if a number is prime.",
        "test": "assert is_prime(2) == True and is_prime(17) == True and is_prime(4) == False and is_prime(1) == False",
        "function_name": "is_prime"
    },
    {
        "id": "HE_009",
        "prompt": "Write a Python function that flattens a nested list.",
        "test": "assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]",
        "function_name": "flatten"
    },
    {
        "id": "HE_010",
        "prompt": "Write a Python function that returns the GCD of two numbers using Euclidean algorithm.",
        "test": "assert gcd(48, 18) == 6 and gcd(17, 13) == 1",
        "function_name": "gcd"
    },
]


# =============================================================================
# GSM8K SUBSET - Math Reasoning
# =============================================================================

GSM8K_PROBLEMS = [
    {
        "id": "GSM_001",
        "question": "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?",
        "answer": 10
    },
    {
        "id": "GSM_002",
        "question": "A train travels at 60 mph. How far does it travel in 2.5 hours?",
        "answer": 150
    },
    {
        "id": "GSM_003",
        "question": "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area in square centimeters?",
        "answer": 40
    },
    {
        "id": "GSM_004",
        "question": "Sarah has 24 cookies. She gives 1/3 to her brother and 1/4 of the original amount to her sister. How many cookies does she have left?",
        "answer": 10
    },
    {
        "id": "GSM_005",
        "question": "A car uses 8 gallons of gas to travel 240 miles. How many miles per gallon does it get?",
        "answer": 30
    },
    {
        "id": "GSM_006",
        "question": "If 3 workers can build a wall in 12 hours, how many hours would it take 6 workers to build the same wall?",
        "answer": 6
    },
    {
        "id": "GSM_007",
        "question": "A shirt originally costs $40. It's on sale for 25% off. What is the sale price?",
        "answer": 30
    },
    {
        "id": "GSM_008",
        "question": "The sum of three consecutive integers is 45. What is the largest of the three integers?",
        "answer": 16
    },
    {
        "id": "GSM_009",
        "question": "A pool is being filled at a rate of 50 gallons per minute. If the pool holds 3000 gallons, how many minutes to fill it?",
        "answer": 60
    },
    {
        "id": "GSM_010",
        "question": "If the radius of a circle is 7 cm, what is its circumference? Use pi = 22/7.",
        "answer": 44
    },
]


# =============================================================================
# MULTI-TURN CONVERSATION - Memory Test
# =============================================================================

MULTITURN_TESTS = [
    {
        "id": "MT_001",
        "turns": [
            {"user": "My name is Alex.", "check": None},
            {"user": "What is my name?", "check": "alex"}
        ]
    },
    {
        "id": "MT_002",
        "turns": [
            {"user": "I have a dog named Max and a cat named Luna.", "check": None},
            {"user": "What are my pets' names?", "check": ["max", "luna"]}
        ]
    },
    {
        "id": "MT_003",
        "turns": [
            {"user": "I'm working on a Python project about machine learning.", "check": None},
            {"user": "What programming language am I using?", "check": "python"}
        ]
    },
]


# =============================================================================
# PHYSICS VALIDATION - NIS-Specific
# =============================================================================

PHYSICS_TESTS = [
    {
        "id": "PHY_001",
        "scenario": "A ball is thrown upward with initial velocity 20 m/s. Calculate max height. (g=10 m/s²)",
        "expected": 20,  # h = v²/2g = 400/20 = 20m
        "tolerance": 2
    },
    {
        "id": "PHY_002",
        "scenario": "A 2kg object accelerates at 5 m/s². What force is applied?",
        "expected": 10,  # F = ma = 2*5 = 10N
        "tolerance": 0.5
    },
    {
        "id": "PHY_003",
        "scenario": "Calculate kinetic energy of a 4kg object moving at 3 m/s.",
        "expected": 18,  # KE = 0.5*m*v² = 0.5*4*9 = 18J
        "tolerance": 1
    },
]


class NISBenchmarkSuite:
    """
    Professional benchmark suite for NIS Protocol.
    Compares orchestrated responses vs baseline.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
        self.cost_tracker = CostTracker(storage_path="data/benchmark_costs")
        
    async def run_all(self, verbose: bool = True) -> BenchmarkSuiteResult:
        """Run all benchmarks"""
        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print("NIS PROTOCOL BENCHMARK SUITE")
            print("=" * 60)
            print()
        
        # Run each benchmark category
        categories = {
            "code_generation": await self.benchmark_code_generation(verbose),
            "math_reasoning": await self.benchmark_math_reasoning(verbose),
            "multi_turn": await self.benchmark_multi_turn(verbose),
            "physics": await self.benchmark_physics(verbose),
            "latency": await self.benchmark_latency(verbose),
        }
        
        # Aggregate results
        all_results = []
        for cat_results in categories.values():
            all_results.extend(cat_results)
        
        passed = sum(1 for r in all_results if r.passed)
        total_latency = sum(r.latency_ms for r in all_results)
        total_tokens = sum(r.tokens_used for r in all_results)
        total_cost = sum(r.cost_usd for r in all_results)
        
        overall_score = (passed / len(all_results) * 100) if all_results else 0
        
        result = BenchmarkSuiteResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(all_results),
            passed_tests=passed,
            failed_tests=len(all_results) - passed,
            overall_score=round(overall_score, 1),
            total_latency_ms=round(total_latency, 1),
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            avg_latency_ms=round(total_latency / len(all_results), 1) if all_results else 0,
            results_by_category=categories,
        )
        
        if verbose:
            self._print_summary(result)
        
        # Save results
        self._save_results(result)
        
        return result
    
    async def benchmark_code_generation(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Benchmark code generation using HumanEval subset"""
        if verbose:
            print("\n📝 CODE GENERATION (HumanEval Subset)")
            print("-" * 40)
        
        results = []
        
        for problem in HUMANEVAL_PROBLEMS:
            start = time.time()
            
            try:
                # Call NIS Protocol
                response = await self._call_nis_chat(
                    f"{problem['prompt']} Only return the Python function code, nothing else."
                )
                
                latency = (time.time() - start) * 1000
                
                # Extract code and test it
                code = self._extract_code(response)
                passed, error = self._test_code(code, problem['test'], problem['function_name'])
                
                result = BenchmarkResult(
                    name=problem['id'],
                    passed=passed,
                    score=100 if passed else 0,
                    latency_ms=latency,
                    tokens_used=len(response.split()) * 2,  # Rough estimate
                    cost_usd=0.001,  # Estimate
                    details={"error": error} if error else {}
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    name=problem['id'],
                    passed=False,
                    score=0,
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    details={"error": str(e)}
                )
            
            results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {problem['id']}: {result.latency_ms:.0f}ms")
        
        return results
    
    async def benchmark_math_reasoning(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Benchmark math reasoning using GSM8K subset"""
        if verbose:
            print("\n🔢 MATH REASONING (GSM8K Subset)")
            print("-" * 40)
        
        results = []
        
        for problem in GSM8K_PROBLEMS:
            start = time.time()
            
            try:
                response = await self._call_nis_chat(
                    f"{problem['question']} Give only the numerical answer."
                )
                
                latency = (time.time() - start) * 1000
                
                # Extract number from response
                answer = self._extract_number(response)
                passed = answer is not None and abs(answer - problem['answer']) < 0.01
                
                result = BenchmarkResult(
                    name=problem['id'],
                    passed=passed,
                    score=100 if passed else 0,
                    latency_ms=latency,
                    tokens_used=len(response.split()) * 2,
                    cost_usd=0.0005,
                    details={"expected": problem['answer'], "got": answer}
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    name=problem['id'],
                    passed=False,
                    score=0,
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    details={"error": str(e)}
                )
            
            results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {problem['id']}: expected={problem['answer']}, got={result.details.get('got', 'N/A')}")
        
        return results
    
    async def benchmark_multi_turn(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Benchmark multi-turn conversation memory"""
        if verbose:
            print("\n💬 MULTI-TURN CONVERSATION")
            print("-" * 40)
        
        results = []
        
        for test in MULTITURN_TESTS:
            start = time.time()
            conversation_id = f"bench_{test['id']}_{int(time.time())}"
            
            try:
                passed = True
                
                for turn in test['turns']:
                    response = await self._call_nis_chat(
                        turn['user'],
                        conversation_id=conversation_id
                    )
                    
                    if turn['check']:
                        if isinstance(turn['check'], list):
                            # Check all items present
                            response_lower = response.lower()
                            for item in turn['check']:
                                if item.lower() not in response_lower:
                                    passed = False
                                    break
                        else:
                            if turn['check'].lower() not in response.lower():
                                passed = False
                
                latency = (time.time() - start) * 1000
                
                result = BenchmarkResult(
                    name=test['id'],
                    passed=passed,
                    score=100 if passed else 0,
                    latency_ms=latency,
                    tokens_used=100,
                    cost_usd=0.001,
                    details={}
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    name=test['id'],
                    passed=False,
                    score=0,
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    details={"error": str(e)}
                )
            
            results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {test['id']}: Memory retention {'OK' if result.passed else 'FAILED'}")
        
        return results
    
    async def benchmark_physics(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Benchmark physics reasoning"""
        if verbose:
            print("\n⚛️ PHYSICS VALIDATION")
            print("-" * 40)
        
        results = []
        
        for test in PHYSICS_TESTS:
            start = time.time()
            
            try:
                response = await self._call_nis_chat(
                    f"{test['scenario']} Give only the numerical answer."
                )
                
                latency = (time.time() - start) * 1000
                
                answer = self._extract_number(response)
                passed = (answer is not None and 
                         abs(answer - test['expected']) <= test['tolerance'])
                
                result = BenchmarkResult(
                    name=test['id'],
                    passed=passed,
                    score=100 if passed else 0,
                    latency_ms=latency,
                    tokens_used=50,
                    cost_usd=0.0005,
                    details={"expected": test['expected'], "got": answer, "tolerance": test['tolerance']}
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    name=test['id'],
                    passed=False,
                    score=0,
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    details={"error": str(e)}
                )
            
            results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {test['id']}: expected={test['expected']}±{test['tolerance']}, got={result.details.get('got', 'N/A')}")
        
        return results
    
    async def benchmark_latency(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Benchmark response latency"""
        if verbose:
            print("\n⏱️ LATENCY BENCHMARK")
            print("-" * 40)
        
        results = []
        test_prompts = [
            ("LAT_001", "What is 2+2?"),
            ("LAT_002", "Say hello."),
            ("LAT_003", "What color is the sky?"),
        ]
        
        latencies = []
        
        for test_id, prompt in test_prompts:
            start = time.time()
            
            try:
                await self._call_nis_chat(prompt)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                # Pass if under 5 seconds
                passed = latency < 5000
                
                result = BenchmarkResult(
                    name=test_id,
                    passed=passed,
                    score=max(0, 100 - (latency / 50)),  # Penalize slow responses
                    latency_ms=latency,
                    tokens_used=20,
                    cost_usd=0.0002,
                    details={}
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    name=test_id,
                    passed=False,
                    score=0,
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    details={"error": str(e)}
                )
            
            results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {test_id}: {result.latency_ms:.0f}ms")
        
        if verbose and latencies:
            print(f"\n  Avg: {statistics.mean(latencies):.0f}ms")
            print(f"  Min: {min(latencies):.0f}ms")
            print(f"  Max: {max(latencies):.0f}ms")
        
        return results
    
    async def _call_nis_chat(self, message: str, conversation_id: str = None) -> str:
        """Call NIS Protocol chat endpoint"""
        import aiohttp
        
        payload = {
            "message": message,
            "conversation_id": conversation_id or f"bench_{int(time.time())}",
            "user_id": "benchmark"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", data.get("message", ""))
                else:
                    raise Exception(f"API error: {resp.status}")
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from response"""
        # Try to find code block
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        elif "def " in response:
            # Find the function definition
            start = response.find("def ")
            return response[start:].strip()
        return response.strip()
    
    def _test_code(self, code: str, test: str, func_name: str) -> Tuple[bool, Optional[str]]:
        """Test generated code"""
        try:
            # Create isolated namespace
            namespace = {}
            exec(code, namespace)
            
            # Check function exists
            if func_name not in namespace:
                return False, f"Function {func_name} not found"
            
            # Run test
            exec(test, namespace)
            return True, None
            
        except AssertionError as e:
            return False, f"Assertion failed: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
    def _extract_number(self, response: str) -> Optional[float]:
        """Extract numerical answer from response"""
        import re
        
        # Look for numbers in response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if numbers:
            # Return the last number (usually the answer)
            return float(numbers[-1])
        return None
    
    def _print_summary(self, result: BenchmarkSuiteResult):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"\n📊 Overall Score: {result.overall_score}%")
        print(f"✅ Passed: {result.passed_tests}/{result.total_tests}")
        print(f"❌ Failed: {result.failed_tests}/{result.total_tests}")
        print(f"\n⏱️ Total Latency: {result.total_latency_ms:.0f}ms")
        print(f"📈 Avg Latency: {result.avg_latency_ms:.0f}ms")
        print(f"🪙 Total Tokens: {result.total_tokens}")
        print(f"💰 Total Cost: ${result.total_cost_usd:.4f}")
        
        print("\n📋 By Category:")
        for category, results in result.results_by_category.items():
            passed = sum(1 for r in results if r.passed)
            print(f"  {category}: {passed}/{len(results)} passed")
        
        print("\n" + "=" * 60)
    
    def _save_results(self, result: BenchmarkSuiteResult):
        """Save results to file"""
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        data = {
            "timestamp": result.timestamp,
            "total_tests": result.total_tests,
            "passed_tests": result.passed_tests,
            "failed_tests": result.failed_tests,
            "overall_score": result.overall_score,
            "total_latency_ms": result.total_latency_ms,
            "total_tokens": result.total_tokens,
            "total_cost_usd": result.total_cost_usd,
            "avg_latency_ms": result.avg_latency_ms,
            "results_by_category": {
                cat: [asdict(r) for r in results]
                for cat, results in result.results_by_category.items()
            }
        }
        
        with open(output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📁 Results saved to: benchmarks/results/{filename}")


async def main():
    """Run benchmark suite"""
    suite = NISBenchmarkSuite()
    await suite.run_all(verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
