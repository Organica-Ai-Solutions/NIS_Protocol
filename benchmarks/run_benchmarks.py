#!/usr/bin/env python3
"""
NIS Protocol Benchmark Runner
Run professional benchmarks and generate comparison report.

Usage:
    python benchmarks/run_benchmarks.py [--quick] [--verbose]
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_suite import NISBenchmarkSuite


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███╗   ██╗██╗███████╗    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗    ║
║     ████╗  ██║██║██╔════╝    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║    ║
║     ██╔██╗ ██║██║███████╗    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║    ║
║     ██║╚██╗██║██║╚════██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║    ║
║     ██║ ╚████║██║███████║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║    ║
║     ╚═╝  ╚═══╝╚═╝╚══════╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝    ║
║                                                               ║
║              Professional AI Benchmark Suite v1.0             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_comparison_table():
    """Print comparison with known LLM benchmarks"""
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REFERENCE: Industry LLM Benchmarks                       │
├─────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Model               │ MMLU     │ HumanEval│ GSM8K    │ HellaSwag│ Cost/1M  │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ GPT-4o              │ 88.7%    │ 90.2%    │ 95.3%    │ 95.3%    │ $2.50    │
│ GPT-4o-mini         │ 82.0%    │ 87.2%    │ 93.2%    │ 89.0%    │ $0.15    │
│ Claude 3.5 Sonnet   │ 88.7%    │ 92.0%    │ 96.4%    │ 89.0%    │ $3.00    │
│ Claude 3 Haiku      │ 75.2%    │ 75.9%    │ 88.9%    │ 85.9%    │ $0.25    │
│ Gemini 1.5 Pro      │ 85.9%    │ 84.1%    │ 91.7%    │ 92.5%    │ $1.25    │
│ Llama 3 70B         │ 82.0%    │ 81.7%    │ 93.0%    │ 88.0%    │ Free*    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ NIS Protocol v4.0   │ TBD      │ TBD      │ TBD      │ N/A      │ Variable │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

* NIS Protocol uses underlying LLMs - benchmarks measure orchestration value
* Cost varies based on provider selection and caching effectiveness
    """)


async def run_benchmarks(quick: bool = False, verbose: bool = True):
    """Run the benchmark suite"""
    
    print_banner()
    print_comparison_table()
    
    print("\n🚀 Starting NIS Protocol Benchmarks...")
    print("   Make sure the backend is running: ./start.sh")
    print()
    
    # Check if server is running
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print("❌ Backend not responding. Please start with ./start.sh")
                    return
                print("✅ Backend is running\n")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Please start the backend with: ./start.sh")
        return
    
    # Run benchmarks
    suite = NISBenchmarkSuite()
    result = await suite.run_all(verbose=verbose)
    
    # Print final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    if result.overall_score >= 90:
        verdict = "🏆 EXCELLENT - Production Ready"
    elif result.overall_score >= 75:
        verdict = "✅ GOOD - Minor improvements needed"
    elif result.overall_score >= 50:
        verdict = "⚠️ FAIR - Significant improvements needed"
    else:
        verdict = "❌ NEEDS WORK - Major issues detected"
    
    print(f"\n{verdict}")
    print(f"Score: {result.overall_score}%")
    print(f"Cost Efficiency: ${result.total_cost_usd:.4f} for {result.total_tests} tests")
    
    # Value proposition
    print("\n📊 VALUE PROPOSITION:")
    print(f"   • Orchestration overhead: {result.avg_latency_ms:.0f}ms avg")
    print(f"   • Multi-agent coordination: {'✅ Working' if result.passed_tests > result.total_tests * 0.7 else '⚠️ Needs tuning'}")
    print(f"   • Memory retention: {'✅ Working' if any(r.passed for r in result.results_by_category.get('multi_turn', [])) else '⚠️ Check memory system'}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="NIS Protocol Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmarks(
        quick=args.quick,
        verbose=not args.quiet
    ))


if __name__ == "__main__":
    main()
