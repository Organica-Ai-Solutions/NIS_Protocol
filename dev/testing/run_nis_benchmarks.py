#!/usr/bin/env python3
"""
Quick execution script for NIS Protocol v3.0 Benchmark Suite
Run this to test KAN, PINN, and combined reasoning capabilities
"""

import sys
import os
import requests
import time

def check_nis_system():
    """Check if NIS Protocol v3.0 is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ NIS Protocol v3.0 is running!")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Components: {data.get('components', {})}")
            return True
        else:
            print(f"⚠️ NIS system responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to NIS Protocol v3.0: {e}")
        print("   Make sure the system is running: docker-compose up -d")
        return False

def main():
    """Run the benchmark suite"""
    print("🔬 NIS PROTOCOL v3.0 BENCHMARK RUNNER")
    print("=" * 50)
    
    # Check if system is running
    if not check_nis_system():
        print("\n🛑 Please start NIS Protocol v3.0 first:")
        print("   docker-compose up -d")
        print("   Then run this script again.")
        return False
    
    print("\n⏳ Starting comprehensive benchmark suite...")
    print("   This will test:")
    print("   🔬 KAN Math Reasoning (arithmetic, polynomials, trigonometry)")
    print("   🧪 PINN Physics Validation (conservation laws, motion)")
    print("   🌍 Combined Reasoning (physical explanation inference)")
    
    # Import and run the benchmark suite
    try:
        from benchmarks.nis_comprehensive_benchmark_suite import NISBenchmarkSuite
        
        benchmark_suite = NISBenchmarkSuite()
        results = benchmark_suite.run_all_benchmarks()
        
        print(f"\n🎯 Benchmark completed!")
        print(f"   Overall Score: {results['overall_score']:.3f}/1.000")
        print(f"   Execution Time: {results['total_time']:.1f}s")
        
        # Quick summary
        summary = results['results_summary']
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   🟡 Partial: {summary['partial']}")
        print(f"   ❌ Failed: {summary['failed']}")
        
        return True
        
    except ImportError:
        print("❌ Benchmark suite not found. Make sure the file exists:")
        print("   benchmarks/nis_comprehensive_benchmark_suite.py")
        return False
    except Exception as e:
        print(f"❌ Benchmark execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 