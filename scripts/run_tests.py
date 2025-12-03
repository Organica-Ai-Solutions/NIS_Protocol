#!/usr/bin/env python3
"""
NIS Protocol Test Runner
Comprehensive test execution with reporting
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list, description: str) -> tuple:
    """Run a command and return (success, output)"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print(f"\n⏱️  Completed in {elapsed:.2f}s")
    
    return result.returncode == 0, result


def run_unit_tests():
    """Run unit tests only"""
    return run_command(
        ["python", "-m", "pytest", "tests/unit", "-v", "-m", "unit", "--tb=short"],
        "Running Unit Tests"
    )


def run_integration_tests():
    """Run integration tests only"""
    return run_command(
        ["python", "-m", "pytest", "tests/integration", "-v", "-m", "integration", "--tb=short"],
        "Running Integration Tests"
    )


def run_all_tests():
    """Run all tests"""
    return run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Running All Tests"
    )


def run_fast_tests():
    """Run fast tests only (exclude slow)"""
    return run_command(
        ["python", "-m", "pytest", "tests/", "-v", "-m", "not slow", "--tb=short"],
        "Running Fast Tests (excluding slow)"
    )


def run_coverage():
    """Run tests with coverage report"""
    return run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"],
        "Running Tests with Coverage"
    )


def run_specific_test(test_path: str):
    """Run a specific test file or test"""
    return run_command(
        ["python", "-m", "pytest", test_path, "-v", "--tb=long"],
        f"Running Specific Test: {test_path}"
    )


def check_infrastructure():
    """Check if infrastructure is available"""
    print("\n🔍 Checking Infrastructure...")
    
    checks = []
    
    # Check Kafka
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 9092))
        checks.append(("Kafka", result == 0))
        sock.close()
    except:
        checks.append(("Kafka", False))
    
    # Check Redis
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 6379))
        checks.append(("Redis", result == 0))
        sock.close()
    except:
        checks.append(("Redis", False))
    
    # Check API
    try:
        import httpx
        response = httpx.get("http://localhost:8000/health", timeout=5)
        checks.append(("API", response.status_code == 200))
    except:
        checks.append(("API", False))
    
    print("\nInfrastructure Status:")
    all_ok = True
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}: {'Available' if status else 'Not Available'}")
        if not status:
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="NIS Protocol Test Runner")
    parser.add_argument(
        "mode",
        choices=["unit", "integration", "all", "fast", "coverage", "check"],
        nargs="?",
        default="fast",
        help="Test mode to run"
    )
    parser.add_argument(
        "--test", "-t",
        help="Run specific test file or test"
    )
    parser.add_argument(
        "--skip-infra-check",
        action="store_true",
        help="Skip infrastructure check"
    )
    
    args = parser.parse_args()
    
    print("🚀 NIS Protocol Test Runner")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    
    # Check infrastructure for integration tests
    if args.mode in ["integration", "all", "coverage"] and not args.skip_infra_check:
        if not check_infrastructure():
            print("\n⚠️  Some infrastructure is not available.")
            print("   Integration tests may fail.")
            response = input("   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
    
    # Run specific test if provided
    if args.test:
        success, _ = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    # Run tests based on mode
    if args.mode == "unit":
        success, _ = run_unit_tests()
    elif args.mode == "integration":
        success, _ = run_integration_tests()
    elif args.mode == "all":
        success, _ = run_all_tests()
    elif args.mode == "fast":
        success, _ = run_fast_tests()
    elif args.mode == "coverage":
        success, _ = run_coverage()
    elif args.mode == "check":
        success = check_infrastructure()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ Tests PASSED")
    else:
        print("❌ Tests FAILED")
    print("="*60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
