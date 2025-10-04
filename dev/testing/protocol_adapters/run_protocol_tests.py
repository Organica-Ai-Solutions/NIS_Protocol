"""
Protocol Adapter Test Runner

Run all protocol adapter tests and generate a coverage report.

Usage:
    python run_protocol_tests.py              # Run all tests
    python run_protocol_tests.py --verbose    # Verbose output
    python run_protocol_tests.py --coverage   # With coverage report
"""

import sys
import pytest
import argparse
from pathlib import Path


def run_tests(verbose=False, coverage=False):
    """Run all protocol adapter tests"""
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Build pytest arguments
    args = [
        str(test_dir),
        "-v" if verbose else "",
        "--tb=short",
    ]
    
    # Add coverage if requested
    if coverage:
        args.extend([
            "--cov=src/adapters",
            "--cov-report=html",
            "--cov-report=term"
        ])
    
    # Filter empty strings
    args = [arg for arg in args if arg]
    
    print("=" * 70)
    print("NIS Protocol - Third-Party Protocol Adapter Tests")
    print("=" * 70)
    print(f"\nRunning tests in: {test_dir}")
    print(f"Verbose: {verbose}")
    print(f"Coverage: {coverage}")
    print("=" * 70 + "\n")
    
    # Run pytest
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")
    
    if coverage:
        print("Coverage report generated in: htmlcov/index.html\n")
    
    return exit_code


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run NIS Protocol adapter tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

