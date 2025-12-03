#!/usr/bin/env python3
"""
NIS Protocol Simple Load Test
Runs without Locust for quick performance validation

Usage:
    python tests/load/run_load_test.py --users 10 --duration 30
"""

import argparse
import asyncio
import time
import statistics
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import httpx
except ImportError:
    import requests as httpx

BASE_URL = os.getenv("NIS_API_URL", "http://localhost:8000")


@dataclass
class RequestResult:
    """Result of a single request"""
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    success: bool
    error: str = None


@dataclass
class LoadTestStats:
    """Statistics for load test"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0
    response_times: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)
    endpoint_stats: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_result(self, result: RequestResult):
        """Add a request result"""
        self.total_requests += 1
        self.total_duration += result.duration_ms
        self.response_times.append(result.duration_ms)
        
        if result.endpoint not in self.endpoint_stats:
            self.endpoint_stats[result.endpoint] = []
        self.endpoint_stats[result.endpoint].append(result.duration_ms)
        
        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            error_key = f"{result.status_code}: {result.error or 'Unknown'}"
            self.errors[error_key] = self.errors.get(error_key, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.response_times:
            return {"error": "No requests completed"}
        
        sorted_times = sorted(self.response_times)
        
        return {
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "success_rate": f"{(self.successful_requests / self.total_requests) * 100:.1f}%",
            "response_times": {
                "min_ms": round(min(self.response_times), 2),
                "max_ms": round(max(self.response_times), 2),
                "avg_ms": round(statistics.mean(self.response_times), 2),
                "median_ms": round(statistics.median(self.response_times), 2),
                "p95_ms": round(sorted_times[int(len(sorted_times) * 0.95)], 2),
                "p99_ms": round(sorted_times[int(len(sorted_times) * 0.99)], 2) if len(sorted_times) > 100 else None
            },
            "throughput": {
                "requests_per_second": round(self.total_requests / (self.total_duration / 1000), 2) if self.total_duration > 0 else 0
            },
            "errors": self.errors
        }


# Test endpoints with weights
ENDPOINTS = [
    # (method, path, weight, payload)
    ("GET", "/health", 10, None),
    ("GET", "/system/status", 5, None),
    ("GET", "/infrastructure/status", 3, None),
    ("GET", "/robotics/capabilities", 4, None),
    ("POST", "/robotics/forward_kinematics", 3, {
        "robot_id": "test_arm",
        "joint_angles": [0.0, 0.5, 1.0, 0.0, 0.5, 0.0],
        "robot_type": "manipulator"
    }),
    ("POST", "/robotics/inverse_kinematics", 2, {
        "robot_id": "test_arm",
        "target_pose": {"position": [0.5, 0.3, 0.8]},
        "robot_type": "manipulator"
    }),
    ("GET", "/robotics/can/status", 2, None),
    ("GET", "/robotics/obd/status", 2, None),
    ("GET", "/physics/capabilities", 3, None),
    ("POST", "/physics/validate", 2, {
        "physics_data": {"velocity": [1.0, 2.0, 3.0], "mass": 10.0},
        "domain": "MECHANICS"
    }),
    ("GET", "/observability/status", 1, None),
    ("GET", "/security/status", 1, None),
    ("GET", "/v4/consciousness/status", 2, None),
]


def make_request(endpoint_info) -> RequestResult:
    """Make a single request"""
    method, path, _, payload = endpoint_info
    url = f"{BASE_URL}{path}"
    
    start = time.time()
    try:
        if hasattr(httpx, 'Client'):
            # httpx
            with httpx.Client(timeout=30) as client:
                if method == "GET":
                    response = client.get(url)
                else:
                    response = client.post(url, json=payload)
        else:
            # requests fallback
            if method == "GET":
                response = httpx.get(url, timeout=30)
            else:
                response = httpx.post(url, json=payload, timeout=30)
        
        duration = (time.time() - start) * 1000
        
        return RequestResult(
            endpoint=path,
            method=method,
            status_code=response.status_code,
            duration_ms=duration,
            success=response.status_code == 200
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return RequestResult(
            endpoint=path,
            method=method,
            status_code=0,
            duration_ms=duration,
            success=False,
            error=str(e)
        )


def weighted_random_endpoint():
    """Select a random endpoint based on weights"""
    import random
    total_weight = sum(e[2] for e in ENDPOINTS)
    r = random.uniform(0, total_weight)
    
    cumulative = 0
    for endpoint in ENDPOINTS:
        cumulative += endpoint[2]
        if r <= cumulative:
            return endpoint
    
    return ENDPOINTS[0]


def user_worker(user_id: int, duration: float, stats: LoadTestStats, lock: threading.Lock):
    """Simulate a single user making requests"""
    end_time = time.time() + duration
    request_count = 0
    
    while time.time() < end_time:
        endpoint = weighted_random_endpoint()
        result = make_request(endpoint)
        
        with lock:
            stats.add_result(result)
        
        request_count += 1
        
        # Small delay between requests (0.1-0.3s)
        time.sleep(0.1 + (0.2 * (hash(str(user_id) + str(request_count)) % 100) / 100))
    
    return request_count


def run_load_test(num_users: int, duration: float) -> LoadTestStats:
    """Run load test with specified users and duration"""
    print(f"\n{'='*60}")
    print(f"🚀 NIS Protocol Load Test")
    print(f"{'='*60}")
    print(f"Target: {BASE_URL}")
    print(f"Users: {num_users}")
    print(f"Duration: {duration}s")
    print(f"{'='*60}\n")
    
    stats = LoadTestStats()
    lock = threading.Lock()
    
    print("Starting load test...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(user_worker, i, duration, stats, lock)
            for i in range(num_users)
        ]
        
        # Wait for all users to complete
        for future in futures:
            future.result()
    
    elapsed = time.time() - start_time
    
    print(f"\nLoad test completed in {elapsed:.2f}s")
    
    return stats


def print_results(stats: LoadTestStats):
    """Print load test results"""
    summary = stats.get_summary()
    
    print(f"\n{'='*60}")
    print(f"📊 Load Test Results")
    print(f"{'='*60}")
    
    print(f"\n📈 Request Summary:")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success Rate: {summary['success_rate']}")
    
    print(f"\n⏱️ Response Times:")
    rt = summary['response_times']
    print(f"  Min: {rt['min_ms']}ms")
    print(f"  Max: {rt['max_ms']}ms")
    print(f"  Avg: {rt['avg_ms']}ms")
    print(f"  Median: {rt['median_ms']}ms")
    print(f"  P95: {rt['p95_ms']}ms")
    if rt['p99_ms']:
        print(f"  P99: {rt['p99_ms']}ms")
    
    print(f"\n🚀 Throughput:")
    print(f"  Requests/second: {summary['throughput']['requests_per_second']}")
    
    if summary['errors']:
        print(f"\n❌ Errors:")
        for error, count in summary['errors'].items():
            print(f"  {error}: {count}")
    
    # Per-endpoint stats
    print(f"\n📋 Per-Endpoint Performance:")
    print(f"  {'Endpoint':<40} {'Requests':>10} {'Avg (ms)':>10} {'P95 (ms)':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    
    for endpoint, times in sorted(stats.endpoint_stats.items()):
        sorted_times = sorted(times)
        avg = statistics.mean(times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0]
        print(f"  {endpoint:<40} {len(times):>10} {avg:>10.2f} {p95:>10.2f}")
    
    # Performance rating
    print(f"\n{'='*60}")
    avg_ms = summary['response_times']['avg_ms']
    success_rate = float(summary['success_rate'].rstrip('%'))
    
    if avg_ms < 50 and success_rate > 99:
        print("🏆 EXCELLENT - Production ready!")
    elif avg_ms < 100 and success_rate > 95:
        print("✅ GOOD - Acceptable performance")
    elif avg_ms < 200 and success_rate > 90:
        print("⚠️ ACCEPTABLE - Consider optimization")
    else:
        print("❌ NEEDS IMPROVEMENT - Performance issues detected")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="NIS Protocol Load Test")
    parser.add_argument("--users", "-u", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", "-d", type=float, default=30, help="Test duration in seconds")
    parser.add_argument("--url", type=str, default=None, help="Target URL")
    
    args = parser.parse_args()
    
    global BASE_URL
    if args.url:
        BASE_URL = args.url
    
    # Verify system is up
    print("Checking system availability...")
    try:
        result = make_request(("GET", "/health", 1, None))
        if not result.success:
            print(f"❌ System not available: {result.error or result.status_code}")
            sys.exit(1)
        print(f"✅ System healthy (response: {result.duration_ms:.2f}ms)")
    except Exception as e:
        print(f"❌ Cannot connect to {BASE_URL}: {e}")
        sys.exit(1)
    
    # Run load test
    stats = run_load_test(args.users, args.duration)
    
    # Print results
    print_results(stats)
    
    # Return exit code based on success rate
    summary = stats.get_summary()
    success_rate = float(summary['success_rate'].rstrip('%'))
    sys.exit(0 if success_rate > 90 else 1)


if __name__ == "__main__":
    main()
