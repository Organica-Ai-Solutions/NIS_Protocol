#!/usr/bin/env python3
"""
Comprehensive API Testing Script for NIS Protocol v3.2
Tests all endpoints systematically and reports results

Enhanced with NVIDIA NeMo Enterprise Integration testing
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import sys

BASE_URL = "http://localhost:8000"

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    response_data: Any = None
    error: str = None

class ComprehensiveAPITester:
    """Comprehensive API testing for all NIS Protocol endpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None,
        expected_status: int = 200,
        timeout: int = 30
    ) -> TestResult:
        """Test a single endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data if data else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                success = (response.status == expected_status or 
                          (response.status >= 200 and response.status < 300))
                
                return TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=success,
                    response_data=response_data
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def test_core_endpoints(self):
        """Test core system endpoints"""
        
        print("🏥 Testing Core System Endpoints...")
        
        core_tests = [
            ("GET", "/health"),
            ("GET", "/"),
            ("GET", "/status"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ]
        
        for method, endpoint in core_tests:
            result = await self.test_endpoint(method, endpoint)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_physics_endpoints(self):
        """Test physics validation endpoints"""
        
        print("\n🔬 Testing Physics Validation Endpoints...")
        
        physics_tests = [
            ("GET", "/physics/capabilities"),
            ("POST", "/physics/validate", {
                "scenario": "A ball is dropped from 10 meters",
                "expected_outcome": "Ball accelerates at 9.81 m/s²"
            }),
            ("POST", "/physics/pinn/solve", {
                "equation_type": "heat_equation",
                "boundary_conditions": {"x0": 0, "x1": 1, "t0": 0}
            }),
            ("GET", "/physics/constants"),
        ]
        
        for method, endpoint, *data in physics_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_nvidia_nemo_endpoints(self):
        """Test NVIDIA NeMo Enterprise Integration endpoints"""
        
        print("\n🚀 Testing NVIDIA NeMo Enterprise Endpoints...")
        
        nemo_tests = [
            ("GET", "/nvidia/nemo/status"),
            ("GET", "/nvidia/nemo/enterprise/showcase"),
            ("GET", "/nvidia/nemo/cosmos/demo"),
            ("GET", "/nvidia/nemo/toolkit/status"),
            ("POST", "/nvidia/nemo/physics/simulate", {
                "scenario_description": "Simulate a pendulum swinging in air",
                "simulation_type": "classical_mechanics",
                "precision": "high"
            }),
            ("POST", "/nvidia/nemo/orchestrate", {
                "workflow_name": "test_coordination",
                "input_data": {"query": "Test agent coordination"},
                "agent_types": ["physics", "research"]
            }),
            ("POST", "/nvidia/nemo/toolkit/test", {
                "test_query": "What is NVIDIA NeMo Agent Toolkit?"
            }),
        ]
        
        for method, endpoint, *data in nemo_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_research_endpoints(self):
        """Test research and deep agent endpoints"""
        
        print("\n🔍 Testing Research & Deep Agent Endpoints...")
        
        research_tests = [
            ("POST", "/research/deep", {
                "query": "Latest developments in transformer architecture",
                "research_depth": "comprehensive"
            }),
            ("POST", "/research/arxiv", {
                "query": "neural networks",
                "max_papers": 5
            }),
            ("POST", "/research/analyze", {
                "content": "This is a test research paper about AI.",
                "analysis_type": "comprehensive"
            }),
            ("GET", "/research/capabilities"),
        ]
        
        for method, endpoint, *data in research_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_agent_endpoints(self):
        """Test agent coordination endpoints"""
        
        print("\n🤖 Testing Agent Coordination Endpoints...")
        
        agent_tests = [
            ("GET", "/agents/status"),
            ("POST", "/agents/consciousness/analyze", {
                "scenario": "Test consciousness analysis",
                "depth": "standard"
            }),
            ("POST", "/agents/memory/store", {
                "content": "Test memory storage",
                "memory_type": "episodic"
            }),
            ("POST", "/agents/planning/create", {
                "goal": "Test autonomous planning",
                "constraints": ["time", "resources"]
            }),
            ("GET", "/agents/capabilities"),
        ]
        
        for method, endpoint, *data in agent_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_mcp_endpoints(self):
        """Test Model Context Protocol endpoints"""
        
        print("\n🔌 Testing MCP Integration Endpoints...")
        
        mcp_tests = [
            ("GET", "/api/mcp/demo"),
            ("GET", "/api/langgraph/status"),
            ("POST", "/api/langgraph/invoke", {
                "messages": [{"role": "user", "content": "Hello"}],
                "session_id": "test_session"
            }),
        ]
        
        for method, endpoint, *data in mcp_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_chat_endpoints(self):
        """Test chat and interaction endpoints"""
        
        print("\n💬 Testing Chat & Interaction Endpoints...")
        
        chat_tests = [
            ("POST", "/chat", {
                "message": "Hello, how are you?",
                "session_id": "test_session"
            }),
            ("POST", "/chat/enhanced", {
                "message": "Test enhanced chat",
                "enable_memory": True,
                "session_id": "test_session"
            }),
            ("GET", "/chat/sessions"),
            ("GET", "/chat/memory/test_session"),
        ]
        
        for method, endpoint, *data in chat_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def run_comprehensive_test(self):
        """Run all endpoint tests"""
        
        print("🧪 Starting Comprehensive API Testing for NIS Protocol v3.2")
        print("=" * 70)
        
        # Test core endpoints first
        await self.test_core_endpoints()
        
        # Test feature endpoints
        await self.test_physics_endpoints()
        await self.test_nvidia_nemo_endpoints()
        await self.test_research_endpoints()
        await self.test_agent_endpoints()
        await self.test_mcp_endpoints()
        await self.test_chat_endpoints()
        
        # Generate summary report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        
        print(f"Total Endpoints Tested: {total_tests}")
        print(f"Successful: {successful_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        
        # Failed endpoints
        if failed_tests > 0:
            print(f"\n❌ FAILED ENDPOINTS ({failed_tests}):")
            for result in self.results:
                if not result.success:
                    error_msg = result.error or f"HTTP {result.status_code}"
                    print(f"  • {result.method} {result.endpoint} - {error_msg}")
        
        # Slowest endpoints
        print(f"\n⏱️ SLOWEST ENDPOINTS (Top 5):")
        slowest = sorted(self.results, key=lambda x: x.response_time, reverse=True)[:5]
        for result in slowest:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.method} {result.endpoint} - {result.response_time:.3f}s")
        
        # Enterprise features status
        nemo_results = [r for r in self.results if "/nvidia/nemo/" in r.endpoint]
        nemo_success = sum(1 for r in nemo_results if r.success)
        nemo_total = len(nemo_results)
        
        print(f"\n🚀 NVIDIA NeMo Enterprise Integration:")
        print(f"  Endpoints Tested: {nemo_total}")
        print(f"  Successful: {nemo_success}/{nemo_total}")
        print(f"  Integration Status: {'🟢 Ready' if nemo_success == nemo_total else '🟡 Partial' if nemo_success > 0 else '🔴 Issues'}")
        
        # System health summary
        health_result = next((r for r in self.results if r.endpoint == "/health"), None)
        if health_result and health_result.success:
            print(f"\n💚 System Health: OK ({health_result.response_time:.3f}s)")
        else:
            print(f"\n💔 System Health: Issues Detected")
        
        print("\n" + "=" * 70)
        print("🎯 Testing Complete! Check individual endpoint results above.")
        print("=" * 70)


async def main():
    """Main testing function"""
    
    print("Waiting for backend to be ready...")
    await asyncio.sleep(5)  # Give backend time to start
    
    async with ComprehensiveAPITester() as tester:
        await tester.run_comprehensive_test()


if __name__ == "__main__":
    # Check if backend is ready first
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        sys.exit(1)

"""
Comprehensive API Testing Script for NIS Protocol v3.2
Tests all endpoints systematically and reports results

Enhanced with NVIDIA NeMo Enterprise Integration testing
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import sys

BASE_URL = "http://localhost:8000"

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    response_data: Any = None
    error: str = None

class ComprehensiveAPITester:
    """Comprehensive API testing for all NIS Protocol endpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None,
        expected_status: int = 200,
        timeout: int = 30
    ) -> TestResult:
        """Test a single endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data if data else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                success = (response.status == expected_status or 
                          (response.status >= 200 and response.status < 300))
                
                return TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=success,
                    response_data=response_data
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def test_core_endpoints(self):
        """Test core system endpoints"""
        
        print("🏥 Testing Core System Endpoints...")
        
        core_tests = [
            ("GET", "/health"),
            ("GET", "/"),
            ("GET", "/status"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ]
        
        for method, endpoint in core_tests:
            result = await self.test_endpoint(method, endpoint)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_physics_endpoints(self):
        """Test physics validation endpoints"""
        
        print("\n🔬 Testing Physics Validation Endpoints...")
        
        physics_tests = [
            ("GET", "/physics/capabilities"),
            ("POST", "/physics/validate", {
                "scenario": "A ball is dropped from 10 meters",
                "expected_outcome": "Ball accelerates at 9.81 m/s²"
            }),
            ("POST", "/physics/pinn/solve", {
                "equation_type": "heat_equation",
                "boundary_conditions": {"x0": 0, "x1": 1, "t0": 0}
            }),
            ("GET", "/physics/constants"),
        ]
        
        for method, endpoint, *data in physics_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_nvidia_nemo_endpoints(self):
        """Test NVIDIA NeMo Enterprise Integration endpoints"""
        
        print("\n🚀 Testing NVIDIA NeMo Enterprise Endpoints...")
        
        nemo_tests = [
            ("GET", "/nvidia/nemo/status"),
            ("GET", "/nvidia/nemo/enterprise/showcase"),
            ("GET", "/nvidia/nemo/cosmos/demo"),
            ("GET", "/nvidia/nemo/toolkit/status"),
            ("POST", "/nvidia/nemo/physics/simulate", {
                "scenario_description": "Simulate a pendulum swinging in air",
                "simulation_type": "classical_mechanics",
                "precision": "high"
            }),
            ("POST", "/nvidia/nemo/orchestrate", {
                "workflow_name": "test_coordination",
                "input_data": {"query": "Test agent coordination"},
                "agent_types": ["physics", "research"]
            }),
            ("POST", "/nvidia/nemo/toolkit/test", {
                "test_query": "What is NVIDIA NeMo Agent Toolkit?"
            }),
        ]
        
        for method, endpoint, *data in nemo_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_research_endpoints(self):
        """Test research and deep agent endpoints"""
        
        print("\n🔍 Testing Research & Deep Agent Endpoints...")
        
        research_tests = [
            ("POST", "/research/deep", {
                "query": "Latest developments in transformer architecture",
                "research_depth": "comprehensive"
            }),
            ("POST", "/research/arxiv", {
                "query": "neural networks",
                "max_papers": 5
            }),
            ("POST", "/research/analyze", {
                "content": "This is a test research paper about AI.",
                "analysis_type": "comprehensive"
            }),
            ("GET", "/research/capabilities"),
        ]
        
        for method, endpoint, *data in research_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_agent_endpoints(self):
        """Test agent coordination endpoints"""
        
        print("\n🤖 Testing Agent Coordination Endpoints...")
        
        agent_tests = [
            ("GET", "/agents/status"),
            ("POST", "/agents/consciousness/analyze", {
                "scenario": "Test consciousness analysis",
                "depth": "standard"
            }),
            ("POST", "/agents/memory/store", {
                "content": "Test memory storage",
                "memory_type": "episodic"
            }),
            ("POST", "/agents/planning/create", {
                "goal": "Test autonomous planning",
                "constraints": ["time", "resources"]
            }),
            ("GET", "/agents/capabilities"),
        ]
        
        for method, endpoint, *data in agent_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_mcp_endpoints(self):
        """Test Model Context Protocol endpoints"""
        
        print("\n🔌 Testing MCP Integration Endpoints...")
        
        mcp_tests = [
            ("GET", "/api/mcp/demo"),
            ("GET", "/api/langgraph/status"),
            ("POST", "/api/langgraph/invoke", {
                "messages": [{"role": "user", "content": "Hello"}],
                "session_id": "test_session"
            }),
        ]
        
        for method, endpoint, *data in mcp_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_chat_endpoints(self):
        """Test chat and interaction endpoints"""
        
        print("\n💬 Testing Chat & Interaction Endpoints...")
        
        chat_tests = [
            ("POST", "/chat", {
                "message": "Hello, how are you?",
                "session_id": "test_session"
            }),
            ("POST", "/chat/enhanced", {
                "message": "Test enhanced chat",
                "enable_memory": True,
                "session_id": "test_session"
            }),
            ("GET", "/chat/sessions"),
            ("GET", "/chat/memory/test_session"),
        ]
        
        for method, endpoint, *data in chat_tests:
            payload = data[0] if data else None
            result = await self.test_endpoint(method, endpoint, payload)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def run_comprehensive_test(self):
        """Run all endpoint tests"""
        
        print("🧪 Starting Comprehensive API Testing for NIS Protocol v3.2")
        print("=" * 70)
        
        # Test core endpoints first
        await self.test_core_endpoints()
        
        # Test feature endpoints
        await self.test_physics_endpoints()
        await self.test_nvidia_nemo_endpoints()
        await self.test_research_endpoints()
        await self.test_agent_endpoints()
        await self.test_mcp_endpoints()
        await self.test_chat_endpoints()
        
        # Generate summary report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        
        print(f"Total Endpoints Tested: {total_tests}")
        print(f"Successful: {successful_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        
        # Failed endpoints
        if failed_tests > 0:
            print(f"\n❌ FAILED ENDPOINTS ({failed_tests}):")
            for result in self.results:
                if not result.success:
                    error_msg = result.error or f"HTTP {result.status_code}"
                    print(f"  • {result.method} {result.endpoint} - {error_msg}")
        
        # Slowest endpoints
        print(f"\n⏱️ SLOWEST ENDPOINTS (Top 5):")
        slowest = sorted(self.results, key=lambda x: x.response_time, reverse=True)[:5]
        for result in slowest:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.method} {result.endpoint} - {result.response_time:.3f}s")
        
        # Enterprise features status
        nemo_results = [r for r in self.results if "/nvidia/nemo/" in r.endpoint]
        nemo_success = sum(1 for r in nemo_results if r.success)
        nemo_total = len(nemo_results)
        
        print(f"\n🚀 NVIDIA NeMo Enterprise Integration:")
        print(f"  Endpoints Tested: {nemo_total}")
        print(f"  Successful: {nemo_success}/{nemo_total}")
        print(f"  Integration Status: {'🟢 Ready' if nemo_success == nemo_total else '🟡 Partial' if nemo_success > 0 else '🔴 Issues'}")
        
        # System health summary
        health_result = next((r for r in self.results if r.endpoint == "/health"), None)
        if health_result and health_result.success:
            print(f"\n💚 System Health: OK ({health_result.response_time:.3f}s)")
        else:
            print(f"\n💔 System Health: Issues Detected")
        
        print("\n" + "=" * 70)
        print("🎯 Testing Complete! Check individual endpoint results above.")
        print("=" * 70)


async def main():
    """Main testing function"""
    
    print("Waiting for backend to be ready...")
    await asyncio.sleep(5)  # Give backend time to start
    
    async with ComprehensiveAPITester() as tester:
        await tester.run_comprehensive_test()


if __name__ == "__main__":
    # Check if backend is ready first
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        sys.exit(1)
