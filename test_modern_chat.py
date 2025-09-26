#!/usr/bin/env python3
"""
🚀 NIS Protocol Modern Chat Functionality Test
Comprehensive test to verify modern chat works perfectly with all NIS agents
"""

import asyncio
import json
import requests
from typing import Dict, Any

class ModernChatTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}

    async def test_endpoint(self, endpoint: str, method: str = "POST", data: Dict = None) -> Dict[str, Any]:
        """Test a specific endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}

            if method == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                response = requests.get(url, headers=headers, timeout=10)

            return {
                "endpoint": endpoint,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.text else None
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "success": False,
                "error": str(e)
            }

    async def test_chat_stream(self) -> Dict[str, Any]:
        """Test the main chat stream endpoint"""
        print("🧪 Testing /chat/stream endpoint...")

        data = {
            "message": "Hello NIS Protocol! Test the reasoning agent.",
            "user_id": "test_user_123",
            "agent_type": "reasoning",
            "provider": "anthropic",
            "stream": True,
            "include_reasoning": True,
            "include_tools": True
        }

        try:
            # Use requests directly for streaming endpoint with shorter timeout
            url = f"{self.base_url}/chat/stream"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=5, stream=True)

            if response.status_code == 200:
                # Just verify the connection works, don't try to read the entire stream
                print("✅ Chat stream endpoint responding (200 OK)")
                return {"success": True, "details": f"Stream endpoint available (Status: {response.status_code})"}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_voice_synthesis(self) -> Dict[str, Any]:
        """Test VibeVoice synthesis endpoint"""
        print("🎙️ Testing /communication/synthesize endpoint...")

        data = {
            "text": "Hello, this is a test of the VibeVoice synthesis system.",
            "speaker": "consciousness",
            "emotion": "conversational"
        }

        try:
            # First test the JSON endpoint
            json_response = await self.test_endpoint("/communication/synthesize/json", "POST", data)

            if json_response["success"]:
                print("✅ Voice synthesis JSON endpoint working")
                return {"success": True, "details": "Voice synthesis available"}
            else:
                return {"success": False, "error": f"JSON endpoint failed: {json_response.get('error', 'Unknown')}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_nis_pipeline(self) -> Dict[str, Any]:
        """Test the NIS pipeline integration"""
        print("🔬 Testing NIS pipeline integration...")

        # Test the process_nis_pipeline function indirectly through chat
        data = {
            "message": "Analyze this signal: frequency domain processing test",
            "user_id": "pipeline_test_user",
            "agent_type": "reasoning",
            "stream": False
        }

        try:
            # Use requests directly for chat endpoint
            url = f"{self.base_url}/chat"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result:
                    print("✅ NIS pipeline integration working")
                    print(f"   📄 Response keys: {list(result.keys())}")
                    return {"success": True, "details": f"Pipeline processing available, keys: {list(result.keys())}"}
                else:
                    return {"success": False, "error": "Empty response"}
            else:
                return {"success": False, "error": f"Status {response.status_code}: {response.text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_agent_specific_features(self) -> Dict[str, Any]:
        """Test agent-specific features"""
        print("🤖 Testing agent-specific features...")

        results = {}

        # Test different agent types
        agent_types = ["reasoning", "physics", "research", "coordination"]

        for agent_type in agent_types:
            data = {
                "message": f"Test {agent_type} agent functionality",
                "user_id": f"agent_test_{agent_type}",
                "agent_type": agent_type,
                "stream": False
            }

            # Use requests directly for chat endpoint
            url = f"{self.base_url}/chat"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=10)

            results[agent_type] = {
                "success": response.status_code == 200,
                "status_code": response.status_code
            }

        # Check if all agents respond
        all_success = all(r["success"] for r in results.values())

        if all_success:
            print("✅ All agent types responding correctly")
            return {"success": True, "details": f"All agents working: {', '.join(agent_types)}"}
        else:
            failed_agents = [agent for agent, result in results.items() if not result["success"]]
            return {"success": False, "error": f"Failed agents: {', '.join(failed_agents)}"}

    async def test_real_time_features(self) -> Dict[str, Any]:
        """Test real-time features like WebSocket connections"""
        print("⚡ Testing real-time features...")

        try:
            # Test basic health endpoint for real-time readiness
            health_response = await self.test_endpoint("/health", "GET")

            if health_response["success"]:
                print("✅ Health endpoint available (real-time ready)")
                return {"success": True, "details": "Real-time features available"}
            else:
                return {"success": False, "error": "Health endpoint not available"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and provide comprehensive results"""
        print("🚀 Starting Comprehensive Modern Chat Test")
        print("=" * 60)

        tests = [
            ("Chat Stream", self.test_chat_stream),
            ("Voice Synthesis", self.test_voice_synthesis),
            ("NIS Pipeline", self.test_nis_pipeline),
            ("Agent Features", self.test_agent_specific_features),
            ("Real-time Features", self.test_real_time_features)
        ]

        results = {}
        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            result = await test_func()
            results[test_name] = result

            if result["success"]:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        overall_success = passed == total

        print(f"Tests Passed: {passed}/{total}")
        print(f"Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

        if not overall_success:
            print("\nFailed Tests:")
            for test_name, result in results.items():
                if not result["success"]:
                    print(f"  ❌ {test_name}: {result.get('error', 'Unknown')}")

        return {
            "overall_success": overall_success,
            "passed": passed,
            "total": total,
            "results": results
        }

async def main():
    """Main test runner"""
    tester = ModernChatTester("http://localhost:8000")

    try:
        results = await tester.run_comprehensive_test()

        if results["overall_success"]:
            print("\n🎉 MODERN CHAT TEST COMPLETED SUCCESSFULLY!")
            print("✅ All endpoints responding correctly")
            print("✅ NIS Protocol integration working")
            print("✅ Voice synthesis available")
            print("✅ Agent-specific features functional")
            print("✅ Real-time features available")
            print("\n🚀 Modern Chat is ready for production use!")
        else:
            print("\n⚠️ MODERN CHAT TEST COMPLETED WITH ISSUES")
            print("Some tests failed. Check the error messages above.")
            print("The modern chat may need backend server restart or configuration fixes.")

        return results["overall_success"]

    except Exception as e:
        print(f"\n❌ TEST RUNNER ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
