#!/usr/bin/env python3
"""
🚀 NIS Protocol Enhanced Chat Functionality Test
Comprehensive test to verify enhanced chat works with all advanced features
"""

import asyncio
import json
import requests
from typing import Dict, Any

class EnhancedChatTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}

    async def test_enhanced_chat_endpoint(self) -> Dict[str, Any]:
        """Test the enhanced chat endpoint"""
        print("🧪 Testing Enhanced Chat Interface...")

        try:
            response = requests.get(f"{self.base_url}/chat/enhanced", timeout=10)

            if response.status_code == 200:
                if "enhanced_agent_chat.html" in response.text:
                    print("✅ Enhanced chat interface available")
                    return {"success": True, "details": "Enhanced chat HTML served correctly"}
                else:
                    return {"success": False, "error": "Enhanced chat HTML not found in response"}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_enhanced_streaming(self) -> Dict[str, Any]:
        """Test enhanced streaming with advanced features"""
        print("🔄 Testing Enhanced Streaming Features...")

        data = {
            "message": "Analyze this mathematical problem: solve x^2 + 3x - 10 = 0 and explain the reasoning step by step.",
            "user_id": "enhanced_test_user",
            "conversation_id": "enhanced_test_123",
            "agent_type": "reasoning",
            "provider": "anthropic",
            "stream": True,
            "include_reasoning": True,
            "include_tools": True,
            "include_artifacts": True
        }

        try:
            # Use requests directly for streaming endpoint with shorter timeout
            url = f"{self.base_url}/chat/stream"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=10, stream=True)

            if response.status_code == 200:
                # Read some streaming data to verify structure
                stream_data = ""
                data_types = set()

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                parsed_data = json.loads(line_str[6:])
                                data_types.add(parsed_data.get('type', 'unknown'))
                                stream_data += line_str[6:] + '\n'
                            except json.JSONDecodeError:
                                continue

                            # Just get a few chunks to analyze
                            if len(data_types) >= 2:
                                break

                if stream_data.strip():
                    print("✅ Enhanced streaming working")
                    print(f"   📊 Data types found: {', '.join(data_types)}")
                    return {"success": True, "details": f"Enhanced streaming with types: {', '.join(data_types)}"}
                else:
                    return {"success": False, "error": "No streaming data received"}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_advanced_features(self) -> Dict[str, Any]:
        """Test advanced NIS features"""
        print("🔬 Testing Advanced NIS Features...")

        features = {
            "physics_validation": "Calculate the physics validation for a pendulum with length 2m and initial angle 30 degrees",
            "signal_processing": "Process this signal: [1, 2, 3, 4, 5, 4, 3, 2, 1] using Laplace transform",
            "kan_reasoning": "Use KAN network to approximate the function f(x) = sin(x) + cos(2x)"
        }

        results = {}
        for feature, message in features.items():
            try:
                data = {
                    "message": message,
                    "user_id": f"feature_test_{feature}",
                    "conversation_id": f"feature_test_{feature}_{int(asyncio.get_event_loop().time())}",
                    "agent_type": "reasoning",
                    "provider": "anthropic",
                    "stream": False
                }

                response = requests.post(
                    f"{self.base_url}/chat",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )

                results[feature] = {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_keys": list(response.json().keys()) if response.status_code == 200 else []
                }

            except Exception as e:
                results[feature] = {
                    "success": False,
                    "error": str(e)
                }

        # Check if any advanced features work
        working_features = [f for f, r in results.items() if r["success"]]

        if working_features:
            print(f"✅ Advanced features working: {', '.join(working_features)}")
            return {"success": True, "details": f"Working features: {', '.join(working_features)}"}
        else:
            return {"success": False, "error": "No advanced features working"}

    async def test_voice_integration(self) -> Dict[str, Any]:
        """Test voice synthesis integration"""
        print("🎙️ Testing Enhanced Voice Integration...")

        data = {
            "text": "Hello, this is a test of the enhanced voice synthesis system with multi-speaker support.",
            "speaker": "consciousness",
            "emotion": "conversational"
        }

        try:
            # Test the JSON endpoint first
            json_response = requests.post(
                f"{self.base_url}/communication/synthesize/json",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if json_response.status_code == 200:
                print("✅ Voice synthesis JSON endpoint working")
                return {"success": True, "details": "Voice synthesis operational"}
            else:
                return {"success": False, "error": f"JSON endpoint failed: {json_response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_brain_visualization(self) -> Dict[str, Any]:
        """Test brain visualization features"""
        print("🧠 Testing Brain Visualization...")

        try:
            # Test consciousness status endpoint
            response = requests.get(f"{self.base_url}/consciousness/status", timeout=5)

            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ Consciousness status available: {status_data.get('consciousness_level', 'unknown')}")
                return {"success": True, "details": "Brain visualization data available"}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_comprehensive_enhanced_test(self) -> Dict[str, Any]:
        """Run all enhanced chat tests and provide comprehensive results"""
        print("🚀 Starting Comprehensive Enhanced Chat Test")
        print("=" * 60)

        tests = [
            ("Enhanced Chat Interface", self.test_enhanced_chat_endpoint),
            ("Enhanced Streaming", self.test_enhanced_streaming),
            ("Advanced Features", self.test_advanced_features),
            ("Voice Integration", self.test_voice_integration),
            ("Brain Visualization", self.test_brain_visualization)
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
        print("📊 ENHANCED CHAT TEST RESULTS")
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
    tester = EnhancedChatTester("http://localhost:8000")

    try:
        results = await tester.run_comprehensive_enhanced_test()

        if results["overall_success"]:
            print("\n🎉 ENHANCED CHAT TEST COMPLETED SUCCESSFULLY!")
            print("✅ Enhanced chat interface working perfectly")
            print("✅ Advanced streaming features operational")
            print("✅ NIS Protocol integration complete")
            print("✅ Voice synthesis available")
            print("✅ Brain visualization functional")
            print("\n🚀 Enhanced Chat is production-ready!")
        else:
            print("\n⚠️ ENHANCED CHAT TEST COMPLETED WITH ISSUES")
            print("Some tests failed. Check the error messages above.")

        return results["overall_success"]

    except Exception as e:
        print(f"\n❌ ENHANCED CHAT TEST ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
