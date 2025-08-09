#!/usr/bin/env python3
"""
🧹 Console Cleanup Verification Test
Verifies that quantum consciousness, perfect memory, and superhuman test buttons were successfully removed
"""

import requests
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("console_cleanup_test")

class ConsoleCleanupTester:
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        
    def test_console_cleanup(self) -> bool:
        """Test that unwanted buttons were successfully removed"""
        try:
            logger.info("🧹 Testing console cleanup...")
            
            # Get the console HTML
            response = requests.get(f"{self.base_url}/console", timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Console not accessible: {response.status_code}")
                return False
            
            html_content = response.text
            
            # Check for removed buttons/content
            removed_content = [
                "Quantum Consciousness",
                "Perfect Memory", 
                "Superhuman Test",
                "quantum consciousness with 99.9% physics compliance",
                "neuroplasticity that never forgets while always learning",
                "runLiveProof()",
                "MAXIMUM CHALLENGE: Solve these simultaneously"
            ]
            
            found_removed_content = []
            for content in removed_content:
                if content in html_content:
                    found_removed_content.append(content)
            
            # Check for content that should still be present
            expected_content = [
                "Physics Demo",
                "Test Formats", 
                "Consciousness",
                "Math Reasoning",
                "Vision Analysis",
                "Document AI",
                "Deep Research",
                "AI Reasoning",
                "Interactive Chart",
                "Live Pipeline"
            ]
            
            missing_expected_content = []
            for content in expected_content:
                if content not in html_content:
                    missing_expected_content.append(content)
            
            # Results
            if found_removed_content:
                logger.error(f"❌ Found content that should have been removed: {found_removed_content}")
                return False
            
            if missing_expected_content:
                logger.error(f"❌ Missing expected content: {missing_expected_content}")
                return False
            
            logger.info("✅ Console cleanup successful!")
            logger.info(f"✅ All unwanted buttons removed: {len(removed_content)} items")
            logger.info(f"✅ All expected buttons preserved: {len(expected_content)} items")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Console cleanup test error: {e}")
            return False
    
    def test_remaining_functionality(self) -> bool:
        """Test that remaining functionality still works"""
        try:
            logger.info("🧪 Testing remaining functionality...")
            
            # Test precision visualization (should still work)
            test_payload = {
                "chart_type": "bar",
                "data": {
                    "categories": ["Signal", "Reasoning", "Physics"],
                    "values": [85, 82, 90],
                    "title": "Clean Console Test Chart",
                    "xlabel": "Component",
                    "ylabel": "Performance (%)"
                },
                "style": "scientific"
            }
            
            response = requests.post(
                f"{self.base_url}/visualization/chart",
                json=test_payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info("✅ Precision visualization still working")
                    return True
                else:
                    logger.error(f"❌ Visualization failed: {result}")
                    return False
            else:
                logger.error(f"❌ Visualization endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Functionality test error: {e}")
            return False
    
    def run_cleanup_verification(self) -> bool:
        """Run complete cleanup verification"""
        logger.info("🧹 Starting Console Cleanup Verification")
        logger.info("=" * 50)
        
        tests = [
            ("Console Cleanup", self.test_console_cleanup),
            ("Remaining Functionality", self.test_remaining_functionality)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🔬 Running: {test_name}")
            logger.info("-" * 30)
            
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: EXCEPTION - {e}")
        
        # Results
        logger.info("\n" + "=" * 50)
        logger.info("🏁 CLEANUP VERIFICATION RESULTS")
        logger.info("=" * 50)
        
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 CLEANUP SUCCESSFUL! Console is clean and functional!")
        else:
            logger.warning("⚠️ Cleanup verification had issues")
        
        return passed_tests == total_tests

def main():
    """Main test execution"""
    print("🧹 NIS Console Cleanup Verification")
    print("=" * 40)
    
    tester = ConsoleCleanupTester()
    success = tester.run_cleanup_verification()
    
    if success:
        print("\n✅ Console cleanup verification PASSED!")
        print("🎉 Unwanted buttons successfully removed!")
        print("✨ Console is clean and ready to use!")
    else:
        print("\n❌ Console cleanup verification FAILED!")
        print("⚠️ Check logs for details")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)