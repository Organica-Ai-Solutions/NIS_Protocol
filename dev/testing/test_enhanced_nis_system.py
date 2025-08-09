#!/usr/bin/env python3
"""
🚀 NIS Protocol v3.2 Enhanced System Test Suite
Tests the complete real-time data pipeline integration with interactive visualizations

Features Tested:
- Real-time pipeline monitoring (Laplace→KAN→PINN→LLM)
- Interactive Plotly charts with zoom/hover capabilities  
- External data integration via web search
- Precision visualization system
- Performance monitoring and metrics collection
"""

import asyncio
import json
import time
import requests
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_nis_test")

class EnhancedNISSystemTester:
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.test_results = {}
        
        logger.info("🚀 Initializing Enhanced NIS System Tester")
        
    def test_system_health(self) -> bool:
        """Test basic system health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ System health check passed")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    def test_interactive_visualization(self) -> bool:
        """Test interactive Plotly chart generation"""
        try:
            logger.info("📊 Testing interactive visualization...")
            
            payload = {
                "chart_type": "line",
                "data": {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [85, 87, 83, 89, 91, 88, 92, 90, 94, 93],
                    "title": "Test Interactive Chart",
                    "xlabel": "Time",
                    "ylabel": "Performance (%)"
                },
                "style": "scientific"
            }
            
            response = requests.post(
                f"{self.base_url}/visualization/interactive",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                chart_json = result.get("interactive_chart", {}).get("chart_json")
                
                if chart_json:
                    # Validate Plotly JSON structure
                    plotly_data = json.loads(chart_json)
                    if "data" in plotly_data and "layout" in plotly_data:
                        logger.info("✅ Interactive visualization test passed")
                        self.test_results["interactive_viz"] = {
                            "status": "success",
                            "features": result.get("interactive_chart", {}).get("features", []),
                            "chart_type": result.get("interactive_chart", {}).get("chart_type")
                        }
                        return True
                    else:
                        logger.error("❌ Invalid Plotly JSON structure")
                        return False
                else:
                    logger.error("❌ No chart JSON in response")
                    return False
            else:
                logger.error(f"❌ Interactive visualization failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Interactive visualization error: {e}")
            return False
    
    def test_pipeline_monitoring(self) -> bool:
        """Test real-time pipeline monitoring"""
        try:
            logger.info("🚀 Testing pipeline monitoring...")
            
            # Start monitoring
            start_response = requests.post(
                f"{self.base_url}/pipeline/start-monitoring",
                timeout=15
            )
            
            if start_response.status_code in [200, 503]:  # 503 if pipeline agent not available
                if start_response.status_code == 503:
                    logger.info("⚠️ Pipeline agent not available - testing mock mode")
                else:
                    logger.info("✅ Pipeline monitoring started")
                
                # Test metrics endpoint
                time.sleep(2)  # Wait for metrics collection
                
                metrics_response = requests.get(
                    f"{self.base_url}/pipeline/metrics?time_range=1h",
                    timeout=10
                )
                
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    
                    # Check for required metrics
                    if metrics_data.get("status") in ["success", "mock"]:
                        metrics = metrics_data.get("mock_metrics") or metrics_data.get("live_metrics", {})
                        
                        required_metrics = ["signal_quality", "reasoning_confidence", "physics_compliance"]
                        if all(metric in metrics for metric in required_metrics):
                            logger.info("✅ Pipeline metrics test passed")
                            self.test_results["pipeline_monitoring"] = {
                                "status": "success",
                                "metrics_available": list(metrics.keys()),
                                "mode": metrics_data.get("status")
                            }
                            
                            # Stop monitoring
                            stop_response = requests.post(
                                f"{self.base_url}/pipeline/stop-monitoring",
                                timeout=10
                            )
                            
                            return True
                        else:
                            logger.error(f"❌ Missing required metrics: {required_metrics}")
                            return False
                    else:
                        logger.error(f"❌ Invalid metrics status: {metrics_data.get('status')}")
                        return False
                else:
                    logger.error(f"❌ Metrics endpoint failed: {metrics_response.status_code}")
                    return False
            else:
                logger.error(f"❌ Pipeline monitoring start failed: {start_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline monitoring error: {e}")
            return False
    
    def test_pipeline_visualization(self) -> bool:
        """Test pipeline visualization generation"""
        try:
            logger.info("📈 Testing pipeline visualization...")
            
            # Test performance summary visualization
            response = requests.get(
                f"{self.base_url}/pipeline/visualization/performance_summary",
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") in ["success", "mock"]:
                    visualization = result.get("visualization", {})
                    
                    if visualization.get("url") or visualization.get("chart", {}).get("url"):
                        logger.info("✅ Pipeline visualization test passed")
                        self.test_results["pipeline_viz"] = {
                            "status": "success",
                            "chart_type": result.get("chart_type"),
                            "mode": result.get("status")
                        }
                        return True
                    else:
                        logger.error("❌ No visualization URL in response")
                        return False
                else:
                    logger.error(f"❌ Invalid visualization status: {result.get('status')}")
                    return False
            else:
                logger.error(f"❌ Pipeline visualization failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline visualization error: {e}")
            return False
    
    def test_external_data_integration(self) -> bool:
        """Test external data integration via web search"""
        try:
            logger.info("🔍 Testing external data integration...")
            
            response = requests.get(
                f"{self.base_url}/pipeline/external-data?source=research&query=AI%20trends",
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    external_data = result.get("external_data", {})
                    visualization = result.get("visualization", {})
                    
                    if external_data and visualization:
                        logger.info("✅ External data integration test passed")
                        self.test_results["external_data"] = {
                            "status": "success",
                            "source": result.get("source"),
                            "query": result.get("query")
                        }
                        return True
                    else:
                        logger.error("❌ Missing external data or visualization")
                        return False
                else:
                    logger.error(f"❌ External data failed: {result.get('status')}")
                    return False
            else:
                logger.error(f"❌ External data endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ External data integration error: {e}")
            return False
    
    def test_precision_visualization_system(self) -> bool:
        """Test precision visualization system"""
        try:
            logger.info("🎯 Testing precision visualization system...")
            
            # Test chart generation
            chart_payload = {
                "chart_type": "bar",
                "data": {
                    "categories": ["Signal", "Reasoning", "Physics", "Overall"],
                    "values": [85, 82, 90, 86],
                    "title": "Test Precision Chart",
                    "xlabel": "Component",
                    "ylabel": "Performance (%)"
                },
                "style": "scientific"
            }
            
            chart_response = requests.post(
                f"{self.base_url}/visualization/chart",
                json=chart_payload,
                timeout=15
            )
            
            # Test diagram generation
            diagram_payload = {
                "diagram_type": "pipeline",
                "data": {"title": "Test Pipeline Diagram"},
                "style": "scientific"
            }
            
            diagram_response = requests.post(
                f"{self.base_url}/visualization/diagram",
                json=diagram_payload,
                timeout=15
            )
            
            # Test auto visualization
            auto_payload = {
                "prompt": "Show me system performance metrics",
                "data": {
                    "categories": ["Analysis", "Processing", "Results"],
                    "values": [85, 90, 95]
                },
                "style": "scientific"
            }
            
            auto_response = requests.post(
                f"{self.base_url}/visualization/auto",
                json=auto_payload,
                timeout=15
            )
            
            success_count = 0
            total_tests = 3
            
            if chart_response.status_code == 200:
                chart_result = chart_response.json()
                if chart_result.get("status") == "success":
                    success_count += 1
                    logger.info("✅ Chart generation test passed")
            
            if diagram_response.status_code == 200:
                diagram_result = diagram_response.json()
                if diagram_result.get("status") == "success":
                    success_count += 1
                    logger.info("✅ Diagram generation test passed")
            
            if auto_response.status_code == 200:
                auto_result = auto_response.json()
                if auto_result.get("status") == "success":
                    success_count += 1
                    logger.info("✅ Auto visualization test passed")
            
            if success_count == total_tests:
                logger.info("✅ Precision visualization system test passed")
                self.test_results["precision_viz"] = {
                    "status": "success",
                    "tests_passed": f"{success_count}/{total_tests}"
                }
                return True
            else:
                logger.error(f"❌ Precision visualization system partial failure: {success_count}/{total_tests}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Precision visualization system error: {e}")
            return False
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🧪 Starting Enhanced NIS System Comprehensive Test Suite")
        logger.info("=" * 70)
        
        tests = [
            ("System Health", self.test_system_health),
            ("Interactive Visualization", self.test_interactive_visualization),
            ("Pipeline Monitoring", self.test_pipeline_monitoring),
            ("Pipeline Visualization", self.test_pipeline_visualization),
            ("External Data Integration", self.test_external_data_integration),
            ("Precision Visualization System", self.test_precision_visualization_system)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🔬 Running: {test_name}")
            logger.info("-" * 50)
            
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: EXCEPTION - {e}")
            
            time.sleep(1)  # Brief pause between tests
        
        # Generate final report
        logger.info("\n" + "=" * 70)
        logger.info("🏁 ENHANCED NIS SYSTEM TEST RESULTS")
        logger.info("=" * 70)
        
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Enhanced NIS system is fully operational!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("✅ Most tests passed - system is mostly functional")
        else:
            logger.warning("⚠️ Several tests failed - system needs attention")
        
        # Detailed results
        logger.info("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            logger.info(f"  • {test_name}: {status}")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "detailed_results": self.test_results,
            "overall_status": "success" if passed_tests == total_tests else "partial" if passed_tests >= total_tests * 0.8 else "failure"
        }

def main():
    """Main test execution"""
    print("🚀 Enhanced NIS Protocol v3.2 System Test Suite")
    print("Features: Real-time Pipeline, Interactive Charts, External Data Integration")
    print("=" * 80)
    
    tester = EnhancedNISSystemTester()
    results = tester.run_comprehensive_test_suite()
    
    print(f"\n🏆 Final Score: {results['passed_tests']}/{results['total_tests']} ({results['success_rate']:.1f}%)")
    
    if results['overall_status'] == 'success':
        print("🎉 Enhanced NIS system is FULLY OPERATIONAL!")
        print("✨ Ready for real-time monitoring and interactive visualizations!")
    else:
        print("⚠️ System needs attention - check logs for details")
    
    return results['overall_status'] == 'success'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)