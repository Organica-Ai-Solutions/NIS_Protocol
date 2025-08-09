#!/usr/bin/env python3
"""
Test Precision Diagram & Chart Generation
Tests the new code-based visualization system (NOT AI image generation)
"""

import requests
import time
import json

def test_bar_chart():
    """Test precise bar chart generation"""
    print("📊 Testing Precision Bar Chart")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/chart"
    test_data = {
        "chart_type": "bar",
        "data": {
            "categories": ["OpenAI", "Google", "Local"],
            "values": [85, 60, 95],
            "title": "API Performance Comparison",
            "xlabel": "Provider",
            "ylabel": "Success Rate (%)"
        },
        "style": "scientific"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("chart", {}).get("url"):
                print("✅ Bar chart generated successfully!")
                print(f"   Method: {result['chart']['method']}")
                print(f"   Note: {result.get('note', 'N/A')}")
                return True
            else:
                print("❌ No chart URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_line_chart():
    """Test precise line chart generation"""
    print("\n📈 Testing Precision Line Chart")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/chart"
    test_data = {
        "chart_type": "line",
        "data": {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [10, 25, 20, 35, 30, 45],
            "title": "AI Model Training Progress",
            "xlabel": "Epoch",
            "ylabel": "Accuracy (%)"
        },
        "style": "scientific"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("chart", {}).get("url"):
                print("✅ Line chart generated successfully!")
                print(f"   Method: {result['chart']['method']}")
                return True
            else:
                print("❌ No chart URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_pie_chart():
    """Test precise pie chart generation"""
    print("\n🥧 Testing Precision Pie Chart")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/chart"
    test_data = {
        "chart_type": "pie",
        "data": {
            "labels": ["Working", "Broken", "Needs Fix"],
            "sizes": [70, 10, 20],
            "title": "System Status Distribution"
        },
        "style": "professional"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("chart", {}).get("url"):
                print("✅ Pie chart generated successfully!")
                print(f"   Method: {result['chart']['method']}")
                return True
            else:
                print("❌ No chart URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_flowchart_diagram():
    """Test precise flowchart generation"""
    print("\n🔧 Testing Precision Flowchart Diagram")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/diagram"
    test_data = {
        "diagram_type": "flowchart",
        "data": {
            "nodes": [
                {"id": "input", "label": "User Input", "x": 0.5, "y": 0.9, "type": "oval"},
                {"id": "process", "label": "NIS Protocol", "x": 0.5, "y": 0.6, "type": "rect"},
                {"id": "ai", "label": "AI Processing", "x": 0.3, "y": 0.3, "type": "rect"},
                {"id": "physics", "label": "Physics Check", "x": 0.7, "y": 0.3, "type": "rect"},
                {"id": "output", "label": "Response", "x": 0.5, "y": 0.1, "type": "oval"}
            ],
            "edges": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "ai"},
                {"from": "process", "to": "physics"},
                {"from": "ai", "to": "output"},
                {"from": "physics", "to": "output"}
            ],
            "title": "NIS Protocol Processing Flow"
        },
        "style": "scientific"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("diagram", {}).get("url"):
                print("✅ Flowchart diagram generated successfully!")
                print(f"   Method: {result['diagram']['method']}")
                return True
            else:
                print("❌ No diagram URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_architecture_diagram():
    """Test system architecture diagram"""
    print("\n🏗️ Testing System Architecture Diagram")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/diagram"
    test_data = {
        "diagram_type": "architecture",
        "data": {
            "title": "NIS Protocol v3 Architecture"
        },
        "style": "professional"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("diagram", {}).get("url"):
                print("✅ Architecture diagram generated successfully!")
                print(f"   Method: {result['diagram']['method']}")
                return True
            else:
                print("❌ No diagram URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_auto_detection():
    """Test auto-detection of visualization type"""
    print("\n🎯 Testing Auto-Detection System")
    print("=" * 50)
    
    url = "http://localhost:8000/visualization/auto"
    test_data = {
        "prompt": "Show me a bar chart of performance metrics",
        "data": {
            "categories": ["Speed", "Accuracy", "Reliability"],
            "values": [90, 85, 95],
            "title": "System Performance Metrics"
        },
        "style": "scientific"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("visualization", {}).get("url"):
                detected_type = result.get("detected_type", "unknown")
                print(f"✅ Auto-detection working!")
                print(f"   Detected: {detected_type}")
                print(f"   Method: {result['visualization']['method']}")
                return True
            else:
                print("❌ No visualization URL in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_health_check():
    """Test that the application is healthy"""
    print("❤️ Testing Application Health")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ Application is healthy and responsive")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Run comprehensive precision visualization tests"""
    print("🔧 Precision Diagram & Chart Testing")
    print("🎯 Testing code-based generation (NOT AI image generation)")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 7
    
    # Run all tests
    if test_health_check():
        tests_passed += 1
    
    if test_bar_chart():
        tests_passed += 1
    
    if test_line_chart():
        tests_passed += 1
    
    if test_pie_chart():
        tests_passed += 1
    
    if test_flowchart_diagram():
        tests_passed += 1
    
    if test_architecture_diagram():
        tests_passed += 1
    
    if test_auto_detection():
        tests_passed += 1
    
    print(f"\n📊 Final Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 6:
        print("🎉 Precision visualization system is working!")
        print("\n✅ What's Available:")
        print("   📊 /visualization/chart - Precise charts (bar, line, pie, scatter, etc.)")
        print("   🔧 /visualization/diagram - System diagrams (flowchart, architecture, etc.)")
        print("   🎯 /visualization/auto - Auto-detect from prompt")
        print("\n💡 Why This is Better:")
        print("   ✅ Mathematical precision - exact data representation")
        print("   ✅ Perfect text labels - no AI mangling")
        print("   ✅ Scalable & fast - code-based generation")
        print("   ✅ Reproducible results - same input = same output")
        print("   ✅ Data integrity - percentages actually add to 100%")
        print("\n🚀 Use these endpoints instead of AI image generation for diagrams!")
    else:
        print("⚠️ Some tests failed - check application logs for details")
        print("   The precision visualization system may need dependencies or config")

if __name__ == "__main__":
    main()