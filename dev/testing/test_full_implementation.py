#!/usr/bin/env python3
"""
Full Implementation Test for NIS Protocol v3.0
Tests the complete Docker Compose stack with all services
"""

import requests
import json
import time
import socket
from typing import Dict, Any, List

def check_port(host: str, port: int) -> bool:
    """Check if a port is open and responding"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def test_service_connectivity() -> Dict[str, bool]:
    """Test connectivity to all backend services"""
    services = {
        "PostgreSQL": ("localhost", 5432),
        "Redis": ("localhost", 6379), 
        "Kafka": ("localhost", 29092),
        "NIS App": ("localhost", 8000),
        "Nginx": ("localhost", 80)
    }
    
    results = {}
    print("🔌 Testing Service Connectivity:")
    print("-" * 40)
    
    for service, (host, port) in services.items():
        is_up = check_port(host, port)
        status = "✅ UP" if is_up else "❌ DOWN"
        print(f"{service:12} {host}:{port:5} {status}")
        results[service] = is_up
    
    return results

def test_api_endpoints() -> Dict[str, Dict[str, Any]]:
    """Test all NIS Protocol API endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        {"url": f"{base_url}/", "method": "GET", "name": "Root"},
        {"url": f"{base_url}/health", "method": "GET", "name": "Health Check"},
        {"url": f"{base_url}/metrics", "method": "GET", "name": "System Metrics"},
        {"url": f"{base_url}/consciousness/status", "method": "GET", "name": "Consciousness Status"},
        {"url": f"{base_url}/infrastructure/status", "method": "GET", "name": "Infrastructure Status"},
    ]
    
    post_endpoints = [
        {
            "url": f"{base_url}/process",
            "method": "POST",
            "name": "Process Input", 
            "data": {"text": "Test full stack integration", "generate_speech": False}
        },
        {
            "url": f"{base_url}/chat",
            "method": "POST",
            "name": "Enhanced Chat",
            "data": {"message": "Hello from full implementation!", "user_id": "full_test"}
        }
    ]
    
    print("\n📡 Testing API Endpoints:")
    print("-" * 40)
    
    results = {}
    
    # Test GET endpoints
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint["url"], timeout=15)
            if response.status_code == 200:
                print(f"✅ {endpoint['name']}")
                results[endpoint["name"]] = {"success": True, "status": 200, "data": response.json()}
            else:
                print(f"⚠️ {endpoint['name']} - Status: {response.status_code}")
                results[endpoint["name"]] = {"success": False, "status": response.status_code}
        except Exception as e:
            print(f"❌ {endpoint['name']} - Error: {str(e)[:50]}...")
            results[endpoint["name"]] = {"success": False, "error": str(e)}
    
    # Test POST endpoints  
    for endpoint in post_endpoints:
        try:
            response = requests.post(endpoint["url"], json=endpoint["data"], timeout=15)
            if response.status_code == 200:
                print(f"✅ {endpoint['name']}")
                results[endpoint["name"]] = {"success": True, "status": 200, "data": response.json()}
            else:
                print(f"⚠️ {endpoint['name']} - Status: {response.status_code}")
                results[endpoint["name"]] = {"success": False, "status": response.status_code}
        except Exception as e:
            print(f"❌ {endpoint['name']} - Error: {str(e)[:50]}...")
            results[endpoint["name"]] = {"success": False, "error": str(e)}
    
    return results

def test_infrastructure_integration() -> Dict[str, Any]:
    """Test integration with Redis, Kafka, PostgreSQL"""
    base_url = "http://localhost:8000"
    
    print("\n🏗️ Testing Infrastructure Integration:")
    print("-" * 40)
    
    results = {}
    
    # Test infrastructure status
    try:
        response = requests.get(f"{base_url}/infrastructure/status", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Infrastructure Status: {data.get('status', 'unknown')}")
            print(f"   Redis: {data.get('redis', 'unknown')}")
            print(f"   Kafka: {data.get('kafka', 'unknown')}")
            results["infrastructure"] = data
        else:
            print(f"⚠️ Infrastructure Status - HTTP {response.status_code}")
            results["infrastructure"] = {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"❌ Infrastructure Status - Error: {str(e)[:50]}...")
        results["infrastructure"] = {"error": str(e)}
    
    # Test consciousness with infrastructure
    try:
        response = requests.get(f"{base_url}/consciousness/status", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Consciousness: {data.get('status', 'unknown')}")
            print(f"   Awareness Level: {data.get('awareness_level', 'unknown')}")
            results["consciousness"] = data
        else:
            print(f"⚠️ Consciousness - HTTP {response.status_code}")
            results["consciousness"] = {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"❌ Consciousness - Error: {str(e)[:50]}...")
        results["consciousness"] = {"error": str(e)}
    
    return results

def test_nginx_proxy() -> bool:
    """Test Nginx reverse proxy"""
    print("\n🌐 Testing Nginx Reverse Proxy:")
    print("-" * 40)
    
    try:
        # Test direct access via Nginx
        response = requests.get("http://localhost/health", timeout=10)
        if response.status_code == 200:
            print("✅ Nginx proxy working - health check via port 80")
            return True
        else:
            print(f"⚠️ Nginx proxy - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Nginx proxy - Error: {str(e)[:50]}...")
        return False

def main():
    """Run comprehensive full implementation test"""
    print("🚀 NIS PROTOCOL v3.0 FULL IMPLEMENTATION TEST")
    print("=" * 60)
    print("Testing complete Docker Compose stack with all services")
    print("=" * 60)
    
    # Wait a moment for services to stabilize
    print("⏳ Waiting for services to stabilize...")
    time.sleep(10)
    
    # Test service connectivity
    connectivity = test_service_connectivity()
    
    # Test API endpoints
    api_results = test_api_endpoints()
    
    # Test infrastructure integration
    infra_results = test_infrastructure_integration()
    
    # Test Nginx proxy
    nginx_working = test_nginx_proxy()
    
    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("🎯 FULL IMPLEMENTATION TEST RESULTS")
    print("=" * 60)
    
    # Service connectivity summary
    services_up = sum(1 for up in connectivity.values() if up)
    total_services = len(connectivity)
    print(f"\n🔌 Service Connectivity: {services_up}/{total_services}")
    for service, is_up in connectivity.items():
        status = "✅" if is_up else "❌"
        print(f"   {status} {service}")
    
    # API endpoints summary
    api_success = sum(1 for result in api_results.values() if result.get("success", False))
    total_endpoints = len(api_results)
    print(f"\n📡 API Endpoints: {api_success}/{total_endpoints}")
    for endpoint, result in api_results.items():
        status = "✅" if result.get("success", False) else "❌"
        print(f"   {status} {endpoint}")
    
    # Infrastructure summary
    print(f"\n🏗️ Infrastructure Integration:")
    infra_status = infra_results.get("infrastructure", {})
    if "error" not in infra_status:
        print(f"   ✅ Status: {infra_status.get('status', 'unknown')}")
        print(f"   ✅ Redis: {infra_status.get('redis', 'unknown')}")
        print(f"   ✅ Kafka: {infra_status.get('kafka', 'unknown')}")
    else:
        print(f"   ❌ Infrastructure: {infra_status.get('error', 'unknown')}")
    
    # Nginx proxy summary
    print(f"\n🌐 Nginx Proxy: {'✅ Working' if nginx_working else '❌ Failed'}")
    
    # Overall assessment
    print("\n" + "-" * 60)
    total_checks = services_up + api_success + (1 if nginx_working else 0)
    max_checks = total_services + total_endpoints + 1
    
    print(f"📊 Overall Score: {total_checks}/{max_checks} ({total_checks/max_checks*100:.1f}%)")
    
    if total_checks >= max_checks * 0.8:
        print("🎉 FULL IMPLEMENTATION IS WORKING EXCELLENTLY!")
        print("✅ NIS Protocol v3.0 complete stack is operational!")
    elif total_checks >= max_checks * 0.6:
        print("👍 FULL IMPLEMENTATION IS MOSTLY WORKING!")
        print("⚠️ Some minor issues detected, but core functionality is good.")
    else:
        print("⚠️ FULL IMPLEMENTATION HAS SIGNIFICANT ISSUES")
        print("❌ Multiple services/endpoints are failing.")
    
    print("\n🎯 Access Points:")
    print(f"   • Main API: http://localhost:8000")
    print(f"   • Via Nginx: http://localhost")
    print(f"   • Dashboard: http://localhost:5000")
    print(f"   • Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 