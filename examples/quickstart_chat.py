#!/usr/bin/env python3
"""
NIS Protocol - Chat Quick Start Example

This example demonstrates how to use the NIS Protocol chat API for:
- Basic chat conversations
- Streaming responses
- Multi-agent collaboration

Run this after starting the NIS Protocol server:
    docker-compose up -d
    python examples/quickstart_chat.py
"""

import requests
import json
import sseclient  # pip install sseclient-py

BASE_URL = "http://localhost"


def example_basic_chat():
    """Simple chat request"""
    print("\n" + "="*60)
    print("💬 Basic Chat Example")
    print("="*60)
    
    payload = {
        "message": "Explain forward kinematics in robotics in 2 sentences.",
        "conversation_id": "demo_001"
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Response:\n{data.get('response', data)[:500]}")
    else:
        print(f"❌ Error: {response.status_code}")
    
    return response


def example_streaming_chat():
    """Streaming chat with real-time response"""
    print("\n" + "="*60)
    print("🌊 Streaming Chat Example")
    print("="*60)
    
    payload = {
        "message": "What is the NIS Protocol?",
        "stream": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        print("\n🤖 Streaming response:")
        client = sseclient.SSEClient(response)
        
        full_response = ""
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    chunk = data.get("chunk", data.get("content", ""))
                    print(chunk, end="", flush=True)
                    full_response += chunk
                except json.JSONDecodeError:
                    print(event.data, end="", flush=True)
        
        print("\n")
        return full_response
        
    except Exception as e:
        print(f"Note: Streaming requires sseclient-py: pip install sseclient-py")
        print(f"Error: {e}")
        return None


def example_physics_chat():
    """Chat about physics with validation"""
    print("\n" + "="*60)
    print("🔬 Physics-Aware Chat Example")
    print("="*60)
    
    payload = {
        "message": "Calculate the kinetic energy of a 2kg object moving at 10 m/s",
        "enable_physics": True
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Response:\n{data.get('response', str(data)[:500])}")
    else:
        print(f"❌ Error: {response.status_code}")
    
    return response


def example_health_check():
    """Check system health"""
    print("\n" + "="*60)
    print("🏥 Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status: {data.get('status')}")
        print(f"   Providers: {data.get('provider', [])}")
        print(f"   Active conversations: {data.get('conversations_active', 0)}")
    else:
        print(f"❌ Error: {response.status_code}")
    
    return response


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║         NIS Protocol - Chat Quick Start Example            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check connection
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("⚠️  Server may not be fully ready")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to NIS Protocol server")
        print("   Make sure to run: docker-compose up -d")
        return
    
    print("✅ Connected to NIS Protocol server\n")
    
    # Run examples
    example_health_check()
    example_basic_chat()
    example_physics_chat()
    
    # Streaming requires extra dependency
    try:
        import sseclient
        example_streaming_chat()
    except ImportError:
        print("\n💡 Install sseclient-py for streaming example: pip install sseclient-py")
    
    print(f"\n{'='*60}")
    print("✅ Examples completed!")
    print("📚 Full API docs: http://localhost/docs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
