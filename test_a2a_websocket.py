#!/usr/bin/env python3
"""
Test script for A2A WebSocket endpoint
Tests the official GenUI A2A protocol implementation
"""

import asyncio
import websockets
import json
from datetime import datetime


async def test_a2a_connection():
    """Test basic A2A WebSocket connection"""
    uri = "ws://localhost:8000/a2a"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # 1. Receive AgentCard
            print("\n📥 Waiting for AgentCard...")
            agent_card_raw = await websocket.recv()
            agent_card = json.loads(agent_card_raw)
            print(f"✅ Received AgentCard:")
            print(json.dumps(agent_card, indent=2))
            
            # 2. Send user message
            print("\n📤 Sending user message...")
            user_message = {
                "type": "user_message",
                "text": "Write a simple Python hello world function",
                "surfaceId": "main"
            }
            await websocket.send(json.dumps(user_message))
            print("✅ Message sent")
            
            # 3. Receive responses
            print("\n📥 Receiving responses...")
            message_count = 0
            
            while True:
                try:
                    response_raw = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=30.0
                    )
                    response = json.loads(response_raw)
                    message_count += 1
                    
                    msg_type = response.get('type')
                    print(f"\n[{message_count}] Received: {msg_type}")
                    
                    if msg_type == 'begin_rendering':
                        task_id = response.get('taskId')
                        print(f"   🎬 Rendering started (task: {task_id})")
                    
                    elif msg_type == 'surface_update':
                        surface_id = response.get('surfaceId')
                        widget_type = response.get('data', {}).get('type')
                        print(f"   🎨 Surface update: {surface_id}")
                        print(f"   📦 Widget: {widget_type}")
                        if widget_type == 'NISCodeBlock':
                            code = response.get('data', {}).get('code', '')
                            print(f"   💻 Code preview: {code[:50]}...")
                    
                    elif msg_type == 'text_chunk':
                        text = response.get('text', '')
                        print(f"   📝 Text: {text[:100]}...")
                    
                    elif msg_type == 'end_rendering':
                        task_id = response.get('taskId')
                        print(f"   🏁 Rendering complete (task: {task_id})")
                        break
                    
                    elif msg_type == 'error':
                        error = response.get('error')
                        print(f"   ❌ Error: {error}")
                        break
                    
                    else:
                        print(f"   ℹ️  Data: {json.dumps(response, indent=2)}")
                
                except asyncio.TimeoutError:
                    print("⏱️  Timeout waiting for response")
                    break
            
            print(f"\n✅ Test complete! Received {message_count} messages")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except ConnectionRefusedError:
        print("❌ Connection refused - is the backend running?")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_multiple_messages():
    """Test sending multiple messages in one session"""
    uri = "ws://localhost:8000/a2a"
    
    print(f"\n{'='*60}")
    print("🔄 Testing multiple messages in one session...")
    print(f"{'='*60}\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Receive AgentCard
            await websocket.recv()
            print("✅ Connected and received AgentCard")
            
            # Send multiple messages
            messages = [
                "What is 2+2?",
                "Write a Python function to calculate factorial",
                "Explain quantum computing in one sentence"
            ]
            
            for i, msg_text in enumerate(messages, 1):
                print(f"\n📤 Message {i}: {msg_text}")
                
                await websocket.send(json.dumps({
                    "type": "user_message",
                    "text": msg_text,
                    "surfaceId": "main"
                }))
                
                # Wait for end_rendering
                while True:
                    response = json.loads(await websocket.recv())
                    if response.get('type') == 'end_rendering':
                        print(f"✅ Response {i} complete")
                        break
            
            print("\n✅ Multiple message test complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_ping_pong():
    """Test ping/pong for connection keepalive"""
    uri = "ws://localhost:8000/a2a"
    
    print(f"\n{'='*60}")
    print("🏓 Testing ping/pong...")
    print(f"{'='*60}\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Receive AgentCard
            await websocket.recv()
            
            # Send ping
            print("📤 Sending ping...")
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Wait for pong
            response = json.loads(await websocket.recv())
            if response.get('type') == 'pong':
                print("✅ Received pong!")
            else:
                print(f"⚠️  Unexpected response: {response.get('type')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    print("="*60)
    print("🧪 A2A WebSocket Protocol Test Suite")
    print("="*60)
    
    # Test 1: Basic connection
    await test_a2a_connection()
    
    await asyncio.sleep(1)
    
    # Test 2: Multiple messages
    await test_multiple_messages()
    
    await asyncio.sleep(1)
    
    # Test 3: Ping/pong
    await test_ping_pong()
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
