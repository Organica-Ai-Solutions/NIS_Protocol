#!/usr/bin/env python3
"""
Test all NIS WebSocket endpoints
"""
import asyncio
import websockets
import json

async def test_genui_ws():
    """Test GenUI WebSocket endpoint"""
    uri = "ws://localhost:8007/genui/ws"
    try:
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "message": "Hello from GenUI test",
                "userId": "test_user",
                "enableStreaming": True
            }))
            
            # Receive response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"✅ GenUI WebSocket: {data.get('type', 'unknown')}")
            return True
    except Exception as e:
        print(f"❌ GenUI WebSocket: {e}")
        return False

async def test_chat_ws():
    """Test Main Chat WebSocket endpoint"""
    uri = "ws://localhost:8007/chat/ws"
    try:
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "message": "Hello from chat test",
                "userId": "test_user",
                "useTools": False
            }))
            
            # Wait for multiple messages (message + complete)
            message_received = False
            complete_received = False
            
            for _ in range(3):  # Try up to 3 messages
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    data = json.loads(response)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'message':
                        message_received = True
                    elif msg_type == 'complete':
                        complete_received = True
                        break
                    elif msg_type == 'error':
                        print(f"❌ Chat WebSocket: {data.get('error', 'Unknown error')}")
                        return False
                except asyncio.TimeoutError:
                    break
            
            if message_received or complete_received:
                print(f"✅ Chat WebSocket: {'message' if message_received else 'complete'}")
                return True
            else:
                print(f"❌ Chat WebSocket: No valid response received")
                return False
    except Exception as e:
        print(f"❌ Chat WebSocket: {e}")
        return False

async def test_agentic_ws():
    """Test Agentic Chat WebSocket endpoint"""
    uri = "ws://localhost:8007/chat/agentic/ws"
    try:
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "message": "Hello from agentic test",
                "enableAgents": False
            }))
            
            # Receive response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"✅ Agentic WebSocket: {data.get('type', 'unknown')}")
            return True
    except Exception as e:
        print(f"❌ Agentic WebSocket: {e}")
        return False

async def test_pipeline_ws():
    """Test pipeline WebSocket endpoint"""
    uri = "ws://localhost:8007/v4/pipeline/ws"
    try:
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "operation": "genesis",
                "data": {"prompt": "Test pipeline"}
            }))
            
            # Receive response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"✅ pipeline WebSocket: {data.get('type', 'unknown')}")
            return True
    except Exception as e:
        print(f"❌ pipeline WebSocket: {e}")
        return False

async def main():
    print("\n🧪 Testing NIS WebSocket Endpoints\n")
    print("=" * 50)
    
    results = await asyncio.gather(
        test_genui_ws(),
        test_chat_ws(),
        test_agentic_ws(),
        test_pipeline_ws(),
        return_exceptions=True
    )
    
    print("=" * 50)
    success_count = sum(1 for r in results if r is True)
    print(f"\n✅ {success_count}/4 WebSocket endpoints working\n")
    
    return success_count == 4

if __name__ == "__main__":
    asyncio.run(main())

