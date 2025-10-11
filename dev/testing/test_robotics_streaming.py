#!/usr/bin/env python3
"""
NIS Protocol Robotics Streaming Test Client
Tests WebSocket, SSE, and HTTP streaming endpoints

Usage:
    python3 dev/testing/test_robotics_streaming.py --mode websocket
    python3 dev/testing/test_robotics_streaming.py --mode sse
    python3 dev/testing/test_robotics_streaming.py --mode trajectory
"""

import asyncio
import json
import time
import argparse
import sys

# WebSocket test
async def test_websocket_control(robot_id="drone_test_001", num_commands=5):
    """Test WebSocket real-time control"""
    print(f"\n🔥 Testing WebSocket Control: {robot_id}")
    print("=" * 60)
    
    try:
        import websockets
        
        uri = f"ws://localhost/ws/robotics/control/{robot_id}"
        print(f"📡 Connecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!\n")
            
            # Test 1: Forward Kinematics
            print("Test 1: Forward Kinematics (Drone)")
            print("-" * 40)
            
            command = {
                'type': 'forward_kinematics',
                'robot_type': 'drone',
                'joint_angles': [5000, 5000, 5000, 5000]
            }
            
            await websocket.send(json.dumps(command))
            print(f"📤 Sent: {command['type']}")
            
            response = await websocket.recv()
            data = json.loads(response)
            
            print(f"📥 Received: {data['type']}")
            print(f"   Robot: {data['robot_id']}")
            print(f"   Computation time: {data['result'].get('computation_time', 0)*1000:.2f}ms")
            if 'total_thrust' in data['result']:
                print(f"   Total thrust: {data['result']['total_thrust']:.2f}N")
                print(f"   Physics valid: {data['result']['physics_valid']}")
            print()
            
            # Test 2: Multiple commands
            print(f"Test 2: Rapid Commands (x{num_commands})")
            print("-" * 40)
            
            start_time = time.time()
            
            for i in range(num_commands):
                # Vary motor speeds
                speeds = [5000 + i*100, 5000 + i*100, 5000 + i*100, 5000 + i*100]
                
                command = {
                    'type': 'forward_kinematics',
                    'robot_type': 'drone',
                    'joint_angles': speeds
                }
                
                await websocket.send(json.dumps(command))
                response = await websocket.recv()
                data = json.loads(response)
                
                print(f"   Command {i+1}/{num_commands}: {data['result'].get('computation_time', 0)*1000:.2f}ms")
            
            elapsed = time.time() - start_time
            rate = num_commands / elapsed
            
            print(f"\n📊 Performance:")
            print(f"   Total time: {elapsed:.3f}s")
            print(f"   Average rate: {rate:.1f} Hz")
            print(f"   Latency: {(elapsed/num_commands)*1000:.2f}ms per command")
            
            # Test 3: Get stats
            print("\nTest 3: Stats Query")
            print("-" * 40)
            
            await websocket.send(json.dumps({'type': 'get_stats'}))
            response = await websocket.recv()
            data = json.loads(response)
            
            stats = data['result']
            print(f"   Total commands: {stats['total_commands']}")
            print(f"   Validated: {stats['validated_commands']}")
            print(f"   Success rate: {stats['success_rate']*100:.1f}%")
            
            print(f"\n✅ WebSocket test complete!")
            
    except ImportError:
        print("❌ websockets library not installed!")
        print("   Install: pip install websockets")
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")


async def test_sse_telemetry(robot_id="drone_test_002", duration=5, update_rate=10):
    """Test Server-Sent Events telemetry"""
    print(f"\n📊 Testing SSE Telemetry: {robot_id}")
    print("=" * 60)
    
    try:
        import httpx
        
        url = f"http://localhost/robotics/telemetry/{robot_id}?update_rate={update_rate}"
        print(f"📡 Connecting to: {url}")
        print(f"   Update rate: {update_rate}Hz")
        print(f"   Duration: {duration}s\n")
        
        frame_count = 0
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=duration + 5) as response:
                print("✅ SSE stream connected!\n")
                
                async for line in response.aiter_lines():
                    if time.time() - start_time > duration:
                        break
                    
                    if line.startswith('data: '):
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        frame_count += 1
                        
                        if frame_count == 1 or frame_count % update_rate == 0:
                            print(f"Frame {data['frame']:4d} | "
                                  f"Rate: {data['update_rate']}Hz | "
                                  f"Status: {data['status']} | "
                                  f"Commands: {data['stats']['total_commands']}")
        
        elapsed = time.time() - start_time
        actual_rate = frame_count / elapsed
        
        print(f"\n📊 Performance:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {elapsed:.2f}s")
        print(f"   Actual rate: {actual_rate:.1f} Hz (target: {update_rate} Hz)")
        print(f"   Frame interval: {(elapsed/frame_count)*1000:.2f}ms")
        
        print(f"\n✅ SSE telemetry test complete!")
        
    except ImportError:
        print("❌ httpx library not installed!")
        print("   Install: pip install httpx")
    except Exception as e:
        print(f"❌ SSE test failed: {e}")


async def test_trajectory_stream(robot_id="drone_test_003"):
    """Test HTTP chunked trajectory streaming"""
    print(f"\n🎬 Testing Trajectory Streaming: {robot_id}")
    print("=" * 60)
    
    try:
        import httpx
        
        url = "http://localhost/robotics/execute_trajectory_stream"
        
        payload = {
            'robot_id': robot_id,
            'robot_type': 'drone',
            'waypoints': [[0, 0, 0], [5, 5, 10], [10, 0, 15]],
            'duration': 3.0,
            'num_points': 50,
            'execution_rate': 25  # 25Hz execution
        }
        
        print(f"📡 POSTing to: {url}")
        print(f"   Waypoints: {len(payload['waypoints'])}")
        print(f"   Points: {payload['num_points']}")
        print(f"   Duration: {payload['duration']}s")
        print(f"   Rate: {payload['execution_rate']}Hz\n")
        
        start_time = time.time()
        update_count = 0
        
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', url, json=payload, timeout=30) as response:
                print("✅ Stream started!\n")
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        update_count += 1
                        
                        status = data['status']
                        
                        if status == 'planning':
                            print(f"🔧 Planning trajectory...")
                        
                        elif status == 'planned':
                            print(f"✅ Planned: {data['total_points']} points, {data['duration']}s duration\n")
                            print("Executing trajectory:")
                        
                        elif status == 'executing':
                            progress = data['progress']
                            point = data['point']
                            total = data['total']
                            pos = data['trajectory_point']['position']
                            
                            # Progress bar
                            bar_length = 30
                            filled = int(bar_length * progress / 100)
                            bar = '█' * filled + '░' * (bar_length - filled)
                            
                            print(f"   [{bar}] {progress:5.1f}% | "
                                  f"Point {point:3d}/{total} | "
                                  f"Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}]",
                                  end='\r')
                        
                        elif status == 'complete':
                            print(f"\n\n✅ Trajectory complete!")
                            print(f"   Total points: {data['total_points']}")
                            print(f"   Execution time: {data['execution_time']:.2f}s")
                        
                        elif status == 'error':
                            print(f"\n❌ Error: {data['error']}")
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 Performance:")
        print(f"   Total updates: {update_count}")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Update rate: {update_count/elapsed:.1f} Hz")
        
        print(f"\n✅ Trajectory streaming test complete!")
        
    except ImportError:
        print("❌ httpx library not installed!")
        print("   Install: pip install httpx")
    except Exception as e:
        print(f"❌ Trajectory stream test failed: {e}")


async def test_all():
    """Run all streaming tests"""
    print("\n" + "="*60)
    print("🤖 NIS PROTOCOL ROBOTICS STREAMING TEST SUITE")
    print("="*60)
    
    # Test 1: WebSocket
    await test_websocket_control(num_commands=10)
    await asyncio.sleep(1)
    
    # Test 2: SSE
    await test_sse_telemetry(duration=3, update_rate=10)
    await asyncio.sleep(1)
    
    # Test 3: Trajectory streaming
    await test_trajectory_stream()
    
    print("\n" + "="*60)
    print("✅ ALL STREAMING TESTS COMPLETE!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test NIS Protocol robotics streaming')
    parser.add_argument('--mode', choices=['websocket', 'sse', 'trajectory', 'all'], 
                        default='all',
                        help='Test mode to run')
    parser.add_argument('--robot-id', default='drone_test_001',
                        help='Robot ID for testing')
    parser.add_argument('--duration', type=int, default=5,
                        help='Duration for SSE test (seconds)')
    parser.add_argument('--rate', type=int, default=10,
                        help='Update rate for SSE test (Hz)')
    parser.add_argument('--commands', type=int, default=10,
                        help='Number of commands for WebSocket test')
    
    args = parser.parse_args()
    
    if args.mode == 'websocket':
        asyncio.run(test_websocket_control(args.robot_id, args.commands))
    elif args.mode == 'sse':
        asyncio.run(test_sse_telemetry(args.robot_id, args.duration, args.rate))
    elif args.mode == 'trajectory':
        asyncio.run(test_trajectory_stream(args.robot_id))
    else:
        asyncio.run(test_all())


if __name__ == "__main__":
    main()

