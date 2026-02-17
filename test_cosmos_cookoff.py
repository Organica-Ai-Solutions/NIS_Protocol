#!/usr/bin/env python3
"""
Quick test for Cosmos Cookoff routes
"""

import asyncio
import sys

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from routes.cosmos_cookoff import router as cookoff_router
        print("✅ cosmos_cookoff routes imported")
    except Exception as e:
        print(f"❌ cosmos_cookoff routes failed: {e}")
        return False
    
    try:
        from routes.cosmos_cookoff_websocket import router as ws_router
        print("✅ cosmos_cookoff_websocket routes imported")
    except Exception as e:
        print(f"❌ cosmos_cookoff_websocket routes failed: {e}")
        return False
    
    try:
        from src.agents.cosmos import (
            CosmosReason2Agent,
            CosmosCookoffAgent,
            get_cosmos_reason2_agent,
            create_cookoff_agent
        )
        print("✅ Cosmos agents imported")
    except Exception as e:
        print(f"❌ Cosmos agents failed: {e}")
        return False
    
    return True

async def test_agent():
    """Test the Cookoff agent"""
    print("\nTesting Cookoff Agent...")
    
    try:
        from src.agents.cosmos import create_cookoff_agent
        import numpy as np
        
        agent = await create_cookoff_agent()
        print(f"✅ Agent created: {agent.__class__.__name__}")
        
        # Test with mock frame
        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        
        # Test video analytics
        result = await agent.analyze_video_stream(
            frames=mock_frames,
            query="Test analysis"
        )
        print(f"✅ Video analytics: confidence={result.combined_confidence:.2f}")
        
        # Test robot planning
        result = await agent.plan_robot_action(
            query="Pick up the red cube",
            camera_image=mock_frames[0]
        )
        print(f"✅ Robot planning: confidence={result.combined_confidence:.2f}")
        print(f"   Actions: {result.action_recommendations[:2]}")
        
        # Get stats
        stats = agent.get_demo_stats()
        print(f"✅ Stats: {stats['total_queries']} queries")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fastapi_routes():
    """Test FastAPI route registration"""
    print("\nTesting FastAPI route registration...")
    
    try:
        from fastapi import FastAPI
        from routes.cosmos_cookoff import router as cookoff_router
        from routes.cosmos_cookoff_websocket import router as ws_router
        
        app = FastAPI()
        app.include_router(cookoff_router)
        app.include_router(ws_router)
        
        # List routes
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        print(f"✅ Registered {len(routes)} routes:")
        for route in routes[:10]:
            print(f"   {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Route registration failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 Cosmos Cookoff Test Suite")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        sys.exit(1)
    
    # Test FastAPI routes
    if not test_fastapi_routes():
        print("\n❌ Route tests failed!")
        sys.exit(1)
    
    # Test agent
    if not asyncio.run(test_agent()):
        print("\n❌ Agent tests failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nCosmos Cookoff routes are ready for deployment!")

if __name__ == "__main__":
    main()
