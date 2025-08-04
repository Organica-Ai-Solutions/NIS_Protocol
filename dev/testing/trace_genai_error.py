#!/usr/bin/env python3
"""
Trace the exact genai error location
"""
import docker
import json

def trace_error():
    """Get detailed traceback of the genai error"""
    print("🔍 Tracing genai error with detailed logging...")
    
    # Add temporary debug logging to find the exact error
    debug_code = '''
import sys
sys.path.append("/home/nisuser/app/src")
import traceback
import asyncio

async def test_provider():
    try:
        from llm.providers.google_provider import GoogleProvider
        
        # Create provider
        config = {"api_key": "AIzaSyBTrH6g_AfGO43fzgTz21S94X6coPVI8tk"}
        provider = GoogleProvider(config)
        
        print("✅ Provider created")
        
        # Test image generation
        result = await provider.generate_image("test dragon", "artistic")
        print(f"Result: {result.get('status')}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("🔍 FULL TRACEBACK:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_provider())
    '''
    
    # Write debug script to container
    import subprocess
    try:
        result = subprocess.run([
            "docker-compose", "exec", "-T", "backend", 
            "python3", "-c", debug_code
        ], capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")  
        print(result.stderr)
        
        if "genai" in result.stderr:
            print("\n🎯 FOUND GENAI ERROR IN TRACEBACK!")
            
    except Exception as e:
        print(f"Error running trace: {e}")

if __name__ == "__main__":
    trace_error()