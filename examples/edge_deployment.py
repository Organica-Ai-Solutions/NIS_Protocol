#!/usr/bin/env python3
"""
NIS Protocol v3.2 - Edge Deployment Example

This example shows how to deploy NIS Protocol on edge devices.
"""

import os
import sys
from pathlib import Path

def check_edge_requirements():
    """Check if edge deployment requirements are met"""
    print("🔍 Checking edge deployment requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for edge deployment")
        return False
    
    # Check available memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            print("⚠️  Warning: Low memory available for edge deployment")
    except ImportError:
        print("ℹ️  psutil not available, skipping memory check")
    
    print("✅ Edge deployment requirements check complete")
    return True

def deploy_edge_agent():
    """Deploy a lightweight NIS Protocol agent for edge devices"""
    print("🚀 Deploying NIS Protocol Edge Agent...")
    
    try:
        # Import minimal components for edge deployment
        from core.agent import NISAgent
        
        # Configure for edge deployment (minimal resources)
        config = {
            "mode": "edge",
            "memory_limit": "512MB",
            "cpu_cores": 1,
            "enable_physics": False,  # Disable heavy physics calculations
            "enable_vision": True,    # Keep vision for edge use cases
            "cache_size": "64MB"
        }
        
        # Initialize edge agent
        agent = NISAgent(config=config)
        
        print("✅ Edge agent deployed successfully!")
        print(f"📊 Configuration: {config}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Edge deployment failed: {e}")
        return None

def main():
    """Main edge deployment workflow"""
    print("🌐 NIS Protocol v3.2 - Edge Deployment")
    print("=" * 50)
    
    # Check requirements
    if not check_edge_requirements():
        sys.exit(1)
    
    # Deploy agent
    agent = deploy_edge_agent()
    
    if agent:
        print("\n🎉 Edge deployment complete!")
        print("Your NIS Protocol agent is ready for edge computing.")
    else:
        print("\n❌ Edge deployment failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
