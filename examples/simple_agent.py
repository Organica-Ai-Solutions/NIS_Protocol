#!/usr/bin/env python3
"""
NIS Protocol v3.2 - Simple Agent Example

This example demonstrates how to create and use a basic NIS Protocol agent.
"""

from core.agent import NISAgent
from agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent

def main():
    """Simple example of using NIS Protocol agents"""
    
    # Initialize a basic NIS agent
    print("🚀 Initializing NIS Protocol Agent...")
    
    try:
        # Create an enhanced conscious agent
        agent = EnhancedConsciousAgent()
        
        # Simple interaction
        response = agent.process_input("Hello, NIS Protocol!")
        print(f"Agent Response: {response}")
        
        print("✅ NIS Protocol agent working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all dependencies are installed.")

if __name__ == "__main__":
    main()
