#!/usr/bin/env python3
"""
Example 2: Autonomous Mode

This example demonstrates the autonomous AI capabilities where
the system automatically detects intent and selects tools.
"""

import asyncio
from nis_protocol import NISCore


async def main():
    print("=" * 60)
    print("Example 2: Autonomous Mode")
    print("=" * 60)
    
    # Initialize NIS Core
    print("\n🤖 Initializing NIS Protocol...")
    nis = NISCore()
    print("   ✅ Initialized!")
    
    # Test cases for different intents
    test_cases = [
        "Calculate 255 * 387",
        "What is 2+2?",
        "Run python code to print hello world",
        "Validate a bouncing ball physics scenario",
        "Tell me about quantum computing",
    ]
    
    print("\n" + "=" * 60)
    print("Running autonomous tests...")
    print("=" * 60)
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n\n{'=' * 60}")
        print(f"Test {i}: {message}")
        print("=" * 60)
        
        try:
            # Process autonomously
            result = await nis.process_autonomously(message)
            
            print(f"\n🎯 Intent Detected: {result['intent']}")
            print(f"🔧 Tools Used: {', '.join(result['tools_used'])}")
            print(f"💭 Reasoning: {result['reasoning']}")
            print(f"\n🤖 Response: {result['response'][:200]}...")
            
            if result['success']:
                print("\n✅ Success!")
            else:
                print("\n⚠️  Partial success")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n\n" + "=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)
    
    print("""
    
    💡 Key Takeaway:
    The system automatically:
    - Detected different intents (math, code, physics, conversation)
    - Selected appropriate tools (calculator, runner, PINN, LLM)
    - Executed everything without manual configuration
    - Provided reasoning for its decisions
    
    This is TRUE autonomous AI! 🚀
    """)


if __name__ == "__main__":
    asyncio.run(main())

