#!/usr/bin/env python3
"""
NIS Protocol Agent Runner
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


async def run_agent():
    """Run a standalone NIS Protocol agent."""
    try:
        from nis_protocol import NISCore
        
        print("""
        ╔════════════════════════════════════════════════════════════════╗
        ║                                                                ║
        ║                  NIS Protocol Agent                            ║
        ║                                                                ║
        ╚════════════════════════════════════════════════════════════════╝
        
        🤖 Initializing NIS Protocol agent...
        """)
        
        # Initialize NIS Core
        nis = NISCore()
        
        print("✅ Agent initialized successfully!")
        print("\n💬 Type your messages below (or 'quit' to exit):\n")
        
        # Interactive loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process autonomously
                result = await nis.process_autonomously(user_input)
                
                print(f"\n🎯 Intent: {result['intent']}")
                print(f"🔧 Tools: {', '.join(result['tools_used'])}")
                print(f"\n🤖 Response: {result['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)


def run():
    """Entry point for agent runner."""
    asyncio.run(run_agent())


if __name__ == "__main__":
    run()

