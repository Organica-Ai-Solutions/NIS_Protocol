#!/usr/bin/env python3
"""
Example 1: Basic NIS Protocol Usage

This example shows the simplest way to use NIS Protocol
for getting LLM responses.
"""

from nis_protocol import NISCore

def main():
    print("=" * 60)
    print("Example 1: Basic NIS Protocol Usage")
    print("=" * 60)
    
    # Initialize NIS Core
    print("\n1️⃣ Initializing NIS Protocol...")
    nis = NISCore()
    print("   ✅ Initialized!")
    
    # Get a simple LLM response
    print("\n2️⃣ Getting LLM response...")
    response = nis.get_llm_response(
        "What is the NIS Protocol in one sentence?"
    )
    
    print(f"\n🤖 Response: {response.get('content', 'No response')}")
    
    # Try with a specific provider
    print("\n3️⃣ Getting response from specific provider (Anthropic)...")
    try:
        response = nis.get_llm_response(
            "What are the benefits of autonomous AI?",
            provider="anthropic"
        )
        print(f"\n🎭 Claude says: {response.get('content', 'No response')}")
    except Exception as e:
        print(f"\n⚠️  Anthropic not available: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

