#!/usr/bin/env python3
"""
Simple Multi-Provider Test

A lightweight test of the multi-provider LLM system without complex dependencies.
"""

import os
import sys
import json
import asyncio

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Direct imports to avoid complex dependency chains
from src.llm.llm_manager import LLMManager
from src.llm.base_llm_provider import LLMMessage, LLMRole

async def test_providers():
    """Test the multi-provider system."""
    print("🤖 NIS Protocol Multi-Provider Test")
    print("=" * 40)
    
    try:
        # Initialize LLM manager
        llm_manager = LLMManager()
        
        # Show available providers
        available = llm_manager.get_available_providers()
        print(f"📦 Available Providers: {available}")
        
        # Show configured providers
        configured = llm_manager.get_configured_providers()
        print(f"✅ Configured Providers: {configured}")
        
        # Get current provider
        current_provider = llm_manager._resolve_provider()
        print(f"🎯 Current Provider: {current_provider}")
        
        # Test provider
        provider = llm_manager.get_provider()
        provider_name = provider.__class__.__name__
        print(f"🔧 Using: {provider_name}")
        
        # Create test messages
        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content="You are an AI assistant specialized in archaeology."
            ),
            LLMMessage(
                role=LLMRole.USER,
                content="What are the key steps in documenting an archaeological find?"
            )
        ]
        
        print(f"\n📝 Test Question: {messages[1].content}")
        print("\n💬 Response:")
        
        # Generate response
        response = await provider.generate(messages, max_tokens=150)
        print(f"{response.content}")
        
        print(f"\n📊 Metadata: {response.metadata}")
        print(f"🏁 Finish Reason: {response.finish_reason}")
        
        print("\n✅ Multi-provider system working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_config_status():
    """Show configuration status."""
    print("\n📋 Configuration Status")
    print("=" * 40)
    
    config_path = "config/llm_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        providers = config.get("providers", {})
        for name, settings in providers.items():
            enabled = settings.get("enabled", False)
            has_key = "api_key" in settings and settings["api_key"] not in [
                "YOUR_API_KEY_HERE", "YOUR_OPENAI_API_KEY", 
                "YOUR_ANTHROPIC_API_KEY", "YOUR_DEEPSEEK_API_KEY"
            ]
            
            status = "✅ READY" if enabled and (has_key or name in ["mock", "bitnet"]) else "❌ NOT CONFIGURED"
            print(f"  {name.upper()}: {status}")
            
        fallback = config.get("agent_llm_config", {}).get("fallback_to_mock", True)
        print(f"\n🔧 Mock Fallback: {'✅ Enabled' if fallback else '❌ Disabled'}")
    else:
        print("❌ Configuration file not found")

if __name__ == "__main__":
    show_config_status()
    asyncio.run(test_providers()) 