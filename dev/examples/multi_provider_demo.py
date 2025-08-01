#!/usr/bin/env python3
"""
NIS Protocol Multi-Provider LLM Demo

This demo shows how to use the modular multi-provider LLM system.
Developers can choose from OpenAI, Anthropic Claude, DeepSeek, or BitNet 2.

Usage:
    python examples/multi_provider_demo.py
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.llm_manager import LLMManager
from src.llm.base_llm_provider import LLMMessage, LLMRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_provider_status():
    """Show the status of all available providers."""
    print("🤖 NIS Protocol Multi-Provider LLM System")
    print("=" * 50)
    
    try:
        llm_manager = LLMManager()
        
        # Show available providers
        available = llm_manager.get_available_providers()
        configured = llm_manager.get_configured_providers()
        
        print(f"📦 Available Providers: {len(available)}")
        for provider in available:
            status = "✅ CONFIGURED" if provider in configured else "❌ NOT CONFIGURED"
            if provider == "mock":
                status = "🔧 DEVELOPMENT ONLY"
            print(f"  • {provider.upper()}: {status}")
        
        print(f"\n🔧 Configured Providers: {len(configured)}")
        if configured:
            for provider in configured:
                print(f"  • {provider.upper()}")
        else:
            print("  • No providers configured - using mock provider")
        
        print(f"\n🎯 Currently Active: {llm_manager._resolve_provider()}")
        
    except Exception as e:
        print(f"❌ Error checking providers: {e}")

def show_configuration_guide():
    """Show how to configure each provider."""
    print("\n📝 Provider Configuration Guide")
    print("=" * 50)
    
    providers = {
        "OpenAI": {
            "steps": [
                "1. Get API key from https://platform.openai.com/api-keys",
                "2. Edit config/llm_config.json",
                "3. Set 'enabled': true in providers.openai",
                "4. Replace 'YOUR_OPENAI_API_KEY' with your actual key"
            ],
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "features": ["Chat completion", "Embeddings", "Function calling"]
        },
        "Anthropic Claude": {
            "steps": [
                "1. Get API key from https://console.anthropic.com/",
                "2. Edit config/llm_config.json", 
                "3. Set 'enabled': true in providers.anthropic",
                "4. Replace 'YOUR_ANTHROPIC_API_KEY' with your actual key"
            ],
            "models": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
            "features": ["Chat completion", "Long context", "Constitutional AI"]
        },
        "DeepSeek": {
            "steps": [
                "1. Get API key from https://platform.deepseek.com/",
                "2. Edit config/llm_config.json",
                "3. Set 'enabled': true in providers.deepseek", 
                "4. Replace 'YOUR_DEEPSEEK_API_KEY' with your actual key"
            ],
            "models": ["deepseek-chat", "deepseek-coder"],
            "features": ["Chat completion", "Code generation", "Embeddings"]
        },
        "BitNet 2": {
            "steps": [
                "1. Install BitNet 2 from https://github.com/microsoft/BitNet",
                "2. Download a compatible model",
                "3. Edit config/llm_config.json",
                "4. Set 'enabled': true and update 'model_path'"
            ],
            "models": ["bitnet-b1.58-3b", "Custom BitNet models"],
            "features": ["Local inference", "1-bit quantization", "CPU efficient"]
        }
    }
    
    for provider, info in providers.items():
        print(f"\n🔧 {provider}")
        print("-" * 30)
        for step in info["steps"]:
            print(f"  {step}")
        print(f"  Models: {', '.join(info['models'])}")
        print(f"  Features: {', '.join(info['features'])}")

async def test_provider_responses():
    """Test responses from different providers."""
    print("\n🧪 Testing Provider Responses")
    print("=" * 50)
    
    try:
        llm_manager = LLMManager()
        
        # Test message
        test_messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content="You are a helpful AI assistant specialized in archaeological research."
            ),
            LLMMessage(
                role=LLMRole.USER,
                content="What are the key considerations when documenting archaeological artifacts?"
            )
        ]
        
        # Get current provider
        provider = llm_manager.get_provider()
        provider_name = provider.__class__.__name__.replace("Provider", "")
        
        print(f"🤖 Using Provider: {provider_name}")
        print(f"📝 Question: {test_messages[1].content}")
        print("\n💬 Response:")
        
        response = await provider.generate(test_messages, max_tokens=200)
        print(f"{response.content}")
        
        print(f"\n📊 Usage: {response.usage}")
        print(f"🏁 Finish Reason: {response.finish_reason}")
        
    except Exception as e:
        print(f"❌ Error testing provider: {e}")

async def demonstrate_agent_specific_providers():
    """Show how different agents can use different providers."""
    print("\n🧠 Agent-Specific Provider Demo")
    print("=" * 50)
    
    agent_types = [
        "perception_agent",
        "memory_agent", 
        "emotional_agent",
        "executive_agent",
        "motor_agent"
    ]
    
    try:
        llm_manager = LLMManager()
        
        for agent_type in agent_types:
            try:
                provider = llm_manager.get_agent_llm(agent_type)
                provider_name = provider.__class__.__name__.replace("Provider", "")
                print(f"  🤖 {agent_type}: {provider_name}")
            except Exception as e:
                print(f"  ❌ {agent_type}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Error getting agent providers: {e}")

def create_example_config():
    """Create an example configuration showing how to enable providers."""
    print("\n📄 Example Configuration")
    print("=" * 50)
    
    example_config = {
        "providers": {
            "openai": {
                "enabled": True,  # ← Enable this provider
                "api_key": "sk-your-actual-openai-api-key-here",
                "models": {
                    "chat": {"name": "gpt-4o"}
                }
            },
            "anthropic": {
                "enabled": False,  # ← Disabled
                "api_key": "YOUR_ANTHROPIC_API_KEY"
            }
        },
        "agent_llm_config": {
            "default_provider": "openai",  # ← Set default
            "perception_agent": {
                "provider": "openai"  # ← Specific provider for this agent
            }
        }
    }
    
    print("Example config/llm_config.json snippet:")
    print(json.dumps(example_config, indent=2))

def main():
    """Main demo function."""
    print("🚀 NIS Protocol Multi-Provider LLM Demo")
    print("🎯 Choose your LLM provider: OpenAI, Anthropic, DeepSeek, or BitNet 2")
    print("\n")
    
    # Show current status
    show_provider_status()
    
    # Show configuration guide
    show_configuration_guide()
    
    # Show example config
    create_example_config()
    
    # Test current provider
    print("\n" + "=" * 50)
    print("🧪 Live Provider Test")
    asyncio.run(test_provider_responses())
    
    # Show agent-specific providers
    asyncio.run(demonstrate_agent_specific_providers())
    
    print("\n" + "=" * 50)
    print("✅ Demo Complete!")
    print("\n💡 Next Steps:")
    print("  1. Choose your preferred provider(s)")
    print("  2. Get API keys from the provider's website")
    print("  3. Edit config/llm_config.json to enable and configure")
    print("  4. Set 'enabled': true and add your API key")
    print("  5. Restart your application to use the new provider")
    print("\n🔧 The system will systematically fall back to mock provider for development.")

if __name__ == "__main__":
    main() 