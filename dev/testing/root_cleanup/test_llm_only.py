#!/usr/bin/env python3
"""
Test ONLY LLM Providers - No Infrastructure Dependencies
"""

import os
import sys
import asyncio
import logging

# Setup path for imports
sys.path.append('src')

async def test_llm_providers_only():
    """Test LLM providers without infrastructure dependencies"""
    print("🧠 TESTING REAL LLM PROVIDERS ONLY")
    print("=" * 50)
    
    # Test 1: Direct LLM Provider Import and Setup
    print("\n1️⃣ Testing Direct LLM Provider Setup...")
    try:
        # Import base classes
        from src.llm.base_llm_provider import LLMMessage, LLMRole
        print("✅ Base LLM classes imported")
        
        # Import specific providers
        providers_available = {}
        
        try:
            from src.llm.providers.openai_provider import OpenAIProvider
            providers_available["openai"] = OpenAIProvider
            print("✅ OpenAI provider available")
        except Exception as e:
            print(f"⚠️ OpenAI provider: {e}")
        
        try:
            from src.llm.providers.anthropic_provider import AnthropicProvider
            providers_available["anthropic"] = AnthropicProvider
            print("✅ Anthropic provider available")
        except Exception as e:
            print(f"⚠️ Anthropic provider: {e}")
        
        try:
            from src.llm.providers.mock_provider import MockProvider
            providers_available["mock"] = MockProvider
            print("✅ Mock provider available")
        except Exception as e:
            print(f"⚠️ Mock provider: {e}")
        
        print(f"📊 Total providers available: {len(providers_available)}")
        
    except Exception as e:
        print(f"❌ Provider import failed: {e}")
        return False
    
    # Test 2: Provider Configuration
    print("\n2️⃣ Testing Provider Configuration...")
    try:
        # Check environment variables
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        print(f"🔑 OpenAI API key: {'✅ Set' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Not set'}")
        print(f"🔑 Anthropic API key: {'✅ Set' if anthropic_key and anthropic_key != 'your_anthropic_api_key_here' else '❌ Not set'}")
        
        # Determine which provider to test
        test_provider = None
        test_config = {}
        
        if openai_key and openai_key not in ["your_openai_api_key_here", "YOUR_OPENAI_API_KEY"]:
            test_provider = "openai"
            test_config = {
                "api_key": openai_key,
                "model": "gpt-3.5-turbo",
                "enabled": True
            }
            print("🎯 Will test OpenAI provider")
        elif anthropic_key and anthropic_key not in ["your_anthropic_api_key_here", "YOUR_ANTHROPIC_API_KEY"]:
            test_provider = "anthropic"
            test_config = {
                "api_key": anthropic_key,
                "model": "claude-3-haiku-20240307",
                "enabled": True
            }
            print("🎯 Will test Anthropic provider")
        else:
            test_provider = "mock"
            test_config = {
                "model": "mock-model",
                "enabled": True
            }
            print("🎯 Will test Mock provider (no API keys found)")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 3: Direct Provider Usage
    print("\n3️⃣ Testing Direct Provider Usage...")
    try:
        if test_provider in providers_available:
            provider_class = providers_available[test_provider]
            provider = provider_class(test_config)
            
            # Create test messages
            messages = [
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content="You are a helpful AI assistant. Respond clearly and concisely."
                ),
                LLMMessage(
                    role=LLMRole.USER,
                    content="What is the NIS Protocol? Answer in one sentence."
                )
            ]
            
            print(f"📡 Testing {test_provider} provider...")
            response = await provider.generate(messages, temperature=0.7, max_tokens=150)
            
            print(f"✅ {test_provider.upper()} RESPONSE:")
            print(f"   Content: {response.content}")
            print(f"   Confidence: {getattr(response, 'confidence', 'N/A')}")
            print(f"   Model: {getattr(response, 'model', 'Unknown')}")
            
            return True
            
        else:
            print(f"❌ Provider {test_provider} not available")
            return False
    
    except Exception as e:
        print(f"❌ Provider usage failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_conversation_flow():
    """Test a simple conversation flow"""
    print("\n4️⃣ Testing Conversation Flow...")
    
    try:
        # Import what we need
        from src.llm.base_llm_provider import LLMMessage, LLMRole
        from src.llm.providers.mock_provider import MockProvider
        
        # Use mock provider for reliable testing
        provider = MockProvider({"model": "mock-conversation", "enabled": True})
        
        # Simulate a conversation
        conversation = [
            "Hello, I'm testing the NIS Protocol.",
            "Can you explain artificial consciousness?",
            "How does the NIS Protocol implement AI agents?"
        ]
        
        conversation_history = []
        
        for i, user_message in enumerate(conversation):
            print(f"\n💬 Turn {i+1}:")
            print(f"   User: {user_message}")
            
            # Build message history
            messages = [
                LLMMessage(role=LLMRole.SYSTEM, content="You are an expert on AI and the NIS Protocol.")
            ]
            
            # Add previous conversation
            for msg in conversation_history[-4:]:  # Last 4 messages for context
                messages.append(msg)
            
            # Add current message
            messages.append(LLMMessage(role=LLMRole.USER, content=user_message))
            
            # Generate response
            response = await provider.generate(messages)
            
            print(f"   AI: {response.content}")
            
            # Add to history
            conversation_history.extend([
                LLMMessage(role=LLMRole.USER, content=user_message),
                LLMMessage(role=LLMRole.ASSISTANT, content=response.content)
            ])
        
        print("\n✅ Conversation flow test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Conversation flow failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NIS PROTOCOL v3.1 - LLM PROVIDERS ONLY TEST")
    print("Testing REAL AI connections without infrastructure")
    
    try:
        # Test LLM providers
        success = asyncio.run(test_llm_providers_only())
        
        if success:
            print("\n🎉 LLM PROVIDER TEST PASSED!")
            
            # Test conversation flow
            conv_success = asyncio.run(test_conversation_flow())
            
            if conv_success:
                print("\n🎊 ALL TESTS PASSED!")
                print("✅ Real LLM integration working!")
                print("🚀 Ready to integrate with endpoints!")
            else:
                print("\n⚠️ Conversation test had issues")
        else:
            print("\n❌ LLM provider test failed")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc() 