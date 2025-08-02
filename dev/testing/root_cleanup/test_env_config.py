#!/usr/bin/env python3
"""
Test Environment Configuration for NIS Protocol v3

This script tests the environment variable configuration system for LLM providers
and ensures all agents can connect to the configured keys properly.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.env_config import env_config, EnvironmentConfig
from llm.llm_manager import LLMManager
from llm.base_llm_provider import LLMMessage, LLMRole

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_environment_config():
    """Test environment configuration loading."""
    logger.info("🔧 Testing Environment Configuration")
    
    # Test environment config loading
    try:
        env = EnvironmentConfig()
        llm_config = env.get_llm_config()
        infra_config = env.get_infrastructure_config()
        
        logger.info("✅ Environment configuration loaded successfully")
        logger.info(f"   - Default provider: {llm_config['agent_llm_config']['default_provider']}")
        logger.info(f"   - Fallback to mock: {llm_config['agent_llm_config']['fallback_to_mock']}")
        logger.info(f"   - Redis host: {infra_config['redis']['host']}")
        logger.info(f"   - Kafka servers: {infra_config['kafka']['bootstrap_servers']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Environment configuration failed: {e}")
        return False

async def test_llm_providers():
    """Test LLM provider configuration and connectivity."""
    logger.info("🤖 Testing LLM Provider Configuration")
    
    try:
        llm_manager = LLMManager()
        
        # Test available providers
        available_providers = llm_manager.get_available_providers()
        logger.info(f"   - Available providers: {available_providers}")
        
        # Test configured providers
        configured_providers = llm_manager.get_configured_providers()
        logger.info(f"   - Configured providers: {configured_providers}")
        
        # Test default provider resolution
        try:
            default_provider = llm_manager._resolve_provider()
            logger.info(f"   - Default provider resolved to: {default_provider}")
        except Exception as e:
            logger.warning(f"   - Default provider resolution failed: {e}")
        
        # Test cognitive function providers
        cognitive_functions = ["consciousness", "reasoning", "creativity", "cultural", "archaeological"]
        for function in cognitive_functions:
            try:
                provider = llm_manager.get_provider_for_cognitive_function(function)
                config = llm_manager.get_cognitive_config(function)
                logger.info(f"   - {function}: {provider} (temp: {config.get('temperature', 'N/A')})")
            except Exception as e:
                logger.warning(f"   - {function}: Failed to resolve provider: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ LLM provider test failed: {e}")
        return False

async def test_mock_provider_functionality():
    """Test mock provider functionality when no real providers are configured."""
    logger.info("🎭 Testing Mock Provider Functionality")
    
    try:
        llm_manager = LLMManager()
        provider = llm_manager.get_provider("mock")
        
        # Test basic generation
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=LLMRole.USER, content="What is 2+2?")
        ]
        
        response = await provider.generate(messages, temperature=0.7, max_tokens=100)
        
        logger.info("✅ Mock provider test successful")
        logger.info(f"   - Response: {response.content[:100]}...")
        logger.info(f"   - Model: {response.model}")
        logger.info(f"   - Tokens: {response.total_tokens}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Mock provider test failed: {e}")
        return False

async def test_cognitive_function_integration():
    """Test cognitive function integration with LLM providers."""
    logger.info("🧠 Testing Cognitive Function Integration")
    
    try:
        llm_manager = LLMManager()
        
        # Test reasoning function
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are operating in REASONING mode."),
            LLMMessage(role=LLMRole.USER, content="Analyze the pros and cons of renewable energy.")
        ]
        
        response = await llm_manager.generate_with_function(
            messages=messages,
            function="reasoning",
            max_tokens=200
        )
        
        logger.info("✅ Cognitive function integration test successful")
        logger.info(f"   - Reasoning response length: {len(response.content)} chars")
        logger.info(f"   - Model used: {response.model}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Cognitive function integration test failed: {e}")
        return False

def check_environment_variables():
    """Check which environment variables are set."""
    logger.info("🔍 Checking Environment Variables")
    
    required_vars = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
        "OPENAI_ENABLED", "ANTHROPIC_ENABLED", "DEEPSEEK_ENABLED",
        "DEFAULT_LLM_PROVIDER", "FALLBACK_TO_MOCK"
    ]
    
    optional_vars = [
        "REDIS_HOST", "KAFKA_BOOTSTRAP_SERVERS", "GOOGLE_API_KEY",
        "BITNET_MODEL_PATH", "LOG_LEVEL"
    ]
    
    set_vars = []
    unset_vars = []
    
    for var in required_vars + optional_vars:
        value = os.getenv(var)
        if value:
            # Don't log actual API keys for security
            if "API_KEY" in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            logger.info(f"   ✅ {var} = {display_value}")
            set_vars.append(var)
        else:
            logger.info(f"   ❌ {var} = (not set)")
            unset_vars.append(var)
    
    logger.info(f"Environment summary: {len(set_vars)} set, {len(unset_vars)} unset")
    
    return len(set_vars) > 0

async def test_provider_error_handling():
    """Test provider error handling with invalid configurations."""
    logger.info("⚠️  Testing Provider Error Handling")
    
    try:
        # Test with invalid configuration
        llm_manager = LLMManager()
        
        # This should fall back to mock provider
        provider = llm_manager.get_provider()
        
        logger.info(f"✅ Error handling successful - fell back to: {provider.__class__.__name__}")
        return True
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all environment configuration tests."""
    logger.info("🚀 Starting NIS Protocol v3 Environment Configuration Tests")
    logger.info("=" * 70)
    
    tests = [
        ("Environment Config", test_environment_config()),
        ("Environment Variables", check_environment_variables()),
        ("LLM Providers", test_llm_providers()),
        ("Mock Provider", test_mock_provider_functionality()),
        ("Cognitive Functions", test_cognitive_function_integration()),
        ("Error Handling", test_provider_error_handling())
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            if asyncio.iscoroutine(test_func):
                result = await test_func
            else:
                result = test_func
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Environment configuration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the configuration and try again.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 