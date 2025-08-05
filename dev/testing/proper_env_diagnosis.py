#!/usr/bin/env python3
"""
Proper Environment Diagnosis - Uses NIS environment loading system
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_and_check_env():
    """Load environment using NIS system and check configuration"""
    print("🔍 NIS Protocol Environment Diagnosis")
    print("=" * 50)
    
    try:
        # Load environment using dotenv (like the app does)
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        
        if env_file.exists():
            print(f"✅ Found .env file at: {env_file}")
            load_dotenv(env_file)
            print("✅ Environment variables loaded from .env")
        else:
            print("❌ No .env file found")
            return False
            
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False
    
    print("\n🔑 API Key Status:")
    
    # Check API keys after loading
    api_keys = {
        "GOOGLE_API_KEY": "Google",
        "OPENAI_API_KEY": "OpenAI", 
        "GCP_PROJECT_ID": "Google Project ID",
        "ANTHROPIC_API_KEY": "Anthropic"
    }
    
    placeholder_values = [
        "your_google_api_key_here",
        "your_openai_api_key_here", 
        "your_anthropic_api_key_here",
        "YOUR_API_KEY_HERE"
    ]
    
    configured_count = 0
    for key, provider in api_keys.items():
        value = os.getenv(key)
        if value and value not in placeholder_values and len(value) > 10:
            print(f"  ✅ {provider}: CONFIGURED ({len(value)} chars)")
            configured_count += 1
        elif value:
            print(f"  ⚠️ {provider}: PLACEHOLDER VALUE")
        else:
            print(f"  ❌ {provider}: NOT SET")
    
    print(f"\n📊 Summary: {configured_count}/{len(api_keys)} providers configured")
    
    if configured_count == 0:
        print("\n🔧 Need to configure API keys in .env file!")
        return False
    
    return True

def test_image_generation():
    """Test actual image generation providers"""
    print("\n🎨 Testing Image Generation Providers:")
    
    try:
        # Test with proper environment loading
        from src.utils.env_config import EnvironmentConfig
        env_config = EnvironmentConfig()
        
        # Test Google Provider
        try:
            from src.llm.providers.google_provider import GoogleProvider
            config = {
                "api_key": env_config.get_env("GOOGLE_API_KEY"),
                "model": "gemini-1.5-flash"
            }
            provider = GoogleProvider(config)
            print(f"  ✅ GoogleProvider initialized")
            print(f"     - GCP Project: {provider.gcp_project_id or 'NOT_SET'}")
            print(f"     - Use Mock: {provider.use_mock}")
            
        except Exception as e:
            print(f"  ❌ GoogleProvider failed: {e}")
        
        # Test OpenAI Provider  
        try:
            from src.llm.providers.openai_provider import OpenAIProvider
            config = {
                "api_key": env_config.get_env("OPENAI_API_KEY"),
                "model": "gpt-4"
            }
            provider = OpenAIProvider(config)
            print(f"  ✅ OpenAIProvider initialized")
            
        except Exception as e:
            print(f"  ❌ OpenAIProvider failed: {e}")
            
    except Exception as e:
        print(f"  ❌ Environment config failed: {e}")

def test_actual_generation():
    """Test generating a real image"""
    print("\n🧪 Testing Actual Image Generation:")
    
    try:
        from src.utils.env_config import EnvironmentConfig
        env_config = EnvironmentConfig()
        
        # Try OpenAI first (usually more reliable)
        openai_key = env_config.get_env("OPENAI_API_KEY")
        if openai_key and len(openai_key) > 10:
            print("  🎯 Testing OpenAI DALL-E...")
            
            try:
                from src.llm.providers.openai_provider import OpenAIProvider
                config = {"api_key": openai_key, "model": "gpt-4"}
                provider = OpenAIProvider(config)
                
                # Test image generation
                import asyncio
                async def test_openai():
                    result = await provider.generate_image(
                        prompt="A simple red circle on white background",
                        size="256x256",
                        quality="standard",
                        num_images=1
                    )
                    return result
                
                result = asyncio.run(test_openai())
                if result.get("status") == "success":
                    print("  ✅ OpenAI image generation SUCCESS!")
                    print(f"     Generated {len(result.get('images', []))} images")
                else:
                    print(f"  ❌ OpenAI failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ OpenAI test failed: {e}")
        else:
            print("  ⚠️ OpenAI API key not configured")
            
    except Exception as e:
        print(f"  ❌ Test setup failed: {e}")

if __name__ == "__main__":
    if load_and_check_env():
        test_image_generation()
        test_actual_generation()
    else:
        print("\n❌ Environment not properly configured")
        print("   Please check your .env file")