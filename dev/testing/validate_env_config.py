#!/usr/bin/env python3
"""
NIS Protocol Environment Validation
Checks if .env file is properly configured before starting
"""

import os
import sys
from pathlib import Path

def validate_env_file():
    """Validate .env file configuration"""
    print("🔍 VALIDATING ENVIRONMENT CONFIGURATION")
    print("=" * 45)
    
    env_file = Path(".env")
    
    # Check if .env exists
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Run ./start.sh to create template")
        return False
    
    # Read .env file with explicit UTF-8 encoding
    env_vars = {}
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except UnicodeDecodeError:
        print("❌ .env file has encoding issues!")
        print("   Try recreating the file or check for special characters")
        return False
    
    print("📋 Checking API key configuration...")
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic", 
        "DEEPSEEK_API_KEY": "DeepSeek",
        "GOOGLE_API_KEY": "Google"
    }
    
    configured_providers = []
    placeholder_values = [
        "your_openai_api_key_here",
        "your_anthropic_api_key_here", 
        "your_deepseek_api_key_here",
        "your_google_api_key_here",
        "YOUR_API_KEY_HERE"
    ]
    
    for key, provider in api_keys.items():
        value = env_vars.get(key, "")
        
        if not value or value in placeholder_values:
            print(f"   ⚠️  {provider}: Not configured")
        elif len(value) < 10:
            print(f"   ❌ {provider}: Too short (likely placeholder)")
        else:
            print(f"   ✅ {provider}: Configured")
            configured_providers.append(provider)
    
    # Check if at least one provider is configured
    if not configured_providers:
        print("\n❌ NO API KEYS CONFIGURED!")
        print("   You need at least one LLM provider to run the system.")
        print("\n🔑 To configure API keys:")
        print("   1. Edit .env file: notepad .env")
        print("   2. Replace placeholder values with real API keys")
        print("   3. Save the file")
        print("\n📖 Get API keys from:")
        print("   • OpenAI: https://platform.openai.com/api-keys")
        print("   • Anthropic: https://console.anthropic.com/")
        print("   • DeepSeek: https://platform.deepseek.com/")
        print("   • Google: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"\n✅ CONFIGURED PROVIDERS: {', '.join(configured_providers)}")
    
    # Check other important settings
    print("\n📋 Checking system configuration...")
    
    required_vars = {
        "DATABASE_URL": "Database connection",
        "KAFKA_BOOTSTRAP_SERVERS": "Kafka messaging",
        "REDIS_HOST": "Redis caching"
    }
    
    all_configured = True
    for key, description in required_vars.items():
        value = env_vars.get(key, "")
        if value:
            print(f"   ✅ {description}: {value}")
        else:
            print(f"   ❌ {description}: Not configured")
            all_configured = False
    
    # Final validation
    print("\n" + "=" * 45)
    if configured_providers and all_configured:
        print("🎉 ENVIRONMENT VALIDATION: PASSED")
        print("   Your system is ready to start!")
        return True
    elif configured_providers:
        print("⚠️  ENVIRONMENT VALIDATION: PARTIAL")
        print("   API keys configured, but some system settings missing")
        return True
    else:
        print("❌ ENVIRONMENT VALIDATION: FAILED")
        print("   Configuration issues need to be resolved")
        return False

if __name__ == "__main__":
    success = validate_env_file()
    if success:
        print(f"\n🚀 Ready to start: ./start.sh")
    sys.exit(0 if success else 1) 