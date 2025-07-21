#!/usr/bin/env python3
"""
Setup Environment Configuration for NIS Protocol v3

This script helps users set up their .env file with the necessary API keys
and configuration for the NIS Protocol system.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a basic .env file from .env.example."""
    project_root = Path(__file__).parent.parent
    env_example_path = project_root / ".env.example"
    env_path = project_root / ".env"
    
    if not env_example_path.exists():
        print("❌ .env.example file not found!")
        return False
    
    if env_path.exists():
        response = input("🔄 .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("✅ Keeping existing .env file")
            return True
    
    # Copy .env.example to .env
    with open(env_example_path, 'r') as src:
        content = src.read()
    
    with open(env_path, 'w') as dst:
        dst.write(content)
    
    print(f"✅ Created .env file at {env_path}")
    return True

def setup_llm_providers():
    """Interactive setup for LLM providers."""
    print("\n🤖 LLM Provider Setup")
    print("=" * 50)
    
    providers = {
        "OpenAI": {
            "key_var": "OPENAI_API_KEY",
            "enable_var": "OPENAI_ENABLED",
            "url": "https://platform.openai.com/api-keys",
            "description": "GPT-4 and other OpenAI models"
        },
        "Anthropic": {
            "key_var": "ANTHROPIC_API_KEY", 
            "enable_var": "ANTHROPIC_ENABLED",
            "url": "https://console.anthropic.com/",
            "description": "Claude models for advanced reasoning"
        },
        "DeepSeek": {
            "key_var": "DEEPSEEK_API_KEY",
            "enable_var": "DEEPSEEK_ENABLED", 
            "url": "https://platform.deepseek.com/",
            "description": "Cost-effective reasoning and coding"
        }
    }
    
    env_updates = {}
    
    for provider_name, config in providers.items():
        print(f"\n📡 {provider_name} ({config['description']})")
        print(f"   Get API key: {config['url']}")
        
        setup_provider = input(f"   Setup {provider_name}? (y/N): ")
        if setup_provider.lower() == 'y':
            api_key = input(f"   Enter {provider_name} API key: ").strip()
            if api_key:
                env_updates[config['key_var']] = api_key
                env_updates[config['enable_var']] = 'true'
                print(f"   ✅ {provider_name} configured")
            else:
                print(f"   ⏭️  Skipping {provider_name}")
        else:
            env_updates[config['enable_var']] = 'false'
    
    return env_updates

def setup_default_provider(configured_providers):
    """Setup default LLM provider."""
    print("\n🎯 Default Provider Setup")
    print("=" * 50)
    
    enabled_providers = [
        provider.lower() for provider, enabled in configured_providers.items() 
        if enabled == 'true' and 'ENABLED' in provider
    ]
    enabled_providers = [p.replace('_enabled', '') for p in enabled_providers]
    
    if not enabled_providers:
        print("   📝 No providers enabled, using mock provider for testing")
        return {"DEFAULT_LLM_PROVIDER": "mock", "FALLBACK_TO_MOCK": "true"}
    
    print(f"   Available providers: {', '.join(enabled_providers)}")
    
    # Recommend Anthropic if available, otherwise first enabled
    if 'anthropic' in enabled_providers:
        default = 'anthropic'
        print(f"   💡 Recommending Anthropic for best reasoning performance")
    else:
        default = enabled_providers[0]
    
    choice = input(f"   Default provider [{default}]: ").strip().lower()
    if not choice:
        choice = default
    
    if choice in enabled_providers:
        print(f"   ✅ Default provider set to: {choice}")
        return {"DEFAULT_LLM_PROVIDER": choice, "FALLBACK_TO_MOCK": "true"}
    else:
        print(f"   ⚠️  Invalid choice, using: {default}")
        return {"DEFAULT_LLM_PROVIDER": default, "FALLBACK_TO_MOCK": "true"}

def setup_infrastructure():
    """Setup infrastructure configuration."""
    print("\n🏗️  Infrastructure Setup")
    print("=" * 50)
    
    env_updates = {}
    
    # Redis setup
    print("\n📦 Redis (for caching and memory)")
    use_redis = input("   Use Redis? (Y/n): ")
    if use_redis.lower() != 'n':
        redis_host = input("   Redis host [localhost]: ").strip() or "localhost"
        redis_port = input("   Redis port [6379]: ").strip() or "6379"
        redis_password = input("   Redis password (optional): ").strip()
        
        env_updates.update({
            "REDIS_HOST": redis_host,
            "REDIS_PORT": redis_port,
            "REDIS_PASSWORD": redis_password
        })
        print("   ✅ Redis configured")
    
    # Kafka setup
    print("\n📨 Kafka (for message processing)")
    use_kafka = input("   Use Kafka? (Y/n): ")
    if use_kafka.lower() != 'n':
        kafka_servers = input("   Kafka servers [localhost:9092]: ").strip() or "localhost:9092"
        env_updates["KAFKA_BOOTSTRAP_SERVERS"] = kafka_servers
        print("   ✅ Kafka configured")
    
    return env_updates

def update_env_file(env_updates):
    """Update the .env file with new values."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found! Run create_env_file() first.")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update lines with new values
    updated_lines = []
    updated_keys = set()
    
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key = line.split('=')[0].strip()
            if key in env_updates:
                updated_lines.append(f"{key}={env_updates[key]}\n")
                updated_keys.add(key)
            else:
                updated_lines.append(line + '\n')
        else:
            updated_lines.append(line + '\n')
    
    # Add any new keys that weren't in the file
    for key, value in env_updates.items():
        if key not in updated_keys:
            updated_lines.append(f"{key}={value}\n")
    
    # Write updated content
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"✅ Updated .env file with {len(env_updates)} settings")
    return True

def main():
    """Main setup function."""
    print("🚀 NIS Protocol v3 Environment Setup")
    print("=" * 50)
    print("This script will help you configure your environment for the NIS Protocol.")
    print("You can skip any step and configure manually later.\n")
    
    # Step 1: Create .env file
    print("📁 Step 1: Create .env file")
    if not create_env_file():
        return False
    
    # Step 2: Setup LLM providers
    llm_updates = setup_llm_providers()
    
    # Step 3: Setup default provider
    default_updates = setup_default_provider(llm_updates)
    
    # Step 4: Setup infrastructure
    infra_updates = setup_infrastructure()
    
    # Combine all updates
    all_updates = {**llm_updates, **default_updates, **infra_updates}
    
    # Step 5: Update .env file
    print("\n💾 Updating .env file...")
    if update_env_file(all_updates):
        print("\n🎉 Environment setup complete!")
        print("\n📋 Next steps:")
        print("   1. Review your .env file")
        print("   2. Run: python scripts/test_env_config.py")
        print("   3. Start using the NIS Protocol!")
        
        # Show summary
        enabled_providers = [
            key.replace('_ENABLED', '').lower() 
            for key, value in all_updates.items() 
            if key.endswith('_ENABLED') and value == 'true'
        ]
        if enabled_providers:
            print(f"\n✅ Enabled providers: {', '.join(enabled_providers)}")
        else:
            print("\n📝 Using mock provider for testing")
        
        return True
    else:
        print("\n❌ Setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 