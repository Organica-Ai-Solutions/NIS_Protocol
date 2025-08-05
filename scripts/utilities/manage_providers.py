#!/usr/bin/env python3
"""
🎯 NIS Protocol v3.2 - Provider Management Utility
CLI tool for managing the dynamic provider routing system

Features:
- View provider statistics
- Test routing for different tasks
- Update provider preferences  
- Monitor performance metrics
- Reload configuration
"""

import sys
import os
import yaml
import argparse
import requests
import json
import time
from datetime import datetime

def load_provider_registry(registry_path="configs/provider_registry.yaml"):
    """Load the provider registry"""
    try:
        with open(registry_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load provider registry: {e}")
        return None

def show_provider_status():
    """Show current provider status and capabilities"""
    print("🎯 NIS Protocol Provider Status")
    print("=" * 50)
    
    registry = load_provider_registry()
    if not registry:
        return
    
    providers = registry.get("providers", {})
    
    for provider_name, models in providers.items():
        print(f"\n📡 {provider_name.upper()}")
        print("-" * 30)
        
        for model_name, config in models.items():
            capabilities = ", ".join(config.get("capabilities", []))
            cost = config.get("cost_per_million_tokens", 0)
            latency = config.get("avg_latency_ms", 0)
            quality = config.get("quality_score", 0)
            
            print(f"  🤖 {model_name}")
            print(f"     Capabilities: {capabilities}")
            print(f"     Cost: ${cost:.2f}/M tokens")
            print(f"     Latency: {latency}ms")
            print(f"     Quality: {quality}/10")

def show_task_routing():
    """Show current task routing configuration"""
    print("\n🎯 Task Routing Configuration")
    print("=" * 50)
    
    registry = load_provider_registry()
    if not registry:
        return
    
    task_routing = registry.get("task_routing", {})
    
    for task_type, config in task_routing.items():
        primary = config.get("primary", [])
        fallback = config.get("fallback", [])
        strategy = config.get("strategy", "unknown")
        
        print(f"\n📋 {task_type.upper()}")
        print(f"   Strategy: {strategy}")
        print(f"   Primary: {', '.join(primary)}")
        print(f"   Fallback: {', '.join(fallback)}")

def test_routing(task_type):
    """Test routing for a specific task type"""
    print(f"\n🧪 Testing Routing for: {task_type}")
    print("=" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json={
                "message": f"Test {task_type} routing - what provider are you?",
                "agent_type": task_type
            },
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            provider = data.get("provider", "unknown")
            model = data.get("model", "unknown")
            duration = (end_time - start_time) * 1000
            
            print(f"✅ Routing successful!")
            print(f"   Provider: {provider}")
            print(f"   Model: {model}")
            print(f"   Response Time: {duration:.0f}ms")
            print(f"   Task: {task_type}")
            
            return True
        else:
            print(f"❌ Request failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def compare_providers():
    """Compare all providers for a task"""
    print("\n📊 Provider Comparison")
    print("=" * 50)
    
    providers = ["openai", "anthropic", "google", "deepseek"]
    results = {}
    
    for provider in providers:
        print(f"Testing {provider}...")
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "message": "Hello! Please respond briefly.",
                    "provider": provider
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                duration = (end_time - start_time) * 1000
                results[provider] = {
                    "success": True,
                    "model": data.get("model", "unknown"),
                    "response_time": duration,
                    "tokens": data.get("tokens_used", 0),
                    "response_length": len(data.get("response", ""))
                }
            else:
                results[provider] = {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            results[provider] = {"success": False, "error": str(e)}
    
    # Display results
    print("\n📊 Comparison Results:")
    print(f"{'Provider':<12} {'Model':<25} {'Time':<8} {'Tokens':<8} {'Status'}")
    print("-" * 65)
    
    for provider, result in results.items():
        if result["success"]:
            model = result["model"][:24]
            time_ms = f"{result['response_time']:.0f}ms"
            tokens = result["tokens"]
            status = "✅"
        else:
            model = "Failed"
            time_ms = "N/A"
            tokens = 0
            status = "❌"
        
        print(f"{provider:<12} {model:<25} {time_ms:<8} {tokens:<8} {status}")

def monitor_performance():
    """Monitor provider performance in real-time"""
    print("\n📈 Real-Time Provider Monitoring")
    print("=" * 50)
    print("Making test requests to monitor performance...")
    
    test_count = 5
    providers = ["openai", "anthropic", "deepseek"]
    
    for i in range(test_count):
        print(f"\nRound {i+1}/{test_count}")
        print("-" * 20)
        
        for provider in providers:
            try:
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8000/chat",
                    headers={"Content-Type": "application/json"},
                    json={
                        "message": f"Quick test {i+1}",
                        "provider": provider
                    },
                    timeout=15
                )
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    print(f"  {provider:10}: {duration:6.0f}ms ✅")
                else:
                    print(f"  {provider:10}: Failed ❌")
                    
            except Exception as e:
                print(f"  {provider:10}: Error ❌")
        
        if i < test_count - 1:
            time.sleep(2)  # Wait between rounds

def update_environment_config(environment, strategy):
    """Update environment-specific configuration"""
    registry_path = "configs/provider_registry.yaml"
    registry = load_provider_registry(registry_path)
    
    if not registry:
        print("❌ Cannot load registry")
        return
    
    if "environments" not in registry:
        registry["environments"] = {}
    
    if environment not in registry["environments"]:
        registry["environments"][environment] = {}
    
    registry["environments"][environment]["default_strategy"] = strategy
    
    try:
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Updated {environment} environment to use {strategy} strategy")
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="🎯 NIS Protocol Provider Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_providers.py status              # Show provider status
  python manage_providers.py routing             # Show task routing config
  python manage_providers.py test consciousness  # Test consciousness routing
  python manage_providers.py compare             # Compare all providers
  python manage_providers.py monitor             # Monitor performance
  python manage_providers.py env dev speed_first # Set dev env to speed_first
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show provider status and capabilities')
    
    # Routing command  
    subparsers.add_parser('routing', help='Show task routing configuration')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test routing for specific task')
    test_parser.add_argument('task_type', help='Task type to test (consciousness, reasoning, physics, etc.)')
    
    # Compare command
    subparsers.add_parser('compare', help='Compare all providers')
    
    # Monitor command
    subparsers.add_parser('monitor', help='Monitor provider performance')
    
    # Environment command
    env_parser = subparsers.add_parser('env', help='Update environment configuration')
    env_parser.add_argument('environment', choices=['development', 'production', 'testing'])
    env_parser.add_argument('strategy', choices=['quality_first', 'cost_performance_balance', 'speed_first', 'balanced'])
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🎯 NIS Protocol v3.2 - Provider Management")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.command == 'status':
        show_provider_status()
        
    elif args.command == 'routing':
        show_task_routing()
        
    elif args.command == 'test':
        test_routing(args.task_type)
        
    elif args.command == 'compare':
        compare_providers()
        
    elif args.command == 'monitor':
        monitor_performance()
        
    elif args.command == 'env':
        update_environment_config(args.environment, args.strategy)

if __name__ == "__main__":
    main()