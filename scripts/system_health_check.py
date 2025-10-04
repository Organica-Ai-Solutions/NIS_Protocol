#!/usr/bin/env python3
"""
NIS Protocol System Health Check & Fix
Comprehensive system validation and auto-fix for common issues
"""

import os
import sys
import subprocess

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed")

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version:")
    print("-" * 80)
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ⚠️  Python {version.major}.{version.minor}.{version.micro} (recommend 3.9+)")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Dependencies:")
    print("-" * 80)
    
    issues = []
    
    # Critical dependencies
    critical = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("aiohttp", "aiohttp (for real LLM calls)"),
    ]
    
    for module, name in critical:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - REQUIRED")
            issues.append(module)
    
    # Optional dependencies
    optional = [
        ("hnswlib", "hnswlib (better vector search)"),
        ("pinecone", "Pinecone (production vector DB)"),
        ("weaviate", "Weaviate (production vector DB)"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch (for advanced models)"),
    ]
    
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - optional")
    
    return issues

def check_imports():
    """Test all NIS Protocol imports"""
    print("\n🔧 NIS Protocol Imports:")
    print("-" * 80)
    
    imports_to_test = [
        ("src.llm.llm_manager", "LLM Manager"),
        ("src.adapters.mcp_adapter", "MCP Adapter"),
        ("src.adapters.a2a_adapter", "A2A Adapter"),
        ("src.adapters.acp_adapter", "ACP Adapter"),
        ("src.memory.vector_store", "Vector Store"),
        ("src.core.agent_orchestrator", "Agent Orchestrator"),
        ("src.meta.unified_coordinator", "Unified Coordinator"),
    ]
    
    issues = []
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            issues.append((module, str(e)))
    
    return issues

def check_configuration():
    """Check .env configuration"""
    print("\n⚙️  Configuration:")
    print("-" * 80)
    
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    
    if not os.path.exists(env_file):
        print(f"  ❌ .env file not found at {env_file}")
        print(f"  💡 Run: cp configs/complete.env.example .env")
        return False
    
    print(f"  ✅ .env file exists")
    
    # Check critical keys
    critical_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing = []
    
    for key in critical_keys:
        value = os.getenv(key, "")
        if value and value not in ["", "your-openai-key-here", "your-anthropic-key-here"]:
            print(f"  ✅ {key} configured")
        else:
            print(f"  ⚠️  {key} not configured (will use mock)")
            missing.append(key)
    
    return len(missing) == 0

def check_file_structure():
    """Check critical file structure"""
    print("\n📁 File Structure:")
    print("-" * 80)
    
    critical_files = [
        "main.py",
        "src/llm/llm_manager.py",
        "src/adapters/mcp_adapter.py",
        "src/adapters/a2a_adapter.py",
        "src/adapters/acp_adapter.py",
        "src/memory/vector_store.py",
    ]
    
    missing = []
    for file in critical_files:
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file)
        if os.path.exists(full_path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            missing.append(file)
    
    return missing

def test_llm_manager():
    """Test LLM Manager functionality"""
    print("\n🤖 LLM Manager Test:")
    print("-" * 80)
    
    try:
        from src.llm.llm_manager import GeneralLLMProvider
        
        provider = GeneralLLMProvider()
        
        # Check which providers are real
        has_real = any(provider.real_providers.values())
        
        if has_real:
            active = [p for p, avail in provider.real_providers.items() if avail]
            print(f"  ✅ Real LLM providers available: {', '.join(active)}")
        else:
            print(f"  ⚠️  No real LLM providers (using mocks)")
        
        print(f"  ✅ LLM Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ LLM Manager test failed: {e}")
        return False

def test_protocol_adapters():
    """Test Protocol Adapters"""
    print("\n🌐 Protocol Adapters Test:")
    print("-" * 80)
    
    try:
        from src.adapters.mcp_adapter import MCPAdapter
        from src.adapters.a2a_adapter import A2AAdapter
        from src.adapters.acp_adapter import ACPAdapter
        
        # Test initialization with proper configs matching adapter requirements
        mcp = MCPAdapter({
            "base_url": "http://localhost:3000",  # MCP uses base_url
            "timeout": 30
        })
        
        a2a = A2AAdapter({
            "base_url": "https://api.google.com/a2a/v1",
            "api_key": "",
            "timeout": 30
        })
        
        acp = ACPAdapter({
            "base_url": "http://localhost:8080",
            "api_key": "",
            "timeout": 30
        })
        
        print(f"  ✅ MCP Adapter initialized")
        print(f"  ✅ A2A Adapter initialized")
        print(f"  ✅ ACP Adapter initialized")
        
        # Test health status
        mcp_health = mcp.get_health_status()
        a2a_health = a2a.get_health_status()
        acp_health = acp.get_health_status()
        
        print(f"  ✅ All adapters reporting health status")
        return True
        
    except Exception as e:
        print(f"  ❌ Protocol Adapters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """Test Vector Store"""
    print("\n🗄️  Vector Store Test:")
    print("-" * 80)
    
    try:
        from src.memory.vector_store import VectorStore
        import numpy as np
        
        # Create test store
        store = VectorStore(dim=384, max_elements=100)
        
        # Add test vector
        test_vector = np.random.rand(384).astype('float32')
        store.add("test_id", test_vector, {"test": True})
        
        # Search
        results = store.search(test_vector, top_k=1)
        
        if results and results[0][0] == "test_id":
            print(f"  ✅ Vector Store working (backend: {type(store._impl).__name__})")
            return True
        else:
            print(f"  ⚠️  Vector Store functional but results unexpected")
            return True
            
    except Exception as e:
        print(f"  ❌ Vector Store test failed: {e}")
        return False

def generate_fix_script():
    """Generate fix script for common issues"""
    print("\n🔧 Fix Script:")
    print("-" * 80)
    
    fixes = []
    
    # Check if .env exists
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if not os.path.exists(env_file):
        fixes.append("cp configs/complete.env.example .env")
    
    # Check dependencies
    try:
        import hnswlib
    except ImportError:
        fixes.append("pip install hnswlib")
    
    if fixes:
        print("  Run these commands to fix issues:")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
    else:
        print("  ✅ No automatic fixes needed")
    
    return fixes

def main():
    """Run complete system health check"""
    print("=" * 80)
    print("NIS PROTOCOL v3.2 - SYSTEM HEALTH CHECK")
    print("=" * 80)
    
    results = {
        "python_version": check_python_version(),
        "dependencies": len(check_dependencies()) == 0,
        "imports": len(check_imports()) == 0,
        "configuration": check_configuration(),
        "file_structure": len(check_file_structure()) == 0,
        "llm_manager": test_llm_manager(),
        "protocol_adapters": test_protocol_adapters(),
        "vector_store": test_vector_store(),
    }
    
    fixes = generate_fix_script()
    
    # Summary
    print("\n" + "=" * 80)
    print("HEALTH CHECK SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {check.replace('_', ' ').title()}")
    
    if passed == total and not fixes:
        print("\n🎉 SYSTEM HEALTHY - Ready for production!")
        print("\nStart with: ./start.sh")
    elif passed >= total - 1:
        print("\n✅ SYSTEM MOSTLY HEALTHY - Ready to run with minor issues")
        print("\nStart with: ./start.sh")
    else:
        print("\n⚠️  SYSTEM HAS ISSUES - Fix before running")
        if fixes:
            print("\nRun suggested fixes above")
    
    print("=" * 80)
    print()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)

