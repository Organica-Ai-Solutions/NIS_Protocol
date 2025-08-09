#!/usr/bin/env python3
"""
Diagnose Image Generation Issues
Checks configuration, imports, and API availability
"""
import os
import sys

def check_environment():
    """Check environment variables"""
    print("🔍 Environment Variables:")
    print(f"  GOOGLE_API_KEY: {'SET' if os.getenv('GOOGLE_API_KEY') and len(os.getenv('GOOGLE_API_KEY', '')) > 10 else 'NOT_SET'}")
    print(f"  GCP_PROJECT_ID: {os.getenv('GCP_PROJECT_ID', 'NOT_SET')}")
    print(f"  OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') and len(os.getenv('OPENAI_API_KEY', '')) > 10 else 'NOT_SET'}")
    print()

def check_imports():
    """Check required library imports"""
    print("📦 Import Status:")
    
    # Check Google imports
    try:
        import google.generativeai as genai
        print("  ✅ google.generativeai: AVAILABLE")
    except ImportError as e:
        print(f"  ❌ google.generativeai: NOT_AVAILABLE - {e}")
    
    try:
        from vertexai.preview.vision_models import ImageGenerationModel
        print("  ✅ vertexai.preview.vision_models: AVAILABLE")
    except ImportError as e:
        print(f"  ❌ vertexai.preview.vision_models: NOT_AVAILABLE - {e}")
    
    try:
        from google.cloud import aiplatform
        print("  ✅ google.cloud.aiplatform: AVAILABLE")
    except ImportError as e:
        print(f"  ❌ google.cloud.aiplatform: NOT_AVAILABLE - {e}")
    
    # Check OpenAI imports
    try:
        import openai
        print("  ✅ openai: AVAILABLE")
    except ImportError as e:
        print(f"  ❌ openai: NOT_AVAILABLE - {e}")
    
    # Check PIL
    try:
        from PIL import Image
        print("  ✅ PIL: AVAILABLE")
    except ImportError as e:
        print(f"  ❌ PIL: NOT_AVAILABLE - {e}")
    print()

def check_configuration_files():
    """Check configuration files"""
    print("📁 Configuration Files:")
    
    service_account_path = "configs/google-service-account.json"
    if os.path.exists(service_account_path):
        print(f"  ✅ {service_account_path}: EXISTS")
    else:
        print(f"  ❌ {service_account_path}: MISSING")
    
    env_file_path = ".env"
    if os.path.exists(env_file_path):
        print(f"  ✅ {env_file_path}: EXISTS")
    else:
        print(f"  ❌ {env_file_path}: MISSING")
    print()

def recommend_fixes():
    """Provide fix recommendations"""
    print("🔧 Recommended Fixes:")
    print("1. Set up environment variables in .env file:")
    print("   GOOGLE_API_KEY=your_actual_google_api_key")
    print("   GCP_PROJECT_ID=your_project_id")
    print("   OPENAI_API_KEY=your_openai_api_key")
    print()
    print("2. Install required packages:")
    print("   pip install google-cloud-aiplatform")
    print("   pip install vertexai")
    print("   pip install google-generativeai")
    print("   pip install openai")
    print("   pip install pillow")
    print()
    print("3. Configure Google Cloud service account:")
    print("   Copy configs/google-service-account.json.example")
    print("   to configs/google-service-account.json")
    print("   Fill in actual credentials")
    print()

def test_providers():
    """Test basic provider functionality"""
    print("🧪 Provider Tests:")
    
    # Test Google Provider
    try:
        sys.path.append('.')
        from src.llm.providers.google_provider import GoogleProvider
        
        config = {
            "api_key": os.getenv("GOOGLE_API_KEY", "test"),
            "model": "gemini-1.5-flash"
        }
        
        provider = GoogleProvider(config)
        print("  ✅ GoogleProvider: INSTANTIATED")
        
        # Check what conditions are failing
        if not provider.gcp_project_id:
            print("  ❌ GoogleProvider: Missing GCP_PROJECT_ID")
        else:
            print(f"  ✅ GoogleProvider: GCP_PROJECT_ID = {provider.gcp_project_id}")
            
    except Exception as e:
        print(f"  ❌ GoogleProvider: FAILED - {e}")
    
    # Test OpenAI Provider
    try:
        from src.llm.providers.openai_provider import OpenAIProvider
        
        config = {
            "api_key": os.getenv("OPENAI_API_KEY", "test"),
            "model": "gpt-4"
        }
        
        provider = OpenAIProvider(config)
        print("  ✅ OpenAIProvider: INSTANTIATED")
        
    except Exception as e:
        print(f"  ❌ OpenAIProvider: FAILED - {e}")
    
    print()

if __name__ == "__main__":
    print("🎨 NIS Protocol Image Generation Diagnostics")
    print("=" * 50)
    print()
    
    check_environment()
    check_imports()
    check_configuration_files()
    test_providers()
    recommend_fixes()