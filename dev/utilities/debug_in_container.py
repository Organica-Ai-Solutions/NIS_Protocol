#!/usr/bin/env python3
"""
Debug script to run inside Docker container
"""
import traceback

try:
    print("🔍 Testing Google imports...")
    
    # Test the exact imports from our code
    from google import genai
    from google.genai import types
    print("✅ Imports successful")
    
    # Test client creation
    print("🔍 Testing client creation...")
    client = genai.Client()
    print("✅ Client creation successful")
    
    # Test with a fake API key (just to see if it gets to the API call)
    print("🔍 Testing configuration...")
    genai.configure(api_key="fake_key_for_test")
    print("✅ Configuration successful")
    
    print("🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("🔍 Full traceback:")
    traceback.print_exc()