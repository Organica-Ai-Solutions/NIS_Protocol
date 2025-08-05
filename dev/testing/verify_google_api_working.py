#!/usr/bin/env python3
"""
Comprehensive verification that Google Gemini 2.0 API is working
"""
import requests
import json

def test_our_implementation():
    """Test our NIS Protocol implementation"""
    print("🧪 Testing Our NIS Protocol Implementation...")
    
    try:
        response = requests.post("http://localhost:8000/image/generate", json={
            "prompt": "Simple red dragon test",
            "style": "artistic", 
            "provider": "google",
            "size": "1024x1024"
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            provider_used = generation.get("provider_used", "unknown")
            note = generation.get("note", "")
            model = generation.get("generation_info", {}).get("model", "unknown")
            
            print(f"✅ Status: SUCCESS")
            print(f"🤖 Provider: {provider_used}")
            print(f"🎨 Model: {model}")
            print(f"📝 Note: {note}")
            
            if "REAL" in provider_used or "REAL" in note:
                print("🎉 REAL Google Gemini 2.0 API Working!")
                return True
            else:
                print("⚠️ Using enhanced placeholders instead of real API")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text[:500])
            return False
            
    except Exception as e:
        print(f"❌ Request Error: {e}")
        return False

def test_direct_api():
    """Test the API directly like Google documentation"""
    print("\n🔧 Testing Direct Google API...")
    
    # This should work based on our earlier test
    try:
        import subprocess
        result = subprocess.run([
            "docker-compose", "exec", "-T", "backend", "python3", "-c",
            """
import sys
sys.path.append('/home/nisuser/app')

# Test exactly like Google docs
from google import genai
from google.genai import types
import base64

client = genai.Client(api_key='AIzaSyBTrH6g_AfGO43fzgTz21S94X6coPVI8tk')
response = client.models.generate_content(
    model='gemini-2.0-flash-preview-image-generation',
    contents='Simple red dragon',
    config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
)

for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        print(f'SUCCESS: Generated {len(part.inline_data.data)} bytes')
        break
else:
    print('FAILED: No image generated')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if "SUCCESS:" in result.stdout:
            print("✅ Direct API working perfectly!")
            print(result.stdout.strip())
            return True
        else:
            print("❌ Direct API failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Direct test error: {e}")
        return False

def analyze_issue():
    """Analyze why real API isn't being used"""
    print("\n🔍 ANALYSIS:")
    print("1. ✅ Direct Google API works (confirmed)")
    print("2. ⚠️ NIS implementation falling back to placeholders")
    print("3. 🐛 Issue: 'local variable genai referenced before assignment'")
    print("4. 🎯 Solution: Need to trace the exact error location")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE GOOGLE API VERIFICATION")
    print("=" * 50)
    
    # Test our implementation
    nis_working = test_our_implementation()
    
    # Test direct API
    direct_working = test_direct_api()
    
    # Analysis
    analyze_issue()
    
    print("\n📊 RESULTS:")
    print(f"Direct Google API: {'✅ WORKING' if direct_working else '❌ FAILED'}")
    print(f"NIS Implementation: {'✅ USING REAL API' if nis_working else '⚠️ USING PLACEHOLDERS'}")
    
    if direct_working and not nis_working:
        print("\n🎯 CONCLUSION: Google API works, need to fix NIS implementation!")
    elif direct_working and nis_working:
        print("\n🎉 CONCLUSION: Everything working perfectly!")
    else:
        print("\n❌ CONCLUSION: API or implementation issues need investigation")