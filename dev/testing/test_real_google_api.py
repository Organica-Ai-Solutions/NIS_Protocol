#!/usr/bin/env python3
"""
Test REAL Google Gemini 2.0 Image Generation API
Following official Google documentation exactly
"""
import asyncio
import base64
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, '.')
sys.path.insert(0, './src')

async def test_real_google_gemini_api():
    """Test the real Google Gemini 2.0 API exactly as documented"""
    print("🎨 Testing REAL Google Gemini 2.0 Image Generation API...")
    
    api_key = "AIzaSyBTrH6g_AfGO43fzgTz21S94X6coPVI8tk"
    
    try:
        # Import exactly as shown in Google's documentation
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        
        print("✅ Successfully imported google.genai modules")
        
        # Create client exactly as in documentation
        client = genai.Client(api_key=api_key)
        print("✅ Created Gemini client")
        
        # Test prompt
        contents = "Create a beautiful dragon flying over a cyberpunk city at sunset"
        
        print(f"🎨 Generating image with prompt: {contents}")
        
        # Generate content exactly as in documentation
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        print("✅ API call successful! Processing response...")
        
        # Extract images and text exactly as in documentation
        generated_images = []
        response_text = ""
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                response_text = part.text
                print(f"📝 Response text: {response_text[:100]}...")
            elif part.inline_data is not None:
                # Convert image data to base64 data URL
                image_data = base64.b64encode(part.inline_data.data).decode()
                data_url = f"data:image/png;base64,{image_data}"
                
                generated_images.append({
                    "url": data_url,
                    "size": len(image_data),
                    "format": "png"
                })
                
                print(f"🖼️ Generated image! Size: {len(image_data)} bytes")
                
                # Save image like in Google docs
                image = Image.open(BytesIO(part.inline_data.data))
                image.save('dev/testing/gemini_real_test.png')
                print("💾 Saved image as gemini_real_test.png")
        
        if generated_images:
            print("🎉 SUCCESS! Real Google Gemini 2.0 API working perfectly!")
            print(f"📊 Generated {len(generated_images)} images")
            return True
        else:
            print("❌ No images generated")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_google_gemini_api())
    if success:
        print("\n🚀 Google Gemini 2.0 API is working! Real image generation confirmed!")
    else:
        print("\n⚠️  Real API not working, using enhanced placeholders")