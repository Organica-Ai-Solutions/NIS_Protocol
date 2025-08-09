#!/usr/bin/env python3
"""
Test Image Generation Fix - Working with your configured API keys
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

async def test_openai_image_generation():
    """Test OpenAI image generation with proper environment loading"""
    print("🎨 Testing OpenAI Image Generation Fix")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or len(openai_key) < 10:
        print("❌ OpenAI API key not properly loaded")
        return False
    
    print(f"✅ OpenAI API key loaded ({len(openai_key)} chars)")
    
    try:
        # Import and test OpenAI provider
        from src.llm.providers.openai_provider import OpenAIProvider
        
        config = {
            "api_key": openai_key,
            "model": "gpt-4"
        }
        
        provider = OpenAIProvider(config)
        print("✅ OpenAI provider initialized")
        
        # Test actual image generation
        print("\n🧪 Generating test image...")
        result = await provider.generate_image(
            prompt="A simple red circle on white background",
            size="256x256",
            quality="standard",
            num_images=1
        )
        
        if result.get("status") == "success":
            print("✅ SUCCESS! Real image generated")
            print(f"   Generated {len(result.get('images', []))} images")
            print(f"   Image data length: {len(result.get('images', [{}])[0].get('url', ''))} chars")
            
            # Save test image
            if result.get('images') and result['images'][0].get('url'):
                image_data = result['images'][0]['url']
                test_file = project_root / "static" / "generated_images" / "test_image_fix.html"
                test_file.parent.mkdir(parents=True, exist_ok=True)
                
                html_content = f"""
<!DOCTYPE html>
<html>
<head><title>NIS Image Generation Test</title></head>
<body>
    <h1>🎉 Image Generation Fixed!</h1>
    <p>Test prompt: "A simple red circle on white background"</p>
    <img src="{image_data}" alt="Generated test image" style="border: 1px solid #ccc;">
    <p>✅ Real image generation working with OpenAI DALL-E</p>
</body>
</html>
"""
                with open(test_file, 'w') as f:
                    f.write(html_content)
                
                print(f"✅ Test image saved to: {test_file}")
                return True
        else:
            print(f"❌ Image generation failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

async def main():
    """Main test function"""
    print("🔧 NIS Protocol Image Generation Fix Test")
    print("Environment properly loaded from .env file")
    print()
    
    success = await test_openai_image_generation()
    
    if success:
        print("\n🎉 SUCCESS! Image generation is now working!")
        print("🔧 Next: Apply this fix to the main application")
    else:
        print("\n❌ Test failed - need to investigate further")

if __name__ == "__main__":
    asyncio.run(main())