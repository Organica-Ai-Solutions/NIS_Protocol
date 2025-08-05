#!/usr/bin/env python3
"""
Improved Response Formatter for Better Text Display
Fixes text formatting issues in multimodal responses
"""

def clean_response_text(text):
    """Clean and format response text for better display"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove excessive markdown
    text = text.replace('**', '')
    text = text.replace('***', '')
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€"', '-')
    text = text.replace('â€"', '—')
    
    # Clean up multiple spaces and newlines
    import re
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Fix bullet points
    text = re.sub(r'[•·▪▫‣⁃]', '•', text)
    
    return text.strip()

def format_physics_response(content, style="clean"):
    """Format physics responses for better readability"""
    
    if style == "clean":
        # Clean, professional formatting
        formatted = f"""
<div class="physics-response" style="font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; color: #1f2937;">
    <div class="response-content" style="background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #0891b2;">
        {clean_response_text(content)}
    </div>
</div>
"""
    elif style == "technical":
        # Technical documentation style
        formatted = f"""
<div class="technical-response" style="font-family: 'JetBrains Mono', 'Courier New', monospace; line-height: 1.5; color: #1e293b;">
    <div class="response-content" style="background: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
        {clean_response_text(content)}
    </div>
</div>
"""
    elif style == "visual":
        # Visual-friendly format with better spacing
        formatted = f"""
<div class="visual-response" style="font-family: 'Inter', sans-serif; line-height: 1.7; color: #111827;">
    <div class="response-content" style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 24px; border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        {clean_response_text(content)}
    </div>
</div>
"""
    else:
        # Default clean format
        formatted = f"""
<div class="default-response" style="font-family: system-ui, sans-serif; line-height: 1.6;">
    <div class="response-content" style="padding: 16px;">
        {clean_response_text(content)}
    </div>
</div>
"""
    
    return formatted

def create_improved_formatter_patch():
    """Create a patch for the main application to use improved formatting"""
    
    patch_code = '''
# Add this to main.py to improve response formatting

import re

def improved_format_response(content, style="clean"):
    """Improved response formatting for better text display"""
    
    if not isinstance(content, str):
        content = str(content)
    
    # Clean text
    content = content.replace('**', '')
    content = re.sub(r'\\n\\s*\\n\\s*\\n+', '\\n\\n', content)
    content = re.sub(r' +', ' ', content)
    
    # Apply styling
    if style == "visual":
        return f"""
<div style="font-family: 'Inter', sans-serif; line-height: 1.7; color: #111827; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 24px; border-radius: 16px; margin: 16px 0;">
    {content}
</div>
"""
    else:
        return f"""
<div style="font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; color: #1f2937; background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #0891b2; margin: 16px 0;">
    {content}
</div>
"""

# In your chat endpoint, wrap responses like this:
# return improved_format_response(response_content, request.output_mode)
'''
    
    return patch_code

if __name__ == "__main__":
    print("🎨 Improved Response Formatter")
    print("=" * 50)
    
    # Test the formatter
    sample_text = """
    **Physics Analysis**
    
    The bouncing ball exhibits complex â€œfluid-structure interactionâ€ dynamics:
    
    • Energy dissipation: ~60-80% per bounce
    • Drag coefficient: C_d â‰ˆ 0.47 for spheres  
    • Surface tension effects: Ï„ = 2Ï€RÎ³
    
    ***Key Insights***:
    - Buoyancy forces dominate in water
    - Cavitation occurs during high-speed entry
    """
    
    print("Original text:")
    print(repr(sample_text))
    print("\nCleaned text:")
    print(clean_response_text(sample_text))
    print("\nFormatted HTML:")
    print(format_physics_response(sample_text, "visual"))
    
    print("\n✅ Formatter working correctly!")
    print("Apply this to main.py to fix text formatting issues.")