#!/usr/bin/env python3
"""
AWS Interview PDF Generator
Creates a professional PDF from the migration presentation
"""

import os
import subprocess
import sys
from pathlib import Path

def create_aws_interview_pdf():
    """Convert the AWS interview presentation to PDF"""
    
    print("🎯 AWS Interview PDF Generator")
    print("=" * 40)
    
    # File paths
    markdown_file = "AWS_INTERVIEW_MIGRATION_PRESENTATION.md"
    pdf_file = "AWS_INTERVIEW_MIGRATION_PRESENTATION.pdf"
    
    if not os.path.exists(markdown_file):
        print(f"❌ Error: {markdown_file} not found")
        return False
    
    print(f"📄 Converting {markdown_file} to PDF...")
    
    # Method 1: Try pandoc (if available)
    try:
        cmd = [
            'pandoc',
            markdown_file,
            '-o', pdf_file,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'documentclass=article',
            '--highlight-style=github',
            '--table-of-contents',
            '--number-sections'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF created successfully: {pdf_file}")
            print(f"📊 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
            return True
        else:
            print(f"⚠️ Pandoc failed: {result.stderr}")
            
    except FileNotFoundError:
        print("⚠️ Pandoc not found, trying alternative method...")
    
    # Method 2: Try markdown-pdf (if available)
    try:
        cmd = ['markdown-pdf', markdown_file, '-o', pdf_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF created successfully: {pdf_file}")
            return True
        else:
            print(f"⚠️ markdown-pdf failed: {result.stderr}")
            
    except FileNotFoundError:
        print("⚠️ markdown-pdf not found")
    
    # Method 3: Create HTML version for manual conversion
    print("📝 Creating HTML version for manual PDF conversion...")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NIS Protocol AWS Migration Plan</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 40px;
            background: #fff;
        }}
        h1 {{
            color: #1e88e5;
            border-bottom: 3px solid #1e88e5;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0277bd;
            margin-top: 30px;
        }}
        h3 {{
            color: #0288d1;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #1e88e5;
            padding-left: 20px;
            margin: 20px 0;
            font-style: italic;
        }}
        .toc {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        @media print {{
            body {{ margin: 20px; }}
            h1 {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
"""
    
    # Read markdown content and convert to HTML
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple markdown to HTML conversion
        content = content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        content = content.replace('## ', '<h2>').replace('\n', '</h2>\n')
        content = content.replace('### ', '<h3>').replace('\n', '</h3>\n')
        content = content.replace('**', '<strong>').replace('**', '</strong>')
        content = content.replace('`', '<code>').replace('`', '</code>')
        content = content.replace('\n\n', '</p><p>')
        content = f"<p>{content}</p>"
        
        html_content += content + """
</body>
</html>"""
        
        html_file = "AWS_INTERVIEW_MIGRATION_PRESENTATION.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML created: {html_file}")
        print("📝 Manual PDF creation options:")
        print("   1. Open HTML file in browser and print to PDF")
        print("   2. Use online HTML to PDF converter")
        print("   3. Install pandoc: https://pandoc.org/installing.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating HTML: {e}")
        return False

def main():
    """Main function"""
    
    success = create_aws_interview_pdf()
    
    if success:
        print("\n🎊 AWS Interview Materials Ready!")
        print("📋 Files created:")
        
        files = [
            "AWS_INTERVIEW_MIGRATION_PRESENTATION.md",
            "AWS_INTERVIEW_MIGRATION_PRESENTATION.html"
        ]
        
        pdf_file = "AWS_INTERVIEW_MIGRATION_PRESENTATION.pdf"
        if os.path.exists(pdf_file):
            files.append(pdf_file)
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024
                print(f"   ✅ {file} ({size:.1f} KB)")
        
        print("\n🚀 Ready for AWS interview!")
        print("💡 Bring the PDF to showcase your migration plan")
        
    else:
        print("\n❌ PDF creation failed")
        print("💡 Use the markdown file directly or convert manually")

if __name__ == "__main__":
    main()