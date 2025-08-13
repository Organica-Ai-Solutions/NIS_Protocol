#!/usr/bin/env python3
# Script to generate PDF version of the NIS Protocol whitepaper
# This script uses WeasyPrint to convert HTML to PDF

import os
from weasyprint import HTML, CSS
from pathlib import Path

def generate_whitepaper_pdf():
    """Generate PDF from the whitepaper HTML file."""
    
    # Get the base directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir.parent
    
    # Input HTML file
    html_file = current_dir / "NIS_Protocol_Whitepaper.html"
    
    # Output PDF file
    pdf_file = current_dir / "finalwhitepaper.pdf"
    
    print(f"Converting {html_file} to PDF...")
    
    # Base URL for resolving relative URLs in the HTML file
    base_url = f"file://{base_dir}/"
    
    # Create PDF
    HTML(filename=str(html_file), base_url=base_url).write_pdf(
        pdf_file,
        # Additional CSS styles for PDF output can be added here
        stylesheets=[
            # CSS(string="@page { size: letter; margin: 1cm; }")
        ]
    )
    
    print(f"PDF generated successfully: {pdf_file}")
    
if __name__ == "__main__":
    generate_whitepaper_pdf() 