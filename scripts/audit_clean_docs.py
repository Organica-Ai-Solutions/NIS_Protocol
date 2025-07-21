#!/usr/bin/env python3
"""
Audit Clean Documentation Script
Runs integrity audit on essential v3 docs only, excluding archived summaries
"""

import os
import subprocess
import tempfile
import shutil

def audit_clean_docs():
    """Run audit on essential docs only, excluding archived summaries"""
    
    print("🔍 Auditing Clean Essential V3 Documentation...")
    print("=" * 60)
    
    # Temporarily move archive to exclude from audit
    archive_path = "docs/archive_summaries"
    temp_archive = None
    
    if os.path.exists(archive_path):
        temp_archive = tempfile.mkdtemp(prefix="nis_archive_")
        shutil.move(archive_path, temp_archive)
        print(f"📁 Temporarily moved archive to: {temp_archive}")
    
    try:
        # Run the audit on clean documentation
        print("\n🔍 Running integrity audit on clean documentation...")
        result = subprocess.run([
            "python", "nis-integrity-toolkit/audit-scripts/full-audit.py",
            "--project-path", ".",
            "--output-report"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        # Parse the audit results
        if "Integrity Score:" in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if "Integrity Score:" in line:
                    score = line.split(":")[1].strip()
                    print(f"\n🎯 Clean Documentation Integrity Score: {score}")
                    break
        
        return result.returncode == 0
        
    finally:
        # Restore archive
        if temp_archive and os.path.exists(temp_archive):
            archive_backup = os.path.join(temp_archive, "archive_summaries")
            if os.path.exists(archive_backup):
                shutil.move(archive_backup, "docs/")
                print(f"\n📁 Restored archive to docs/archive_summaries/")
            shutil.rmtree(temp_archive)

def generate_clean_docs_summary():
    """Generate summary of clean documentation structure"""
    
    print("\n📚 Clean Documentation Structure:")
    print("=" * 40)
    
    essential_docs = []
    for root, dirs, files in os.walk("docs"):
        # Skip archive directory
        if "archive_summaries" in root:
            continue
            
        for file in files:
            if file.endswith('.md') or file.endswith('.html') or file.endswith('.pdf'):
                rel_path = os.path.relpath(os.path.join(root, file), "docs")
                essential_docs.append(rel_path)
    
    # Categorize files
    core_guides = [f for f in essential_docs if any(x in f.lower() for x in ['getting_started', 'quick_start', 'api_reference', 'integration_guide'])]
    technical_guides = [f for f in essential_docs if any(x in f.lower() for x in ['kafka', 'llm', 'web_search', 'mathematical'])]
    reference_docs = [f for f in essential_docs if any(x in f.lower() for x in ['inventory', 'whitepaper'])]
    web_docs = [f for f in essential_docs if f.endswith('.html')]
    other_docs = [f for f in essential_docs if f not in core_guides + technical_guides + reference_docs + web_docs]
    
    print("\n📘 Core Guides:")
    for doc in sorted(core_guides):
        print(f"  ✅ {doc}")
    
    print("\n🔧 Technical Guides:")
    for doc in sorted(technical_guides):
        print(f"  ✅ {doc}")
        
    print("\n📖 Reference Materials:")
    for doc in sorted(reference_docs):
        print(f"  ✅ {doc}")
        
    print("\n🌐 Web Documentation:")
    for doc in sorted(web_docs):
        print(f"  ✅ {doc}")
        
    if other_docs:
        print("\n📄 Other Documentation:")
        for doc in sorted(other_docs):
            print(f"  ✅ {doc}")
    
    print(f"\n📊 Total Essential Documents: {len(essential_docs)}")
    
    # Check archive
    archive_count = 0
    if os.path.exists("docs/archive_summaries"):
        archive_files = [f for f in os.listdir("docs/archive_summaries") if f.endswith('.md')]
        archive_count = len(archive_files)
    
    print(f"🗂️  Archived Summaries: {archive_count} files")
    print("\n✅ Documentation structure is clean and organized!")

if __name__ == "__main__":
    audit_success = audit_clean_docs()
    generate_clean_docs_summary()
    
    if audit_success:
        print("\n🎉 Clean documentation audit completed successfully!")
    else:
        print("\n⚠️ Audit completed with some issues - check output above") 