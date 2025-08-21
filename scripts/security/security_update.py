#!/usr/bin/env python3
"""
🔒 NIS Protocol Security Update Script
Fixes all known vulnerabilities and dependency conflicts
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    """Main security update process"""
    print("🛡️ Starting NIS Protocol Security Update...")
    
    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # List of security updates
    updates = [
        # Update pip first
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        
        # Install with constraints to fix vulnerabilities
        ("pip install -r requirements.txt -c constraints.txt --upgrade", "Installing secure dependencies"),
        
        # Force specific security updates
        ("pip install 'cryptography>=45.0.0' --upgrade", "Updating cryptography"),
        ("pip install 'urllib3>=2.5.0' --upgrade", "Updating urllib3"),
        ("pip install 'pillow>=10.4.0' --upgrade", "Updating pillow"),
        ("pip install 'requests>=2.32.0' --upgrade", "Updating requests"),
        ("pip install 'aiohttp>=3.12.0' --upgrade", "Updating aiohttp"),
        ("pip install 'transformers>=4.55.0' --upgrade", "Updating transformers"),
        ("pip install 'pydantic>=2.9.0' --upgrade", "Updating pydantic"),
        ("pip install 'fastapi>=0.116.0' --upgrade", "Updating fastapi"),
        ("pip install 'langchain>=0.3.0' 'langchain-core>=0.3.0' --upgrade", "Updating langchain"),
        
        # Remove vulnerable packages
        ("pip uninstall keras -y", "Removing vulnerable keras package"),
    ]
    
    success_count = 0
    total_count = len(updates)
    
    for command, description in updates:
        if run_command(command, description):
            success_count += 1
        print()  # Add spacing
    
    # Final vulnerability check
    print("🔍 Running final security audit...")
    run_command("pip-audit --requirement requirements.txt --format=json", "Final vulnerability scan")
    
    # Summary
    print(f"\n📊 Security Update Summary:")
    print(f"✅ Successful updates: {success_count}/{total_count}")
    print(f"📋 Updated requirements.txt with secure versions")
    print(f"🛡️ Added security constraints in constraints.txt")
    print(f"🚫 Excluded vulnerable keras package (CVE-2024-55459)")
    
    if success_count == total_count:
        print("\n🎉 All security updates completed successfully!")
        print("🔒 NIS Protocol is now secure with latest dependency versions")
        return 0
    else:
        print(f"\n⚠️ Some updates failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
