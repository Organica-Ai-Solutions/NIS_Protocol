#!/usr/bin/env python3
"""
NIS Protocol v3.0 - PyPI Publishing Script

This script helps publish the NIS Protocol package to PyPI test and production.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result


def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning build artifacts...")
    run_command("rm -rf dist/ build/ *.egg-info/", check=False)


def build_package():
    """Build the package"""
    print("🔨 Building package...")
    run_command("python -m build")


def check_package():
    """Check package integrity"""
    print("🔍 Checking package...")
    run_command("twine check dist/*")


def upload_to_testpypi():
    """Upload to PyPI test"""
    print("🚀 Uploading to PyPI test...")
    run_command("twine upload --repository testpypi dist/*")


def upload_to_pypi():
    """Upload to production PyPI"""
    print("🚀 Uploading to production PyPI...")
    run_command("twine upload dist/*")


def verify_installation(test=True):
    """Verify the package can be installed"""
    repo = "test.pypi.org" if test else "pypi.org"
    print(f"🧪 Testing installation from {repo}...")
    
    # Create a temporary virtual environment
    run_command("python -m venv temp_test_env")
    
    if test:
        install_cmd = "temp_test_env/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nis-protocol-v3"
    else:
        install_cmd = "temp_test_env/bin/pip install nis-protocol-v3"
    
    try:
        run_command(install_cmd)
        print("✅ Package installed successfully!")
        
        # Test basic import
        test_import = "temp_test_env/bin/python -c 'import core.agent; print(\"Import successful!\")'"
        run_command(test_import)
        print("✅ Package imports successfully!")
        
    except:
        print("❌ Package installation/import failed")
    finally:
        run_command("rm -rf temp_test_env", check=False)


def main():
    parser = argparse.ArgumentParser(description="Publish NIS Protocol to PyPI")
    parser.add_argument("--test", action="store_true", help="Publish to PyPI test")
    parser.add_argument("--prod", action="store_true", help="Publish to production PyPI")
    parser.add_argument("--build-only", action="store_true", help="Only build, don't upload")
    parser.add_argument("--verify", action="store_true", help="Verify installation after upload")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build()
        return
    
    # Default to test if no target specified
    if not args.test and not args.prod and not args.build_only:
        args.test = True
    
    print("🎯 NIS Protocol v3.0 - PyPI Publishing")
    print("=" * 50)
    
    # Check we're in the right directory
    if not Path("setup.py").exists():
        print("❌ Error: setup.py not found. Run this from the project root.")
        sys.exit(1)
    
    # Clean and build
    clean_build()
    build_package()
    check_package()
    
    if args.build_only:
        print("✅ Build complete!")
        print("\nBuilt files:")
        run_command("ls -la dist/")
        return
    
    # Upload
    if args.test:
        try:
            upload_to_testpypi()
            print("✅ Successfully uploaded to PyPI test!")
            print("🔗 View at: https://test.pypi.org/project/nis-protocol-v3/")
            
            if args.verify:
                verify_installation(test=True)
                
        except Exception as e:
            print(f"❌ Upload to PyPI test failed: {e}")
            print("\n💡 Setup instructions:")
            print("1. Create account at https://test.pypi.org/account/register/")
            print("2. Generate API token at https://test.pypi.org/manage/account/token/")
            print("3. Edit ~/.pypirc with your token")
            
    if args.prod:
        try:
            upload_to_pypi()
            print("✅ Successfully uploaded to production PyPI!")
            print("🔗 View at: https://pypi.org/project/nis-protocol-v3/")
            
            if args.verify:
                verify_installation(test=False)
                
        except Exception as e:
            print(f"❌ Upload to production PyPI failed: {e}")
    
    print("\n🎉 Publishing workflow complete!")
    
    # Provide usage instructions
    print("\n📦 Installation instructions:")
    if args.test:
        print("pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nis-protocol-v3")
    if args.prod:
        print("pip install nis-protocol-v3")


if __name__ == "__main__":
    main() 