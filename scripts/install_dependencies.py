#!/usr/bin/env python3
"""
NIS Protocol Dependency Installer
This script installs the required dependencies for the NIS Protocol
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_step(text):
    """Print a step with formatting."""
    print(f"\n>> {text}")

def run_command(command, timeout=300):
    """Run a command and return output with timeout."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr,
            'code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': f'Command timed out after {timeout} seconds',
            'code': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'code': -1
        }

def check_venv():
    """Check if running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def install_packages():
    """Install required packages."""
    print_header("INSTALLING DEPENDENCIES")
    
    if not check_venv():
        print("[WARNING] Not running in a virtual environment!")
        print("It's recommended to use a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return False
    
    packages = [
        "requests",
        "torch",
        "numpy",
        "scipy",
        "python-dotenv"
    ]
    
    for package in packages:
        print_step(f"Installing {package}")
        
        if package == "torch":
            # Use specific torch command based on platform
            if platform.system() == "Windows":
                cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            else:
                cmd = f"{sys.executable} -m pip install torch torchvision torchaudio"
        else:
            cmd = f"{sys.executable} -m pip install {package}"
            
        result = run_command(cmd)
        
        if result['success']:
            print(f"[OK] {package} installed successfully")
        else:
            print(f"[ERROR] Failed to install {package}")
            print(f"Error: {result['error']}")
            return False
    
    return True

def main():
    """Main function."""
    print_header("NIS PROTOCOL DEPENDENCY INSTALLER")
    print("This script will install the required dependencies for the NIS Protocol.")
    print("\nPackages to be installed:")
    print("- requests: For API calls")
    print("- torch: PyTorch for machine learning")
    print("- numpy: Numerical computing")
    print("- scipy: Scientific computing")
    print("- python-dotenv: Environment variable management")
    
    response = input("\nProceed with installation? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    success = install_packages()
    
    print_header("INSTALLATION COMPLETE")
    if success:
        print("\nAll dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Run the fix_terminal.py script again")
        print("2. Try the NVIDIA API test")
    else:
        print("\nSome dependencies failed to install.")
        print("Please check the error messages above and try again.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()