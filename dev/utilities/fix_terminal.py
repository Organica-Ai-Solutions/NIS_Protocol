#!/usr/bin/env python3
"""
Terminal Diagnostic and Fix Script
Run this directly from your file explorer by double-clicking
"""

import os
import sys
import platform
import subprocess
import time
import io

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_step(text):
    """Print a step with formatting."""
    print(f"\n>> {text}")

def run_command(command, timeout=10):
    """Run a command and return output with timeout."""
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

def diagnose_system():
    """Diagnose system environment."""
    print_header("SYSTEM DIAGNOSTICS")
    
    print_step("Checking operating system")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    print_step("Checking environment variables")
    path = os.environ.get('PATH', '')
    print(f"PATH length: {len(path)} characters")
    
    home = os.environ.get('HOME', os.environ.get('USERPROFILE', ''))
    print(f"HOME/USERPROFILE: {home}")
    
    print_step("Checking current directory")
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    
    print_step("Checking file permissions")
    try:
        test_file = "terminal_test.txt"
        with open(test_file, "w") as f:
            f.write("Test file write")
        os.remove(test_file)
        print("[OK] File write/delete: SUCCESS")
    except Exception as e:
        print(f"[ERROR] File write/delete: FAILED - {e}")

def test_shells():
    """Test different shells."""
    print_header("SHELL TESTS")
    
    shells = [
        {"name": "CMD", "command": "cmd /c echo CMD test successful"},
        {"name": "PowerShell", "command": "powershell -Command \"Write-Host 'PowerShell test successful'\""},
        {"name": "Python", "command": f"{sys.executable} -c \"print('Python subprocess test successful')\""}
    ]
    
    # Try to detect Git Bash
    git_bash = os.environ.get('PROGRAMFILES', '') + "\\Git\\bin\\bash.exe"
    if os.path.exists(git_bash):
        shells.append({"name": "Git Bash", "command": f"\"{git_bash}\" -c \"echo 'Git Bash test successful'\""})
    
    for shell in shells:
        print_step(f"Testing {shell['name']}")
        result = run_command(shell['command'])
        
        if result['success']:
            print(f"[OK] {shell['name']}: SUCCESS")
            print(f"Output: {result['output'].strip()}")
        else:
            print(f"[ERROR] {shell['name']}: FAILED")
            print(f"Error: {result['error']}")
            print(f"Return code: {result['code']}")

def check_python_packages():
    """Check Python packages."""
    print_header("PYTHON PACKAGE CHECK")
    
    packages = ["requests", "json", "torch", "numpy", "scipy"]
    
    for package in packages:
        print_step(f"Checking {package}")
        result = run_command(f"{sys.executable} -c \"import {package}; print({package}.__version__)\"")
        
        if result['success']:
            print(f"[OK] {package}: INSTALLED - {result['output'].strip()}")
        else:
            print(f"[ERROR] {package}: FAILED - {result['error']}")

def test_nvidia_api():
    """Test NVIDIA API connection."""
    print_header("NVIDIA API TEST")
    
    print_step("Checking for NVIDIA API key")
    api_key = None
    
    # Try to load from .env file
    env_files = ['.env', '.env~', '.env.local']
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"Found {env_file} file")
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('NVIDIA_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            print(f"[OK] NVIDIA API key found in {env_file}: {api_key[:10]}...")
                            break
            except Exception as e:
                print(f"[ERROR] Error reading {env_file}: {e}")
    
    if not api_key:
        print("[ERROR] NVIDIA API key not found in environment files")
        return
    
    print_step("Testing quick_nvidia_test.py existence")
    if os.path.exists('quick_nvidia_test.py'):
        print("[OK] quick_nvidia_test.py found")
    else:
        print("[ERROR] quick_nvidia_test.py not found")
        return
    
    print_step("Testing NVIDIA API connection")
    print("This may take a minute...")
    
    # Create a minimal test script
    test_script = """
import os
import requests
import json
import time

def test():
    api_key = None
    env_files = ['.env', '.env~', '.env.local']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('NVIDIA_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except:
                pass
    
    if not api_key:
        print("[ERROR] NVIDIA API key not found")
        return
    
    print(f"[OK] API key loaded: {api_key[:10]}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            print(f"[OK] Connected! Found {len(models.get('data', []))} models")
            return True
        else:
            print(f"[ERROR] Connection failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    test()
"""
    
    # Write the test script using UTF-8 encoding
    with io.open('nvidia_quick_test_temp.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    # Run the test script
    result = run_command(f"{sys.executable} nvidia_quick_test_temp.py", timeout=30)
    
    if result['success']:
        print(f"Output: {result['output'].strip()}")
        if "Connected!" in result['output']:
            print("[OK] NVIDIA API connection SUCCESSFUL!")
        else:
            print("[ERROR] NVIDIA API connection FAILED")
    else:
        print(f"[ERROR] Test execution failed: {result['error']}")
    
    # Clean up
    try:
        os.remove('nvidia_quick_test_temp.py')
    except:
        pass

def fix_terminal_issues():
    """Attempt to fix terminal issues."""
    print_header("TERMINAL FIXES")
    
    print_step("Creating .bat launcher for quick_nvidia_test.py")
    
    # Create a batch file to run the NVIDIA test
    bat_content = f"""@echo off
echo Running NVIDIA API Test...
"{sys.executable}" "%~dp0quick_nvidia_test.py"
echo.
echo Test complete! Press any key to exit...
pause > nul
"""
    
    with io.open('run_nvidia_test.bat', 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print("[OK] Created run_nvidia_test.bat")
    print("You can double-click this file to run the NVIDIA test")
    
    print_step("Creating PowerShell launcher")
    
    # Create a PowerShell script to run the NVIDIA test
    ps_content = f"""
Write-Host "Running NVIDIA API Test..." -ForegroundColor Cyan
& '{sys.executable}' "$PSScriptRoot\\quick_nvidia_test.py"
Write-Host ""
Write-Host "Test complete! Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"""
    
    with io.open('run_nvidia_test.ps1', 'w', encoding='utf-8') as f:
        f.write(ps_content)
    
    print("[OK] Created run_nvidia_test.ps1")
    print("You can right-click this file and select 'Run with PowerShell'")
    
    print_step("Creating Python launcher")
    
    # Create a Python launcher script
    py_launcher = """#!/usr/bin/env python3
import os
import sys
import subprocess

print("Running NVIDIA API Test...")
print("=" * 50)

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
test_script = os.path.join(script_dir, "quick_nvidia_test.py")

# Run the test script
try:
    subprocess.run([sys.executable, test_script], check=True)
    print("=" * 50)
    print("Test complete!")
except Exception as e:
    print(f"Error running test: {e}")

input("Press Enter to exit...")
"""
    
    with io.open('run_nvidia_test_launcher.py', 'w', encoding='utf-8') as f:
        f.write(py_launcher)
    
    print("[OK] Created run_nvidia_test_launcher.py")
    print("You can double-click this file to run the NVIDIA test")

def main():
    """Main function."""
    print_header("TERMINAL DIAGNOSTIC AND FIX TOOL")
    print("This script will diagnose and fix terminal issues.")
    print("It will also test the NVIDIA API connection.")
    print("\nRunning diagnostics...")
    
    # Run diagnostics
    diagnose_system()
    test_shells()
    check_python_packages()
    test_nvidia_api()
    fix_terminal_issues()
    
    print_header("DIAGNOSTICS COMPLETE")
    print("\nFix files created:")
    print("1. run_nvidia_test.bat - Double-click to run with CMD")
    print("2. run_nvidia_test.ps1 - Right-click and 'Run with PowerShell'")
    print("3. run_nvidia_test_launcher.py - Double-click to run with Python")
    
    print("\nNext steps:")
    print("1. Try running one of the launchers above")
    print("2. Check if the NVIDIA API test works")
    print("3. If successful, continue with NIS Protocol development")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()