#!/usr/bin/env python3
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
