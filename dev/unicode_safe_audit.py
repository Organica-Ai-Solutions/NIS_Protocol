#!/usr/bin/env python3
"""
Unicode-Safe Audit Script

Runs the NIS integrity audit without Unicode character issues.
"""

import subprocess
import sys
import os

def run_safe_audit():
    """Run audit with Unicode handling"""
    print("STARTING NIS ENGINEERING INTEGRITY AUDIT")
    print("=" * 50)
    
    try:
        # Set environment to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the audit script
        result = subprocess.run([
            sys.executable, 
            'nis-integrity-toolkit/audit-scripts/full-audit.py', 
            '--project-path', '.', 
            '--output-report'
        ], capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print("AUDIT COMPLETED SUCCESSFULLY")
            print(result.stdout)
        else:
            print("AUDIT COMPLETED WITH ISSUES")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        # Try to extract key metrics
        lines = result.stdout.split('\n') if result.stdout else []
        for line in lines:
            if 'Integrity Score:' in line or 'Total Issues:' in line:
                print(f"KEY METRIC: {line.strip()}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"AUDIT ERROR: {e}")
        return False

if __name__ == "__main__":
    success = run_safe_audit()
    print(f"\nAUDIT SUCCESS: {success}")
    sys.exit(0 if success else 1) 