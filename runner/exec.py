#!/usr/bin/env python3
"""
NIS Protocol Tool Runner
Secure execution environment for Python scripts and shell commands
"""

import subprocess
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureRunner:
    def __init__(self, workspace: str = "/app/workspace"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)
        self.execution_log = []
        
    def run_shell(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command with security constraints"""
        start_time = time.time()
        
        # Security allowlist - only allow safe commands
        allowed_prefixes = ['ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'head', 'tail', 'wc']
        if not any(cmd.strip().startswith(prefix) for prefix in allowed_prefixes):
            return {
                "success": False, 
                "error": f"Command '{cmd}' not in allowlist",
                "execution_time": 0
            }
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=self.workspace
            )
            
            execution_time = time.time() - start_time
            
            # Log execution for audit
            log_entry = {
                "timestamp": time.time(),
                "command": cmd,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "output_hash": hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
            }
            self.execution_log.append(log_entry)
            logger.info(f"Shell exec: {cmd} | Success: {result.returncode == 0} | Time: {execution_time:.3f}s")
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timeout after {timeout}s", "execution_time": timeout}
        except Exception as e:
            return {"success": False, "error": str(e), "execution_time": time.time() - start_time}

    def run_python(self, filepath: str, args: Optional[list] = None, timeout: int = 60) -> Dict[str, Any]:
        """Execute Python script with security constraints"""
        start_time = time.time()
        
        script_path = self.workspace / filepath
        if not script_path.exists():
            return {"success": False, "error": f"Script {filepath} not found", "execution_time": 0}
        
        # Validate script is in workspace
        if not str(script_path.resolve()).startswith(str(self.workspace.resolve())):
            return {"success": False, "error": "Script outside workspace", "execution_time": 0}
        
        try:
            cmd = ["python", str(script_path)]
            if args:
                cmd.extend(args)
                
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            execution_time = time.time() - start_time
            
            # Log execution for audit
            log_entry = {
                "timestamp": time.time(),
                "script": filepath,
                "args": args or [],
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "output_hash": hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
            }
            self.execution_log.append(log_entry)
            logger.info(f"Python exec: {filepath} | Success: {result.returncode == 0} | Time: {execution_time:.3f}s")
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Script timeout after {timeout}s", "execution_time": timeout}
        except Exception as e:
            return {"success": False, "error": str(e), "execution_time": time.time() - start_time}

    def get_audit_log(self) -> list:
        """Return execution audit log"""
        return self.execution_log

    def write_script(self, filename: str, content: str) -> Dict[str, Any]:
        """Write a Python script to workspace"""
        try:
            script_path = self.workspace / filename
            script_path.write_text(content)
            logger.info(f"Script written: {filename}")
            return {"success": True, "path": str(script_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    runner = SecureRunner()
    
    # Test shell command
    print("=== Testing Shell Execution ===")
    result = runner.run_shell("ls -la")
    print(f"Shell result: {result}")
    
    # Test Python script creation and execution
    print("\n=== Testing Python Execution ===")
    
    # Create a test script
    test_script = '''
import time
import sys

print("Hello from NIS Protocol Runner!")
print(f"Current time: {time.time()}")
print(f"Python version: {sys.version}")

# Calculate confidence demo
factors = [0.8, 0.9, 0.7]
confidence = sum(factors) / len(factors)
print(f"Calculated confidence: {confidence:.3f}")
'''
    
    write_result = runner.write_script("test_confidence.py", test_script)
    print(f"Write result: {write_result}")
    
    if write_result["success"]:
        run_result = runner.run_python("test_confidence.py")
        print(f"Python result: {run_result}")
    
    # Show audit log
    print("\n=== Audit Log ===")
    for entry in runner.get_audit_log():
        print(json.dumps(entry, indent=2))
