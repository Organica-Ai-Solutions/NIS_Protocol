"""
NIS Protocol Secure Code Runner
Sandboxed environment for safe code execution
"""

import os
import sys
import asyncio
import tempfile
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import signal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles
import psutil
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, safe_globals
from RestrictedPython.transformer import RestrictingNodeTransformer
from RestrictedPython.PrintCollector import PrintCollector

# Security configuration
from security_config import SecurityConfig

app = FastAPI(
    title="NIS Protocol Optimized Secure Runner",
    description="Sandboxed code execution environment with advanced tool optimization and token efficiency",
    version="3.2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
WORKSPACE_DIR = Path("/app/workspace")
TEMP_DIR = Path("/app/temp")
MAX_EXECUTION_TIME = 30  # seconds
MAX_MEMORY_MB = 512
MAX_PROCESSES = 5

# Execution tracking
active_executions: Dict[str, Dict[str, Any]] = {}

class CodeExecutionRequest(BaseModel):
    code_content: str = Field(..., description="Code content to execute")  # Clear parameter name
    programming_language: str = Field(default="python", description="Programming language (python, javascript, shell)")  # Unambiguous name
    execution_timeout_seconds: int = Field(default=30, description="Maximum execution time in seconds")  # Descriptive name
    memory_limit_mb: int = Field(default=512, description="Memory limit in megabytes")  # Clear units
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables for execution")  # Descriptive name
    additional_files: List[str] = Field(default_factory=list, description="Additional files to include in execution context")  # Clear purpose
    response_format: str = Field(default="detailed", description="Response format: concise, detailed, structured")  # Tool optimization
    token_limit: Optional[int] = Field(default=None, description="Maximum tokens for response (optimization feature)")  # Token efficiency

class ExecutionResult(BaseModel):
    execution_id: str
    success: bool
    output: str
    error: str
    execution_time_seconds: float  # Clear units
    memory_used_mb: int  # Clear units
    exit_code: int
    security_violations: List[str]
    
    # Tool optimization metadata
    response_format: str = "detailed"
    token_estimate: Optional[int] = None
    optimization_applied: bool = False

class SecurityConfig:
    """Security configuration for code execution"""
    
    # Restricted imports
    BLOCKED_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib', 'poplib', 'imaplib',
        'telnetlib', 'multiprocessing', 'threading', 'asyncio',
        'ctypes', 'marshal', 'pickle', 'dill', 'joblib'
    }
    
    # Allowed builtins
    SAFE_BUILTINS = {
        'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
        'set', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
        'sum', 'min', 'max', 'abs', 'round', 'pow', 'divmod',
        'print', 'repr', 'type', 'isinstance', 'hasattr', 'getattr'
    }
    
    # File system restrictions
    ALLOWED_EXTENSIONS = {'.py', '.txt', '.json', '.csv', '.md'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def safe_import_function(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom safe import function that only allows whitelisted modules"""
    # Allow only modules that are in our allowed list
    if name in ['math', 'random', 'datetime', 'json', 'requests']:
        return __import__(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Module '{name}' is not allowed")

def create_safe_globals():
    """Create a safe global namespace for code execution"""
    # Use RestrictedPython's safe_globals as base
    restricted_globals = safe_globals.copy()
    
    # Add additional safe functions
    restricted_globals.update({
        '__builtins__': {
            **safe_builtins,
            '_print_': PrintCollector,
            '_getattr_': getattr,
            '_getitem_': lambda obj, key: obj[key],
            '_getiter_': lambda obj: iter(obj),
            '_iter_unpack_sequence_': lambda it, spec: list(it),
            # Essential Python built-ins
            'len': len,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'range': range,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'all': all,
            'any': any,
            '__import__': safe_import_function,
        }
    })
    
    # Add safe modules
    import math
    import random
    import datetime
    import json
    import requests
    from browser_security import get_secure_browser, SecureBrowser
    
    restricted_globals.update({
        'math': math,
        'random': random,
        'datetime': datetime,
        'json': json,
        'requests': requests,
        'get_secure_browser': get_secure_browser,
        'SecureBrowser': SecureBrowser,
    })
    
    return restricted_globals

def validate_code_security(code: str) -> List[str]:
    """Validate code for security violations"""
    violations = []
    
    # Check for blocked imports
    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Extract module names more precisely
            if line.startswith('import '):
                # Handle: import module, import module as alias
                import_part = line[7:].split(' as ')[0].strip()
                modules = [m.strip() for m in import_part.split(',')]
            elif line.startswith('from '):
                # Handle: from module import ...
                from_part = line[5:].split(' import ')[0].strip()
                modules = [from_part]
            else:
                modules = []
            
            for module in modules:
                if module in SecurityConfig.BLOCKED_IMPORTS:
                    violations.append(f"Line {line_num}: Blocked import '{module}'")
        
        # Check for dangerous operations
        dangerous_patterns = [
            'eval(', 'exec(', 'compile(', '__import__(',
            'open(', 'file(', 'input(', 'raw_input(',
            'globals(', 'locals(', 'vars(', 'dir('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in line:
                violations.append(f"Line {line_num}: Dangerous operation '{pattern}'")
    
    return violations

async def execute_python_code(
    code_content: str,  # Clear parameter name
    execution_id: str,
    timeout_seconds: int = 30,  # Clear units
    memory_limit_mb: int = 512,  # Clear units
    response_format: str = "detailed",  # Response optimization
    token_limit: Optional[int] = None  # Token efficiency
) -> ExecutionResult:
    """Execute Python code in a restricted environment"""
    
    start_time = time.time()
    
    # Security validation with optimized parameter name
    security_violations = validate_code_security(code_content)
    if security_violations:
        return ExecutionResult(
            execution_id=execution_id,
            success=False,
            output="",
            error="Security violations detected",
            execution_time_seconds=0.0,  # Updated field name
            memory_used_mb=0,  # Updated field name
            exit_code=-1,
            security_violations=security_violations,
            response_format=response_format,
            optimization_applied=True
        )
    
    try:
        # Compile code with RestrictedPython
        compiled_code = compile_restricted(code_content, '<string>', 'exec')  # Updated parameter name
        if compiled_code is None:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error="Code compilation failed",
                execution_time_seconds=time.time() - start_time,  # Updated field name
                memory_used_mb=0,  # Updated field name
                exit_code=-1,
                security_violations=["Compilation failed"]
            )
        
        # Create safe execution environment
        safe_globals = create_safe_globals()
        safe_locals = {}
        
        try:
            # Execute with timeout
            def timeout_handler(signum, frame):
                raise TimeoutError("Execution timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            # Track memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute the code
            exec(compiled_code, safe_globals, safe_locals)
            
            # Stop timeout
            signal.alarm(0)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = max(0, int(memory_after - memory_before))
            
            # Get print output from PrintCollector
            output = ""
            if '_print' in safe_locals:
                print_collector = safe_locals['_print']
                if hasattr(print_collector, 'txt'):
                    # PrintCollector.txt is a list, join it
                    if isinstance(print_collector.txt, list):
                        output = ''.join(print_collector.txt)
                    else:
                        output = str(print_collector.txt)
            
            return ExecutionResult(
                execution_id=execution_id,
                success=True,
                output=output,
                error="",
                execution_time=time.time() - start_time,
                memory_used=memory_used,
                exit_code=0,
                security_violations=[]
            )
            
        except TimeoutError:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error="Execution timeout",
                execution_time=timeout,
                memory_used=0,
                exit_code=-1,
                security_violations=["Timeout"]
            )
        except Exception as e:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                memory_used=0,
                exit_code=-1,
                security_violations=[]
            )
        finally:
            signal.alarm(0)
            
    except Exception as e:
        return ExecutionResult(
            execution_id=execution_id,
            success=False,
            output="",
            error=f"Execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            memory_used=0,
            exit_code=-1,
            security_violations=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "nis-secure-runner",
        "version": "3.2.1",
        "active_executions": len(active_executions),
        "workspace_available": WORKSPACE_DIR.exists(),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }

@app.post("/execute", response_model=ExecutionResult)
async def execute_code(request: CodeExecutionRequest):
    """Execute code in a secure sandbox"""
    
    # Check system limits
    if len(active_executions) >= MAX_PROCESSES:
        raise HTTPException(
            status_code=429, 
            detail="Too many active executions"
        )
    
    # Generate execution ID
    execution_id = str(uuid.uuid4())
    
    # Track execution with optimization metadata
    active_executions[execution_id] = {
        "start_time": time.time(),
        "language": request.programming_language,
        "status": "running",
        "response_format": request.response_format,
        "optimization_enabled": True
    }
    
    try:
        if request.programming_language.lower() == "python":
            result = await execute_python_code(
                request.code_content,  # Updated parameter name
                execution_id,
                min(request.execution_timeout_seconds, MAX_EXECUTION_TIME),
                min(request.memory_limit_mb, MAX_MEMORY_MB),
                request.response_format,  # Pass response format
                request.token_limit  # Pass token limit
            )
        else:
            result = ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=f"Unsupported language: {request.programming_language}",  # Updated parameter name
                execution_time_seconds=0.0,  # Updated field name
                memory_used_mb=0,  # Updated field name
                exit_code=-1,
                security_violations=["Unsupported language"],
                response_format=request.response_format,
                optimization_applied=True
            )
        
        return result
        
    finally:
        # Clean up tracking
        if execution_id in active_executions:
            del active_executions[execution_id]

@app.get("/executions")
async def list_executions():
    """List active executions"""
    return {
        "active_executions": active_executions,
        "total_active": len(active_executions),
        "max_concurrent": MAX_PROCESSES
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the workspace"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_path = Path(file.filename)
    if file_path.suffix not in SecurityConfig.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed: {file_path.suffix}"
        )
    
    # Read and validate size
    content = await file.read()
    if len(content) > SecurityConfig.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail="File too large"
        )
    
    # Save to workspace
    workspace_file = WORKSPACE_DIR / file.filename
    async with aiofiles.open(workspace_file, 'wb') as f:
        await f.write(content)
    
    return {
        "filename": file.filename,
        "size": len(content),
        "path": str(workspace_file)
    }

@app.get("/workspace")
async def list_workspace():
    """List files in workspace"""
    if not WORKSPACE_DIR.exists():
        return {"files": []}
    
    files = []
    for file_path in WORKSPACE_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "path": str(file_path)
            })
    
    return {"files": files}

@app.delete("/workspace/{filename}")
async def delete_workspace_file(filename: str):
    """Delete a file from workspace"""
    file_path = WORKSPACE_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    file_path.unlink()
    return {"message": f"File {filename} deleted"}

@app.get("/system/stats")
async def system_stats():
    """Get system resource statistics"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        },
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "active_executions": len(active_executions),
        "workspace_files": len(list(WORKSPACE_DIR.glob("*"))) if WORKSPACE_DIR.exists() else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
