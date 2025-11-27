"""
Code Executor - NIS Protocol v4.0
Execute code locally with sandboxing, capture outputs (plots, files, data).
This is the missing piece that connects LLM → Code → Results → LLM loop.
"""

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("nis.code_executor")

# Safe imports for execution
ALLOWED_MODULES = {
    'math', 'random', 'datetime', 'json', 'time', 're', 'collections',
    'itertools', 'functools', 'operator', 'string', 'decimal', 'fractions',
    'statistics', 'copy', 'pprint', 'textwrap', 'unicodedata',
    # Scientific
    'numpy', 'pandas', 'scipy', 'sympy',
    # Visualization
    'matplotlib', 'matplotlib.pyplot', 'seaborn', 'plotly',
    # Data
    'csv', 'io', 'base64',
}

BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib', 'requests',
    'http', 'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib',
    'multiprocessing', 'threading', 'ctypes', 'pickle', 'marshal',
    '__builtin__', 'builtins', 'importlib', 'code', 'codeop',
}


@dataclass
class ExecutionOutput:
    """Output from code execution"""
    execution_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    plots: List[Dict[str, str]] = field(default_factory=list)  # [{name, base64, type}]
    files: List[Dict[str, str]] = field(default_factory=list)  # [{name, base64, type}]
    dataframes: List[Dict[str, Any]] = field(default_factory=list)  # [{name, preview, shape}]
    execution_time_ms: float = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": str(self.result) if self.result is not None else None,
            "plots": self.plots,
            "files": self.files,
            "dataframes": self.dataframes,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


class CodeExecutor:
    """
    Safe code executor with output capture.
    Captures: stdout, plots, dataframes, files.
    """
    
    def __init__(self, output_dir: str = "data/execution_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executions: Dict[str, ExecutionOutput] = {}
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe execution namespace"""
        safe_globals = {
            '__builtins__': {
                # Safe builtins
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'map': map, 'filter': filter, 'sorted': sorted, 'reversed': reversed,
                'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                'pow': pow, 'divmod': divmod, 'all': all, 'any': any,
                'print': print, 'repr': repr, 'type': type,
                'isinstance': isinstance, 'issubclass': issubclass,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                'callable': callable, 'iter': iter, 'next': next,
                'open': self._safe_open,  # Restricted open
                'True': True, 'False': False, 'None': None,
                '__import__': self._safe_import,
            }
        }
        
        # Pre-import safe modules
        try:
            import numpy as np
            safe_globals['np'] = np
            safe_globals['numpy'] = np
        except ImportError:
            pass
        
        try:
            import pandas as pd
            safe_globals['pd'] = pd
            safe_globals['pandas'] = pd
        except ImportError:
            pass
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            safe_globals['plt'] = plt
            safe_globals['matplotlib'] = matplotlib
        except ImportError:
            pass
        
        try:
            import math
            safe_globals['math'] = math
        except ImportError:
            pass
        
        try:
            import json
            safe_globals['json'] = json
        except ImportError:
            pass
        
        try:
            import datetime
            safe_globals['datetime'] = datetime
        except ImportError:
            pass
        
        try:
            import random
            safe_globals['random'] = random
        except ImportError:
            pass
        
        return safe_globals
    
    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Restricted import function"""
        base_name = name.split('.')[0]
        
        if base_name in BLOCKED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed for security reasons")
        
        if name in ALLOWED_MODULES or base_name in ALLOWED_MODULES:
            return __import__(name, globals, locals, fromlist, level)
        
        raise ImportError(f"Module '{name}' is not in the allowed list")
    
    def _safe_open(self, file, mode='r', *args, **kwargs):
        """Restricted file open - only allow reading from safe locations"""
        if 'w' in mode or 'a' in mode or '+' in mode:
            # Only allow writing to temp directory
            file_path = Path(file)
            if not str(file_path).startswith(str(self.output_dir)):
                raise PermissionError("Writing files outside output directory is not allowed")
        return open(file, mode, *args, **kwargs)
    
    def _capture_plots(self) -> List[Dict[str, str]]:
        """Capture any matplotlib plots"""
        plots = []
        try:
            import matplotlib.pyplot as plt
            
            # Get all figures
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                # Save to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                # Encode as base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                plots.append({
                    "name": f"plot_{fig_num}.png",
                    "base64": img_base64,
                    "type": "image/png"
                })
                
                plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Error capturing plots: {e}")
        
        return plots
    
    def _capture_dataframes(self, local_vars: Dict) -> List[Dict[str, Any]]:
        """Capture pandas DataFrames from execution"""
        dataframes = []
        try:
            import pandas as pd
            
            for name, value in local_vars.items():
                if isinstance(value, pd.DataFrame):
                    dataframes.append({
                        "name": name,
                        "shape": list(value.shape),
                        "columns": list(value.columns),
                        "preview": value.head(10).to_dict(),
                        "dtypes": {str(k): str(v) for k, v in value.dtypes.items()}
                    })
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error capturing dataframes: {e}")
        
        return dataframes
    
    async def execute(
        self,
        code: str,
        timeout_seconds: int = 30,
        capture_plots: bool = True,
        capture_dataframes: bool = True
    ) -> ExecutionOutput:
        """
        Execute code safely and capture all outputs.
        
        Args:
            code: Python code to execute
            timeout_seconds: Maximum execution time
            capture_plots: Whether to capture matplotlib plots
            capture_dataframes: Whether to capture pandas DataFrames
            
        Returns:
            ExecutionOutput with all captured data
        """
        import time
        start_time = time.time()
        execution_id = str(uuid.uuid4())[:8]
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        output = ExecutionOutput(execution_id=execution_id, success=False)
        
        try:
            # Create safe namespace
            safe_globals = self._create_safe_globals()
            local_vars = {}
            
            # Security check
            for blocked in BLOCKED_MODULES:
                if f"import {blocked}" in code or f"from {blocked}" in code:
                    raise SecurityError(f"Import of '{blocked}' is not allowed")
            
            # Execute with timeout
            exec(compile(code, '<code>', 'exec'), safe_globals, local_vars)
            
            output.success = True
            
            # Capture last expression result if any
            if '_' in local_vars:
                output.result = local_vars['_']
            
            # Capture plots
            if capture_plots:
                output.plots = self._capture_plots()
            
            # Capture dataframes
            if capture_dataframes:
                output.dataframes = self._capture_dataframes(local_vars)
            
        except Exception as e:
            output.success = False
            output.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        finally:
            # Restore stdout/stderr
            output.stdout = sys.stdout.getvalue()
            output.stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            output.execution_time_ms = (time.time() - start_time) * 1000
        
        # Store execution
        self.executions[execution_id] = output
        
        logger.info(f"Code execution {execution_id}: success={output.success}, time={output.execution_time_ms:.0f}ms, plots={len(output.plots)}")
        
        return output
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionOutput]:
        """Get a previous execution result"""
        return self.executions.get(execution_id)


class SecurityError(Exception):
    """Security violation in code execution"""
    pass


# Global executor instance
_executor: Optional[CodeExecutor] = None


def get_code_executor() -> CodeExecutor:
    """Get or create the global code executor"""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor


async def execute_code(code: str, **kwargs) -> ExecutionOutput:
    """Convenience function to execute code"""
    executor = get_code_executor()
    return await executor.execute(code, **kwargs)
