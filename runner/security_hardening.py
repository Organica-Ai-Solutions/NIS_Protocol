"""
Enhanced Security Hardening for NIS Protocol Runner
Implements network isolation, resource limits, audit logging, and input validation
"""

import os
import re
import hashlib
import logging
import resource
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("nis.runner.security")


@dataclass
class SecurityViolation:
    """Security violation record"""
    violation_type: str
    severity: str  # critical, high, medium, low
    description: str
    code_snippet: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionAudit:
    """Execution audit record"""
    execution_id: str
    code_hash: str
    language: str
    success: bool
    execution_time: float
    memory_used_mb: int
    violations: List[SecurityViolation]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "execution_id": self.execution_id,
            "code_hash": self.code_hash,
            "language": self.language,
            "success": self.success,
            "execution_time": self.execution_time,
            "memory_used_mb": self.memory_used_mb,
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description
                }
                for v in self.violations
            ],
            "timestamp": self.timestamp
        }


class NetworkIsolation:
    """
    Network isolation for code execution
    Blocks all network access during code execution
    """
    
    @staticmethod
    def block_network():
        """
        Block network access by overriding socket module
        This prevents code from making any network requests
        """
        import socket
        import urllib.request
        import urllib.error
        
        # Store original functions
        original_socket = socket.socket
        original_urlopen = urllib.request.urlopen
        
        def blocked_socket(*args, **kwargs):
            raise PermissionError("Network access is blocked during code execution")
        
        def blocked_urlopen(*args, **kwargs):
            raise PermissionError("Network access is blocked during code execution")
        
        # Override socket creation
        socket.socket = blocked_socket
        urllib.request.urlopen = blocked_urlopen
        
        logger.info("Network isolation enabled")
        
        return original_socket, original_urlopen
    
    @staticmethod
    def restore_network(original_socket, original_urlopen):
        """Restore network access"""
        import socket
        import urllib.request
        
        socket.socket = original_socket
        urllib.request.urlopen = original_urlopen
        
        logger.info("Network isolation disabled")


class ResourceLimiter:
    """
    Enhanced resource limits for code execution
    Implements CPU, memory, and file size limits
    """
    
    @staticmethod
    def set_limits(
        cpu_time_seconds: int = 30,
        memory_mb: int = 512,
        file_size_mb: int = 10,
        max_processes: int = 1
    ):
        """
        Set resource limits using resource module
        
        Args:
            cpu_time_seconds: Maximum CPU time
            memory_mb: Maximum memory usage
            file_size_mb: Maximum file size
            max_processes: Maximum number of processes
        """
        try:
            # CPU time limit (RLIMIT_CPU)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (cpu_time_seconds, cpu_time_seconds)
            )
            
            # Memory limit (RLIMIT_AS - address space)
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_bytes, memory_bytes)
            )
            
            # File size limit (RLIMIT_FSIZE)
            file_bytes = file_size_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (file_bytes, file_bytes)
            )
            
            # Set process limit (prevent fork bombs)
            # Allow enough processes for scientific libraries (NumPy/OpenBLAS need 8+ threads)
            if hasattr(resource, 'RLIMIT_NPROC'):
                resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
            
            logger.info(
                f"Resource limits set: CPU={cpu_time_seconds}s, "
                f"Memory={memory_mb}MB, FileSize={file_size_mb}MB, "
                f"Processes=50"
            )
            
        except Exception as e:
            logger.error(f"Failed to set resource limits: {e}")
            raise


class CodeValidator:
    """
    Input validation for code execution
    Detects dangerous patterns and potential security issues
    """
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = {
        'import_bypass': (
            r'__import__\s*\(',
            'critical',
            'Attempted to bypass import restrictions'
        ),
        'eval_exec': (
            r'(eval|exec|compile)\s*\(',
            'critical',
            'Attempted to use eval/exec/compile'
        ),
        'file_operations': (
            r'(open|file)\s*\(',
            'high',
            'Attempted file operations (use allowed libraries instead)'
        ),
        'subprocess': (
            r'(subprocess|os\.system|os\.popen)',
            'critical',
            'Attempted to execute system commands'
        ),
        'network_access': (
            r'(socket\.|urllib\.|requests\.|http\.)',
            'high',
            'Attempted network access'
        ),
        'code_injection': (
            r'(globals|locals|vars|dir)\s*\(',
            'medium',
            'Attempted to access global/local scope'
        ),
        'pickle_marshal': (
            r'(pickle|marshal|dill|joblib)\.',
            'high',
            'Attempted to use serialization libraries'
        ),
        'ctypes': (
            r'ctypes\.',
            'critical',
            'Attempted to use ctypes (low-level memory access)'
        ),
    }
    
    # Suspicious patterns that should be logged but not blocked
    SUSPICIOUS_PATTERNS = {
        'infinite_loop': (
            r'while\s+True\s*:',
            'medium',
            'Potential infinite loop detected'
        ),
        'large_allocation': (
            r'\[\s*\d+\s*\]\s*\*\s*\d{6,}',
            'medium',
            'Large memory allocation detected'
        ),
    }
    
    def validate(self, code: str, language: str = "python") -> Tuple[bool, List[SecurityViolation]]:
        """
        Validate code for security issues
        
        Args:
            code: Code to validate
            language: Programming language
        
        Returns:
            Tuple of (is_safe, violations)
        """
        violations = []
        
        if language == "python":
            # Check dangerous patterns
            for pattern_name, (pattern, severity, description) in self.DANGEROUS_PATTERNS.items():
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    violations.append(SecurityViolation(
                        violation_type=pattern_name,
                        severity=severity,
                        description=description,
                        code_snippet=match.group(0)
                    ))
            
            # Check suspicious patterns (log but don't block)
            for pattern_name, (pattern, severity, description) in self.SUSPICIOUS_PATTERNS.items():
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    violations.append(SecurityViolation(
                        violation_type=pattern_name,
                        severity=severity,
                        description=f"WARNING: {description}",
                        code_snippet=match.group(0)
                    ))
        
        # Block if any critical violations found
        critical_violations = [v for v in violations if v.severity == 'critical']
        is_safe = len(critical_violations) == 0
        
        if not is_safe:
            logger.warning(f"Code validation failed: {len(critical_violations)} critical violations")
        
        return is_safe, violations


class AuditLogger:
    """
    Comprehensive audit logging for code execution
    Logs all executions with security metadata
    """
    
    def __init__(self, log_dir: Path = Path("/app/logs/audit")):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit_{today}.jsonl"
        
        logger.info(f"Audit logging initialized: {self.log_file}")
    
    def log_execution(self, audit: ExecutionAudit):
        """
        Log execution audit record
        
        Args:
            audit: Execution audit record
        """
        try:
            # Append to JSONL file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(audit.to_dict()) + '\n')
            
            # Also log to standard logger
            if audit.violations:
                logger.warning(
                    f"Execution {audit.execution_id}: {len(audit.violations)} violations detected"
                )
            else:
                logger.info(
                    f"Execution {audit.execution_id}: Success={audit.success}, "
                    f"Time={audit.execution_time:.2f}s, Memory={audit.memory_used_mb}MB"
                )
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_recent_audits(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent audit records
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of audit records
        """
        try:
            audits = []
            
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        audits.append(json.loads(line))
            
            return audits
            
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []
    
    def get_violation_summary(self) -> Dict[str, int]:
        """
        Get summary of security violations
        
        Returns:
            Dictionary of violation types and counts
        """
        try:
            summary = {}
            
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    for line in f:
                        audit = json.loads(line)
                        for violation in audit.get('violations', []):
                            v_type = violation['type']
                            summary[v_type] = summary.get(v_type, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate violation summary: {e}")
            return {}


class SecurityManager:
    """
    Centralized security management for runner
    Coordinates all security features
    """
    
    def __init__(self):
        """Initialize security manager"""
        self.validator = CodeValidator()
        self.auditor = AuditLogger()
        self.network_isolation = NetworkIsolation()
        self.resource_limiter = ResourceLimiter()
        
        logger.info("Security manager initialized")
    
    def validate_code(self, code: str, language: str = "python") -> Tuple[bool, List[SecurityViolation]]:
        """Validate code before execution"""
        return self.validator.validate(code, language)
    
    def prepare_execution(
        self,
        cpu_time: int = 30,
        memory_mb: int = 512,
        enable_network_isolation: bool = True
    ) -> Optional[Tuple]:
        """
        Prepare secure execution environment
        
        Args:
            cpu_time: CPU time limit in seconds
            memory_mb: Memory limit in MB
            enable_network_isolation: Whether to block network access
        
        Returns:
            Network restoration tuple if network isolation enabled, None otherwise
        """
        # Set resource limits
        self.resource_limiter.set_limits(
            cpu_time_seconds=cpu_time,
            memory_mb=memory_mb
        )
        
        # Enable network isolation
        if enable_network_isolation:
            return self.network_isolation.block_network()
        
        return None
    
    def cleanup_execution(self, network_restore_tuple: Optional[Tuple]):
        """
        Cleanup after execution
        
        Args:
            network_restore_tuple: Tuple returned from prepare_execution
        """
        if network_restore_tuple:
            original_socket, original_urlopen = network_restore_tuple
            self.network_isolation.restore_network(original_socket, original_urlopen)
    
    def log_execution(
        self,
        execution_id: str,
        code: str,
        language: str,
        success: bool,
        execution_time: float,
        memory_used_mb: int,
        violations: List[SecurityViolation]
    ):
        """Log execution audit record"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        audit = ExecutionAudit(
            execution_id=execution_id,
            code_hash=code_hash,
            language=language,
            success=success,
            execution_time=execution_time,
            memory_used_mb=memory_used_mb,
            violations=violations
        )
        
        self.auditor.log_execution(audit)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "violation_summary": self.auditor.get_violation_summary(),
            "recent_audits_count": len(self.auditor.get_recent_audits(limit=10)),
            "log_file": str(self.auditor.log_file)
        }
