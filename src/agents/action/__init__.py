"""
Action Agents Package

This package contains action agents responsible for executing actions in the environment,
including automated fixing of audit violations and code quality issues.
"""

try:
    from .simple_audit_fixing_agent import SimpleAuditFixingAgent, create_simple_audit_fixer
    AUDIT_FIXING_AVAILABLE = True
except ImportError:
    AUDIT_FIXING_AVAILABLE = False
    SimpleAuditFixingAgent = None
    create_simple_audit_fixer = None

# Export main classes
__all__ = [
    "SimpleAuditFixingAgent",
    "create_simple_audit_fixer", 
    "AUDIT_FIXING_AVAILABLE"
] 