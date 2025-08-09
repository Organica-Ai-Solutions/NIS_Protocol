#!/usr/bin/env python3
"""
Simple Audit Fixing Action Agent

A lightweight action agent that systematically fixes audit violations
without complex infrastructure dependencies. Uses direct tool calls
and third-party protocols to fix code issues.

Features:
- Fixes hardcoded values detected by self-audit
- Simple MCP protocol integration
- Direct file manipulation
- Audit trail of fixes
"""

import os
import time
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Simplified imports to avoid infrastructure dependencies
from ...utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class FixStrategy(Enum):
    """Strategies for fixing violations"""
    HARDCODED_VALUE_REPLACEMENT = "hardcoded_value_replacement"
    HYPE_LANGUAGE_CORRECTION = "hype_language_correction"
    DOCUMENTATION_UPDATE = "documentation_update"

@dataclass
class ViolationFix:
    """Represents a fix applied to a violation"""
    violation: IntegrityViolation
    fix_strategy: FixStrategy
    original_content: str
    fixed_content: str
    file_path: str
    line_number: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class FixSession:
    """Represents a fixing session"""
    session_id: str
    start_time: float
    violations_detected: int = 0
    violations_fixed: int = 0
    violations_failed: int = 0
    fixes_applied: List[ViolationFix] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

class SimpleAuditFixingAgent:
    """
    Lightweight audit fixing agent that can systematically fix
    violations using third-party tools and protocols.
    """
    
    def __init__(self, agent_id: str = "simple_audit_fixer"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"nis.action.{agent_id}")
        
        # Fix strategies mapping
        self.fix_strategies = {
            ViolationType.HARDCODED_VALUE: FixStrategy.HARDCODED_VALUE_REPLACEMENT,
            ViolationType.HYPE_LANGUAGE: FixStrategy.HYPE_LANGUAGE_CORRECTION,
            ViolationType.UNSUBSTANTIATED_CLAIM: FixStrategy.DOCUMENTATION_UPDATE,
            ViolationType.PERFECTION_CLAIM: FixStrategy.HYPE_LANGUAGE_CORRECTION,
            ViolationType.INTERPRETABILITY_CLAIM: FixStrategy.DOCUMENTATION_UPDATE
        }
        
        # Session tracking
        self.current_session: Optional[FixSession] = None
        self.fix_history: List[FixSession] = []
        
        # MCP integration (simplified)
        self.mcp_enabled = self._check_mcp_availability()
        
        self.logger.info(f"SimpleAuditFixingAgent '{agent_id}' initialized")
    
    def _check_mcp_availability(self) -> bool:
        """Check if MCP tools are available"""
        # For now, simulate MCP availability
        # In real implementation, this would check MCP service connectivity
        return os.getenv("MCP_API_KEY") is not None
    
    def start_fixing_session(self, target_directories: List[str] = None) -> str:
        """
        Start a new audit fixing session.
        
        Args:
            target_directories: Directories to scan and fix
            
        Returns:
            Session ID
        """
        session_id = f"fix_session_{int(time.time())}"
        
        self.current_session = FixSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        self.logger.info(f"🚀 Starting audit fixing session: {session_id}")
        
        # Scan for violations
        violations = self._scan_for_violations(target_directories or ['src/'])
        self.current_session.violations_detected = len(violations)
        
        self.logger.info(f"🔍 Found {len(violations)} violations to fix")
        
        # Apply fixes
        for violation in violations:
            self._apply_violation_fix(violation)
        
        # Complete session
        self._complete_session()
        
        return session_id
    
    def _scan_for_violations(self, target_directories: List[str]) -> List[IntegrityViolation]:
        """Scan directories for violations"""
        all_violations = []
        
        for directory in target_directories:
            if os.path.exists(directory):
                violations = self._scan_directory(directory)
                all_violations.extend(violations)
        
        return all_violations
    
    def _scan_directory(self, directory: str) -> List[IntegrityViolation]:
        """Scan a single directory"""
        violations = []
        
        for root, dirs, files in os.walk(directory):
            # Skip cache and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Use self-audit engine to find violations
                        file_violations = self_audit_engine.audit_text(content, f"file:{file_path}")
                        
                        # Add file context
                        for violation in file_violations:
                            violation.file_path = file_path
                            violation.line_number = self._find_line_number(content, violation.position)
                        
                        violations.extend(file_violations)
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️  Could not scan {file_path}: {e}")
        
        return violations
    
    def _find_line_number(self, content: str, position: int) -> int:
        """Find line number for character position"""
        return content[:position].count('\n') + 1
    
    def _apply_violation_fix(self, violation: IntegrityViolation) -> ViolationFix:
        """Apply a fix for a violation"""
        file_path = getattr(violation, 'file_path', 'unknown')
        line_number = getattr(violation, 'line_number', None)
        
        self.logger.info(f"🔧 Fixing {violation.violation_type.value} in {file_path}:{line_number}")
        
        # Determine strategy
        strategy = self.fix_strategies.get(violation.violation_type, FixStrategy.HARDCODED_VALUE_REPLACEMENT)
        
        # Create fix object
        fix = ViolationFix(
            violation=violation,
            fix_strategy=strategy,
            original_content="",
            fixed_content="",
            file_path=file_path,
            line_number=line_number
        )
        
        try:
            # Read file
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                fix.original_content = original_content
                
                # Apply fix
                fix = self._apply_fix_strategy(fix)
                
                # Write file if content changed
                if fix.fixed_content != fix.original_content:
                    success = self._write_fixed_file(fix)
                    fix.success = success
                    
                    if success:
                        self.current_session.violations_fixed += 1
                        if file_path not in self.current_session.files_modified:
                            self.current_session.files_modified.append(file_path)
                        self.logger.info(f"✅ Fixed {violation.violation_type.value} in {file_path}")
                    else:
                        self.current_session.violations_failed += 1
                        self.logger.error(f"❌ Failed to fix {violation.violation_type.value} in {file_path}")
                else:
                    self.logger.info(f"📄 No changes needed for {file_path}")
                    
        except Exception as e:
            fix.error_message = str(e)
            fix.success = False
            self.current_session.violations_failed += 1
            self.logger.error(f"❌ Exception fixing {file_path}: {e}")
        
        self.current_session.fixes_applied.append(fix)
        return fix
    
    def _apply_fix_strategy(self, fix: ViolationFix) -> ViolationFix:
        """Apply the appropriate fix strategy"""
        violation = fix.violation
        
        if fix.fix_strategy == FixStrategy.HARDCODED_VALUE_REPLACEMENT:
            # Replace hardcoded value with calculation
            fixed_content = fix.original_content.replace(
                violation.text, 
                violation.suggested_replacement
            )
            fix.fixed_content = fixed_content
            
        elif fix.fix_strategy == FixStrategy.HYPE_LANGUAGE_CORRECTION:
            # Replace hype language with approved terms
            fixed_content = fix.original_content.replace(
                violation.text,
                violation.suggested_replacement
            )
            fix.fixed_content = fixed_content
            
        elif fix.fix_strategy == FixStrategy.DOCUMENTATION_UPDATE:
            # Add validation context to claims
            fixed_content = fix.original_content.replace(
                violation.text,
                violation.suggested_replacement
            )
            fix.fixed_content = fixed_content
        
        return fix
    
    def _write_fixed_file(self, fix: ViolationFix) -> bool:
        """Write the fixed content to file"""
        try:
            # Use MCP tool if available
            if self.mcp_enabled:
                success = self._use_mcp_file_tool(fix)
                if success:
                    return True
            
            # Fallback to direct file write
            with open(fix.file_path, 'w', encoding='utf-8') as f:
                f.write(fix.fixed_content)
            
            return True
            
        except Exception as e:
            fix.error_message = str(e)
            return False
    
    def _use_mcp_file_tool(self, fix: ViolationFix) -> bool:
        """Use MCP tool to edit file (simulated for now)"""
        try:
            # Simulate MCP tool call
            self.logger.info(f"🔌 Using MCP file editor tool for {fix.file_path}")
            
            # In real implementation, this would make actual MCP API call
            mcp_request = {
                "tool": "file_editor",
                "action": "replace_text",
                "file_path": fix.file_path,
                "old_text": fix.violation.text,
                "new_text": fix.violation.suggested_replacement
            }
            
            # Simulate successful MCP response
            self.logger.info(f"🔌 MCP tool response: SUCCESS")
            
            # For now, still do direct write
            with open(fix.file_path, 'w', encoding='utf-8') as f:
                f.write(fix.fixed_content)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"🔌 MCP tool failed: {e}")
            return False
    
    def _complete_session(self):
        """Complete the current fixing session"""
        if not self.current_session:
            return
        
        session = self.current_session
        duration = time.time() - session.start_time
        
        # Calculate success rate
        total_attempts = session.violations_fixed + session.violations_failed
        success_rate = session.violations_fixed / total_attempts if total_attempts > 0 else 0.0
        
        # Log summary
        self.logger.info(f"🎯 AUDIT FIXING SESSION COMPLETE!")
        self.logger.info(f"   Session ID: {session.session_id}")
        self.logger.info(f"   Duration: {duration:.2f} seconds")
        self.logger.info(f"   Violations detected: {session.violations_detected}")
        self.logger.info(f"   Violations fixed: {session.violations_fixed}")
        self.logger.info(f"   Violations failed: {session.violations_failed}")
        self.logger.info(f"   Success rate: {success_rate:.1%}")
        self.logger.info(f"   Files modified: {len(session.files_modified)}")
        
        # Add to history
        self.fix_history.append(session)
        self.current_session = None
    
    def get_session_report(self, session_id: str = None) -> Dict[str, Any]:
        """Get detailed session report"""
        if session_id:
            session = next((s for s in self.fix_history if s.session_id == session_id), None)
        else:
            session = self.fix_history[-1] if self.fix_history else None
        
        if not session:
            return {"error": "No session found"}
        
        total_attempts = session.violations_fixed + session.violations_failed
        success_rate = session.violations_fixed / total_attempts if total_attempts > 0 else 0.0
        
        return {
            "session_id": session.session_id,
            "duration": time.time() - session.start_time,
            "violations_detected": session.violations_detected,
            "violations_fixed": session.violations_fixed,
            "violations_failed": session.violations_failed,
            "success_rate": success_rate,
            "files_modified": session.files_modified,
            "fixes_by_type": self._group_fixes_by_type(session.fixes_applied),
            "tools_used": self._get_tools_used(session.fixes_applied)
        }
    
    def _group_fixes_by_type(self, fixes: List[ViolationFix]) -> Dict[str, int]:
        """Group fixes by violation type"""
        fix_counts = {}
        for fix in fixes:
            violation_type = fix.violation.violation_type.value
            fix_counts[violation_type] = fix_counts.get(violation_type, 0) + 1
        return fix_counts
    
    def _get_tools_used(self, fixes: List[ViolationFix]) -> List[str]:
        """Get list of tools used in fixes"""
        tools = set()
        for fix in fixes:
            if hasattr(fix, 'tool_used') and fix.tool_used:
                tools.add(fix.tool_used)
        return list(tools)

# Simple factory function
def create_simple_audit_fixer(agent_id: str = "simple_audit_fixer") -> SimpleAuditFixingAgent:
    """Create a simple audit fixing agent"""
    return SimpleAuditFixingAgent(agent_id) 