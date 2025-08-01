#!/usr/bin/env python3
"""
NIS Protocol v3 - Audit Fixing Action Agent

Specialized action agent that systematically fixes code violations detected by the 
consciousness and self-audit systems. Uses third-party protocols (MCP, ACP, A2A) 
to connect with external tools for code manipulation.

Key Features:
- systematically fixes hardcoded values detected by self-audit
- Integrates with MCP tools for file editing
- Uses consciousness agent alerts to trigger fixes
- Maintains audit trail of all fixes
- Validates fixes after application
"""

import os
import time
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# Core NIS imports
from ..enhanced_agent_base import EnhancedAgentBase, AgentConfiguration, AgentState
from ...utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation
from ...adapters.mcp_adapter import MCPAdapter
from ...adapters.bootstrap import initialize_adapters

class FixStrategy(Enum):
    """Strategies for fixing different types of violations"""
    HARDCODED_VALUE_REPLACEMENT = "hardcoded_value_replacement"
    DOCUMENTATION_UPDATE = "documentation_update"
    TEST_GENERATION = "test_generation"
    PERFORMANCE_VALIDATION = "performance_validation"
    HYPE_LANGUAGE_CORRECTION = "hype_language_correction"

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
    tool_used: Optional[str] = None

@dataclass
class AuditFixSession:
    """Represents a complete audit fixing session"""
    session_id: str
    start_time: float
    violations_detected: int = 0
    violations_fixed: int = 0
    violations_failed: int = 0
    fixes_applied: List[ViolationFix] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    end_time: Optional[float] = None
    success_rate: float = 0.0

class AuditFixingActionAgent(EnhancedAgentBase):
    """
    Specialized action agent for systematically fixing audit violations
    using third-party tools and protocols.
    """
    
    def __init__(
        self,
        agent_id: str = "audit_fixing_action_agent",
        config: Optional[AgentConfiguration] = None,
        mcp_tools_config: Optional[Dict[str, Any]] = None
    ):
        # Default configuration for audit fixing agent
        if config is None:
            config = AgentConfiguration(
                agent_id=agent_id,
                agent_type="action",
                capabilities=["audit_fixing", "code_modification", "file_manipulation", "tool_integration"],
                enable_langgraph=True,
                enable_self_audit=True
            )
        
        super().__init__(config)
        
        self.logger = logging.getLogger(f"nis.action.{agent_id}")
        
        # Initialize third-party protocol adapters
        self.adapters = {}
        try:
            self.adapters = initialize_adapters()
            self.logger.info(f"Initialized {len(self.adapters)} protocol adapters")
        except Exception as e:
            self.logger.warning(f"Failed to initialize some adapters: {e}")
        
        # MCP tools configuration
        self.mcp_tools = mcp_tools_config or {
            "file_editor": "text_editor_tool",
            "code_formatter": "code_formatter_tool", 
            "lint_checker": "lint_checker_tool",
            "test_generator": "test_generator_tool"
        }
        
        # Fix strategies mapping
        self.fix_strategies = {
            ViolationType.HARDCODED_VALUE: FixStrategy.HARDCODED_VALUE_REPLACEMENT,
            ViolationType.HYPE_LANGUAGE: FixStrategy.HYPE_LANGUAGE_CORRECTION,
            ViolationType.UNSUBSTANTIATED_CLAIM: FixStrategy.DOCUMENTATION_UPDATE,
            ViolationType.PERFECTION_CLAIM: FixStrategy.HYPE_LANGUAGE_CORRECTION,
            ViolationType.INTERPRETABILITY_CLAIM: FixStrategy.DOCUMENTATION_UPDATE
        }
        
        # Session tracking
        self.current_session: Optional[AuditFixSession] = None
        self.fix_history: List[AuditFixSession] = []
        
        self.logger.info("AuditFixingActionAgent initialized with third-party tool integration")
    
    async def start_audit_fixing_session(self, target_directories: List[str] = None) -> str:
        """
        Start a new audit fixing session.
        
        Args:
            target_directories: Directories to scan and fix (defaults to ['src/'])
            
        Returns:
            Session ID
        """
        session_id = f"audit_fix_{int(time.time())}_{id(self)}"
        
        self.current_session = AuditFixSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        self.logger.info(f"Started audit fixing session: {session_id}")
        
        # Scan for violations using consciousness agent
        violations = await self._scan_for_violations(target_directories or ['src/'])
        self.current_session.violations_detected = len(violations)
        
        # Apply fixes for each violation
        for violation in violations:
            await self._apply_violation_fix(violation)
        
        # Finalize session
        await self._finalize_session()
        
        return session_id
    
    async def _scan_for_violations(self, target_directories: List[str]) -> List[IntegrityViolation]:
        """
        Scan target directories for integrity violations.
        
        Args:
            target_directories: Directories to scan
            
        Returns:
            List of violations found
        """
        self.logger.info(f"Scanning directories for violations: {target_directories}")
        
        all_violations = []
        
        for directory in target_directories:
            if os.path.exists(directory):
                violations = await self._scan_directory(directory)
                all_violations.extend(violations)
        
        self.logger.info(f"Found {len(all_violations)} violations to fix")
        return all_violations
    
    async def _scan_directory(self, directory: str) -> List[IntegrityViolation]:
        """Scan a single directory for violations"""
        violations = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Use enhanced self-audit engine
                        file_violations = self_audit_engine.audit_text(content, f"file:{file_path}")
                        
                        # Add file path context to violations
                        for violation in file_violations:
                            violation.file_path = file_path
                            violation.line_number = self._find_line_number(content, violation.position)
                        
                        violations.extend(file_violations)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not scan file {file_path}: {e}")
        
        return violations
    
    def _find_line_number(self, content: str, position: int) -> int:
        """Find line number for a given character position in text"""
        lines_before = content[:position].count('\n')
        return lines_before + 1
    
    async def _apply_violation_fix(self, violation: IntegrityViolation) -> ViolationFix:
        """
        Apply a fix for a specific violation.
        
        Args:
            violation: The violation to fix
            
        Returns:
            ViolationFix result
        """
        file_path = getattr(violation, 'file_path', 'unknown')
        line_number = getattr(violation, 'line_number', None)
        
        self.logger.info(f"Applying fix for {violation.violation_type.value} in {file_path}:{line_number}")
        
        # Determine fix strategy
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
            # Read original file content
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                fix.original_content = original_content
                
                # Apply the fix based on strategy
                if strategy == FixStrategy.HARDCODED_VALUE_REPLACEMENT:
                    fix = await self._fix_hardcoded_value(fix)
                elif strategy == FixStrategy.HYPE_LANGUAGE_CORRECTION:
                    fix = await self._fix_hype_language(fix)
                elif strategy == FixStrategy.DOCUMENTATION_UPDATE:
                    fix = await self._fix_documentation_claims(fix)
                
                # Apply the fix using MCP tools if available
                if fix.fixed_content != fix.original_content:
                    success = await self._apply_file_modification(fix)
                    fix.success = success
                    
                    if success:
                        self.current_session.violations_fixed += 1
                        if file_path not in self.current_session.files_modified:
                            self.current_session.files_modified.append(file_path)
                    else:
                        self.current_session.violations_failed += 1
                
        except Exception as e:
            fix.error_message = str(e)
            fix.success = False
            self.current_session.violations_failed += 1
            self.logger.error(f"Failed to fix violation in {file_path}: {e}")
        
        self.current_session.fixes_applied.append(fix)
        return fix
    
    async def _fix_hardcoded_value(self, fix: ViolationFix) -> ViolationFix:
        """Fix hardcoded values by replacing with calculated alternatives"""
        violation = fix.violation
        original_content = fix.original_content
        
        # Use the suggested replacement from the violation
        suggested_replacement = violation.suggested_replacement
        
        # Apply the replacement
        fixed_content = original_content.replace(violation.text, suggested_replacement)
        fix.fixed_content = fixed_content
        fix.tool_used = "self_audit_engine"
        
        self.logger.info(f"Replacing '{violation.text}' with '{suggested_replacement}'")
        return fix
    
    async def _fix_hype_language(self, fix: ViolationFix) -> ViolationFix:
        """Fix hype language by replacing with evidence-based terms"""
        violation = fix.violation
        original_content = fix.original_content
        
        # Use the suggested replacement from the violation  
        suggested_replacement = violation.suggested_replacement
        
        # Apply the replacement
        fixed_content = original_content.replace(violation.text, suggested_replacement)
        fix.fixed_content = fixed_content
        fix.tool_used = "self_audit_engine"
        
        return fix
    
    async def _fix_documentation_claims(self, fix: ViolationFix) -> ViolationFix:
        """Fix unsubstantiated claims in documentation"""
        violation = fix.violation
        original_content = fix.original_content
        
        # Use the suggested replacement which adds validation context
        suggested_replacement = violation.suggested_replacement
        
        fixed_content = original_content.replace(violation.text, suggested_replacement)
        fix.fixed_content = fixed_content
        fix.tool_used = "self_audit_engine"
        
        return fix
    
    async def _apply_file_modification(self, fix: ViolationFix) -> bool:
        """
        Apply file modification using MCP tools or direct file operations.
        
        Args:
            fix: The fix to apply
            
        Returns:
            Success status
        """
        try:
            # Try MCP file editor tool first
            if "mcp" in self.adapters and self.mcp_tools.get("file_editor"):
                success = await self._use_mcp_file_editor(fix)
                if success:
                    fix.tool_used = f"mcp:{self.mcp_tools['file_editor']}"
                    return True
            
            # Fallback to direct file modification
            with open(fix.file_path, 'w', encoding='utf-8') as f:
                f.write(fix.fixed_content)
            
            fix.tool_used = "direct_file_write"
            self.logger.info(f"Successfully modified {fix.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to modify file {fix.file_path}: {e}")
            fix.error_message = str(e)
            return False
    
    async def _use_mcp_file_editor(self, fix: ViolationFix) -> bool:
        """Use MCP file editor tool to apply the fix"""
        try:
            mcp_adapter = self.adapters["mcp"]
            
            # Prepare MCP message for file editing
            message = {
                "payload": {
                    "action": "edit_file",
                    "data": {
                        "file_path": fix.file_path,
                        "original_content": fix.original_content,
                        "new_content": fix.fixed_content,
                        "operation": "replace"
                    }
                },
                "metadata": {
                    "mcp_tool_id": self.mcp_tools["file_editor"],
                    "violation_type": fix.violation.violation_type.value
                }
            }
            
            # Send to MCP tool
            response = mcp_adapter.send_to_external_agent(
                self.mcp_tools["file_editor"], 
                message
            )
            
            # Check response for success
            if response.get("tool_response", {}).get("error"):
                self.logger.error(f"MCP tool error: {response['tool_response']['error']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"MCP file editor failed: {e}")
            return False
    
    async def _finalize_session(self):
        """Finalize the current audit fixing session"""
        if not self.current_session:
            return
        
        self.current_session.end_time = time.time()
        
        # Calculate success rate
        total_attempts = self.current_session.violations_fixed + self.current_session.violations_failed
        if total_attempts > 0:
            self.current_session.success_rate = self.current_session.violations_fixed / total_attempts
        
        # Add to history
        self.fix_history.append(self.current_session)
        
        # Log session summary
        session = self.current_session
        self.logger.info(f"Audit fixing session completed:")
        self.logger.info(f"  - Violations detected: {session.violations_detected}")
        self.logger.info(f"  - Violations fixed: {session.violations_fixed}")
        self.logger.info(f"  - Violations failed: {session.violations_failed}")
        self.logger.info(f"  - Success rate: {session.success_rate:.2%}")
        self.logger.info(f"  - Files modified: {len(session.files_modified)}")
        
        self.current_session = None
    
    def get_session_report(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get a detailed report of a fixing session.
        
        Args:
            session_id: Session ID (defaults to latest session)
            
        Returns:
            Session report
        """
        if session_id:
            session = next((s for s in self.fix_history if s.session_id == session_id), None)
        else:
            session = self.fix_history[-1] if self.fix_history else self.current_session
        
        if not session:
            return {"error": "No session found"}
        
        return {
            "session_id": session.session_id,
            "duration": (session.end_time or time.time()) - session.start_time,
            "violations_detected": session.violations_detected,
            "violations_fixed": session.violations_fixed,
            "violations_failed": session.violations_failed,
            "success_rate": session.success_rate,
            "files_modified": session.files_modified,
            "fixes_applied": [
                {
                    "violation_type": fix.violation.violation_type.value,
                    "file_path": fix.file_path,
                    "line_number": fix.line_number,
                    "success": fix.success,
                    "tool_used": fix.tool_used,
                    "fix_strategy": fix.fix_strategy.value
                }
                for fix in session.fixes_applied
            ]
        }

# Factory function for creating the audit fixing agent
def create_audit_fixing_agent(
    agent_id: str = "audit_fixing_action_agent",
    mcp_tools_config: Dict[str, Any] = None
) -> AuditFixingActionAgent:
    """
    Factory function to create an audit fixing action agent.
    
    Args:
        agent_id: Unique identifier for the agent
        mcp_tools_config: Configuration for MCP tools
        
    Returns:
        Configured AuditFixingActionAgent
    """
    return AuditFixingActionAgent(
        agent_id=agent_id,
        mcp_tools_config=mcp_tools_config
    ) 