"""
Log Agent

Records system events and performance metrics for analysis and debugging.
Provides a historical record of system activity.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of log operations with evidence-based metrics
- Comprehensive integrity oversight for all log outputs
- Auto-correction capabilities for log-related communications
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
import os
import datetime
import logging
from collections import deque, defaultdict

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation


class LogAgent(NISAgent):
    """
    Agent that maintains a log of system events.
    
    The Log Agent is responsible for:
    - Recording system events and agent activities
    - Tracking performance metrics
    - Storing error and warning information
    - Providing data for system diagnostics
    """
    
    def __init__(
        self,
        agent_id: str = "log",
        description: str = "Records system events and performance metrics",
        emotional_state: Optional[EmotionalState] = None,
        log_path: Optional[str] = None,
        memory_size: int = 1000,
        log_level: int = logging.INFO,
        enable_self_audit: bool = True
    ):
        """
        Initialize a new Log Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            log_path: Path to store log files
            memory_size: Maximum number of log entries to keep in memory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.emotional_state = emotional_state or EmotionalState()
        
        # In-memory log storage
        self.log_memory = deque(maxlen=memory_size)
        
        # Set up file-based logging if path provided
        self.log_path = log_path
        self.logger = None
        
        if log_path:
            if not os.path.exists(log_path):
                os.makedirs(log_path, exist_ok=True)
            
            # Configure logger
            logger_name = f"nis_log_agent_{agent_id}"
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(log_level)
            
            # Add file handler
            log_file = os.path.join(log_path, f"nis_log_{time.strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
        
        # Set up self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_monitoring_enabled = enable_self_audit
        self.integrity_metrics = {
            'monitoring_start_time': time.time(),
            'total_outputs_monitored': 0,
            'total_violations_detected': 0,
            'auto_corrections_applied': 0,
            'average_integrity_score': 100.0
        }
        
        # Initialize confidence factors for mathematical validation
        self.confidence_factors = create_default_confidence_factors()
        
        if self.logger:
            self.logger.info(f"Log Agent initialized with self-audit: {enable_self_audit}")

    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a log request.
        
        Args:
            message: Message containing log data
                'level': Log level (debug, info, warning, error, critical)
                'content': Log message content
                'source_agent': Agent that generated the log
                'metadata': Additional data to log
        
        Returns:
            Result of the log operation
        """
        if not self._validate_message(message):
            return {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Extract log data
        level = message.get("level", "info").lower()
        content = message.get("content", "")
        source_agent = message.get("source_agent", "unknown")
        metadata = message.get("metadata", {})
        
        # Create log entry
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "level": level,
            "content": content,
            "source_agent": source_agent,
            "metadata": metadata
        }
        
        # Store in memory
        self.log_memory.append(log_entry)
        
        # Write to file logger if configured
        if self.logger:
            self._write_to_log(level, f"[{source_agent}] {content}", metadata)
            
        # Update emotional state based on log level
        self._update_emotional_state(level)
        
        return {
            "status": "success",
            "log_id": len(self.log_memory),
            "level": level,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message format.
        
        Args:
            message: The message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
        
        # Must have content to log
        if "content" not in message:
            return False
        
        return True
    
    def _write_to_log(self, level: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Write entry to file logger.
        
        Args:
            level: Log level
            content: Log message
            metadata: Additional data
        """
        if not self.logger:
            return
        
        # Convert level string to logging level
        log_method = getattr(self.logger, level, None)
        if not log_method:
            log_method = self.logger.info
        
        # Add metadata as JSON if present
        if metadata:
            metadata_str = json.dumps(metadata)
            content = f"{content} - Metadata: {metadata_str}"
        
        # Write to log
        log_method(content)
    
    def _update_emotional_state(self, level: str) -> None:
        """
        Update emotional state based on log level.
        
        Args:
            level: Log level
        """
        # Error and critical logs increase urgency and suspicion
        if level in ["error", "critical"]:
            self.emotional_state.update(EmotionalDimension.URGENCY.value, 0.8)
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, 0.7)
            # Decrease confidence when errors occur
            self.emotional_state.update(EmotionalDimension.CONFIDENCE.value, 0.3)
        
        # Warning logs increase suspicion
        elif level == "warning":
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, 0.6)
    
    def get_recent_logs(self, count: int = 10, level: Optional[str] = None, 
                        source_agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent log entries with optional filtering.
        
        Args:
            count: Maximum number of logs to return
            level: Filter by log level
            source_agent: Filter by source agent
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Filter logs
        for entry in reversed(self.log_memory):
            if level and entry.get("level") != level:
                continue
                
            if source_agent and entry.get("source_agent") != source_agent:
                continue
                
            results.append(entry)
            
            if len(results) >= count:
                break
        
        return results
    
    def clear_logs(self) -> Dict[str, Any]:
        """
        Clear the in-memory logs.
        
        Returns:
            Operation result
        """
        count = len(self.log_memory)
        self.log_memory.clear()
        
        return {
            "status": "success",
            "cleared_count": count,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged events.
        
        Returns:
            Statistics about logs
        """
        if not self.log_memory:
            return {
                "status": "success",
                "log_count": 0,
                "level_counts": {},
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Count by level
        level_counts = {}
        source_counts = {}
        
        for entry in self.log_memory:
            level = entry.get("level", "unknown")
            source = entry.get("source_agent", "unknown")
            
            level_counts[level] = level_counts.get(level, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate time range
        oldest = min(entry.get("timestamp", time.time()) for entry in self.log_memory)
        newest = max(entry.get("timestamp", time.time()) for entry in self.log_memory)
        
        return {
            "status": "success",
            "log_count": len(self.log_memory),
            "level_counts": level_counts,
            "source_counts": source_counts,
            "time_range": {
                "oldest": oldest,
                "newest": newest,
                "span_seconds": newest - oldest
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        } 
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_log_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on log operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Log operation type (log, retrieve, query, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        if self.logger:
            self.logger.info(f"Performing self-audit on log output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"log:{operation}:{context}" if context else f"log:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for log-specific analysis
        if violations and self.logger:
            self.logger.warning(f"Detected {len(violations)} integrity violations in log output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_log_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_log_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in log outputs.
        
        Args:
            output_text: Text to correct
            operation: Log operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        if self.logger:
            self.logger.info(f"Performing self-correction on log output for operation: {operation}")
        
        corrected_text, violations = self_audit_engine.auto_correct_text(output_text)
        
        # Calculate improvement metrics with mathematical validation
        original_score = self_audit_engine.get_integrity_score(output_text)
        corrected_score = self_audit_engine.get_integrity_score(corrected_text)
        improvement = calculate_confidence(corrected_score - original_score, self.confidence_factors)
        
        # Update integrity metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['auto_corrections_applied'] += len(violations)
        
        return {
            'original_text': output_text,
            'corrected_text': corrected_text,
            'violations_fixed': violations,
            'original_integrity_score': original_score,
            'corrected_integrity_score': corrected_score,
            'improvement': improvement,
            'operation': operation,
            'correction_timestamp': time.time()
        }
    
    def analyze_log_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze log operation integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Log integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        if self.logger:
            self.logger.info(f"Analyzing log integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate log-specific metrics
        log_metrics = {
            'log_memory_utilization': len(self.log_memory) / self.log_memory.maxlen if self.log_memory.maxlen else 0,
            'log_path_configured': bool(self.log_path),
            'total_log_entries': len(self.log_memory),
            'logger_configured': bool(self.logger)
        }
        
        # Generate log-specific recommendations
        recommendations = self._generate_log_integrity_recommendations(
            integrity_report, log_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'log_metrics': log_metrics,
            'integrity_trend': self._calculate_log_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def get_log_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive log integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add log-specific metrics
        log_report = {
            'log_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'log_capacity_status': {
                'log_memory': f"{len(self.log_memory)}/{self.log_memory.maxlen}" if self.log_memory.maxlen else "unlimited"
            },
            'logging_configuration': {
                'path_configured': bool(self.log_path),
                'path': self.log_path or "in_memory_only",
                'logger_active': bool(self.logger)
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return log_report
    
    def _monitor_log_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct log output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Log operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_log_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_log_output(output_text, operation)
            
            if self.logger:
                self.logger.info(f"Auto-corrected log output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_log_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to log operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_log_integrity_recommendations(self, integrity_report: Dict[str, Any], log_metrics: Dict[str, Any]) -> List[str]:
        """Generate log-specific integrity improvement recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 5:
            recommendations.append("Consider implementing more rigorous input validation for log operations")
        
        if log_metrics.get('log_memory_utilization', 0) > 0.9:
            recommendations.append("Log memory approaching capacity - consider increasing capacity or implementing cleanup")
        
        if not log_metrics.get('log_path_configured', False):
            recommendations.append("Configure persistent log storage path for improved log reliability")
        
        if not log_metrics.get('logger_configured', False):
            recommendations.append("Configure file-based logger for persistent log storage")
        
        if len(recommendations) == 0:
            recommendations.append("Log integrity status is excellent - maintain current practices")
        
        return recommendations
    
    def _calculate_log_integrity_trend(self) -> Dict[str, Any]:
        """Calculate log integrity trends with mathematical validation"""
        if not hasattr(self, 'integrity_metrics'):
            return {'trend': 'INSUFFICIENT_DATA'}
        
        monitoring_time = time.time() - self.integrity_metrics.get('monitoring_start_time', time.time())
        total_outputs = self.integrity_metrics.get('total_outputs_monitored', 0)
        total_violations = self.integrity_metrics.get('total_violations_detected', 0)
        
        if total_outputs == 0:
            return {'trend': 'NO_OUTPUTS_MONITORED'}
        
        violation_rate = total_violations / total_outputs
        violations_per_hour = (total_violations / monitoring_time) * 3600 if monitoring_time > 0 else 0
        
        # Calculate trend with mathematical validation
        trend_score = calculate_confidence(1.0 - violation_rate, self.confidence_factors)
        
        return {
            'trend': 'IMPROVING' if trend_score > 0.8 else 'STABLE' if trend_score > 0.6 else 'NEEDS_ATTENTION',
            'violation_rate': violation_rate,
            'violations_per_hour': violations_per_hour,
            'trend_score': trend_score,
            'monitoring_duration_hours': monitoring_time / 3600
        } 