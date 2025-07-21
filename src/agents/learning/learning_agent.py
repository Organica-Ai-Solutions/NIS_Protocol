"""
Learning Agent Base Class

Provides the foundation for learning and adaptation in the NIS Protocol.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of learning operations with evidence-based metrics
- Comprehensive integrity oversight for all learning outputs
- Auto-correction capabilities for learning-related communications
"""

from typing import Dict, Any, Optional, List
import time
import logging
from collections import defaultdict

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class LearningAgent(NISAgent):
    """
    Base class for learning agents in the NIS Protocol.
    
    This agent provides the foundation for implementing different learning
    strategies and mechanisms for system adaptation.
    """
    
    def __init__(
        self,
        agent_id: str,
        description: str = "Base learning agent",
        emotional_state: Optional[EmotionalState] = None,
        learning_rate: float = 0.1,
        enable_self_audit: bool = True
    ):
        """
        Initialize the learning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            learning_rate: Base learning rate for parameter updates
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.LEARNING, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.learning_rate = learning_rate
        
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
        
        # Set up logging
        self.logger = logging.getLogger(f"nis_learning_agent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Learning Agent initialized with self-audit: {enable_self_audit}")
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a learning-related request with integrated self-audit monitoring.
        
        Args:
            message: Message containing learning operation
                'operation': Operation to perform
                + Additional parameters based on operation
                
        Returns:
            Result of the learning operation with integrity monitoring
        """
        operation = message.get("operation", "").lower()
        
        # Route to appropriate handler with self-audit monitoring
        try:
            if operation == "update":
                result = self._update_parameters(message)
            elif operation == "get_params":
                result = self._get_parameters(message)
            elif operation == "reset":
                result = self._reset_parameters(message)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Apply self-audit monitoring to all responses
            if self.enable_self_audit and result:
                result = self._apply_learning_integrity_monitoring(result, operation)
            
            return result
            
        except Exception as e:
            error_response = {
                "status": "error", 
                "error": f"Learning operation failed: {str(e)}",
                "operation": operation,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            # Apply self-audit monitoring to exception response
            if self.enable_self_audit:
                error_text = error_response.get("error", "")
                error_response["error"] = self._monitor_learning_output_integrity(error_text, f"{operation}_error")
            
            self.logger.error(f"Learning operation {operation} failed: {str(e)}")
            
            return error_response
    
    def _update_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update learning parameters based on feedback.
        
        Args:
            message: Message with update parameters
            
        Returns:
            Update operation result
        """
        raise NotImplementedError("Subclasses must implement _update_parameters()")
    
    def _get_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current learning parameters.
        
        Args:
            message: Message with parameter query
            
        Returns:
            Current parameter values
        """
        raise NotImplementedError("Subclasses must implement _get_parameters()")
    
    def _reset_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset learning parameters to initial values.
        
        Args:
            message: Message with reset parameters
            
        Returns:
            Reset operation result
        """
        raise NotImplementedError("Subclasses must implement _reset_parameters()")
    
    def adjust_learning_rate(self, factor: float) -> None:
        """
        Adjust the learning rate by a factor.
        
        Args:
            factor: Multiplier for the learning rate
        """
        self.learning_rate *= factor 
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_learning_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on learning operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Learning operation type (update, get_params, reset, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on learning output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"learning:{operation}:{context}" if context else f"learning:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for learning-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in learning output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_learning_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_learning_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in learning outputs.
        
        Args:
            output_text: Text to correct
            operation: Learning operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on learning output for operation: {operation}")
        
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
    
    def get_learning_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add learning-specific metrics
        learning_report = {
            'learning_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'learning_configuration': {
                'learning_rate': self.learning_rate,
                'agent_type': 'base_learning_agent'
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return learning_report
    
    def _monitor_learning_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct learning output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Learning operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_learning_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_learning_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected learning output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _apply_learning_integrity_monitoring(self, result: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Apply integrity monitoring to response data"""
        if not self.enable_self_audit or not result:
            return result
        
        # Monitor text fields in the response
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 10:  # Only monitor substantial text
                monitored_text = self._monitor_learning_output_integrity(value, f"{operation}_{key}")
                if monitored_text != value:
                    result[key] = monitored_text
        
        return result
    
    def _categorize_learning_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to learning operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories) 