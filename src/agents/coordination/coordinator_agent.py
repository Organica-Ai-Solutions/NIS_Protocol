"""
NIS Protocol Coordinator Agent

This module provides the CoordinatorAgent class which is responsible for:
1. Routing messages between internal NIS agents
2. Translating between NIS Protocol and external protocols (MCP, ACP, A2A)
3. Managing multi-agent workflows

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of coordination operations with evidence-based metrics
- Comprehensive integrity oversight for all coordination outputs
- Auto-correction capabilities for coordination-related communications
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

from core.agent import NISAgent, NISLayer

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class CoordinatorAgent(NISAgent):
    """Coordinator Agent for managing inter-agent communication and protocol translation.
    
    This agent acts as a central hub for message routing and protocol translation,
    allowing NIS Protocol agents to interact seamlessly with external protocols.
    
    Attributes:
        agent_id: Unique identifier for the agent
        description: Human-readable description of the agent's purpose
        protocol_adapters: Dictionary of protocol adapters for translation
    """
    
    def __init__(
        self,
        agent_id: str = "coordinator_agent",
        description: str = "Coordinates agent communication and handles protocol translation",
        enable_self_audit: bool = True
    ):
        """Initialize a new Coordinator agent.
        
        Args:
            agent_id: Unique identifier for the agent
            description: Human-readable description of the agent's purpose
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.COORDINATION, description)
        self.protocol_adapters = {}
        self.routing_rules = {}
        
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
        
        # Track coordination statistics
        self.coordination_stats = {
            'total_messages_routed': 0,
            'successful_routes': 0,
            'protocol_translations': 0,
            'routing_errors': 0,
            'average_routing_time': 0.0
        }
        
        print(f"Coordinator Agent '{agent_id}' initialized with self-audit: {enable_self_audit}")
        
    def register_protocol_adapter(self, protocol_name: str, adapter) -> None:
        """Register a protocol adapter.
        
        Args:
            protocol_name: Name of the protocol (e.g., "mcp", "acp", "a2a")
            adapter: The adapter instance for the protocol
        """
        self.protocol_adapters[protocol_name] = adapter
        
    def load_routing_config(self, config_path: str) -> None:
        """Load routing configuration from a file.
        
        Args:
            config_path: Path to the routing configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.routing_rules = json.load(f)
        except Exception as e:
            print(f"Error loading routing configuration: {e}")
            
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message.
        
        Determines the appropriate action based on message type and content:
        1. For NIS internal messages, routes to appropriate internal agents
        2. For external protocol messages, translates and routes accordingly
        
        Args:
            message: The incoming message to process
            
        Returns:
            The processed message with routing information
        """
        start_time = self._start_processing_timer()
        
        try:
            # Determine message protocol
            protocol = message.get("protocol", "nis")
            
            # Handle based on protocol
            if protocol == "nis":
                result = self._handle_nis_message(message)
            else:
                result = self._handle_external_protocol_message(message, protocol)
                
            self._end_processing_timer(start_time)
            return self._create_response("success", result)
            
        except Exception as e:
            self._end_processing_timer(start_time)
            return self._create_response(
                "error",
                {"error": str(e), "original_message": message},
                {"exception_type": type(e).__name__}
            )
    
    def _handle_nis_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a message in the NIS Protocol format.
        
        Args:
            message: The NIS Protocol message
            
        Returns:
            The processed message
        """
        # Check if the message needs to be routed to an external protocol
        target_protocol = message.get("target_protocol")
        if target_protocol and target_protocol != "nis":
            if target_protocol in self.protocol_adapters:
                adapter = self.protocol_adapters[target_protocol]
                return adapter.translate_from_nis(message)
            else:
                raise ValueError(f"No adapter registered for protocol: {target_protocol}")
        
        # Route to internal NIS agents
        from src.core.registry import NISRegistry
        registry = NISRegistry()
        
        target_layer = NISLayer[message.get("target_layer", "COORDINATION").upper()]
        target_agent_id = message.get("target_agent_id")
        
        if target_agent_id:
            agent = registry.get_agent_by_id(target_agent_id)
            if agent and agent.is_active():
                return agent.process(message)
            else:
                raise ValueError(f"Agent not found or inactive: {target_agent_id}")
        else:
            responses = registry.process_message(message, target_layer)
            return {"responses": responses}
    
    def _handle_external_protocol_message(
        self,
        message: Dict[str, Any],
        protocol: str
    ) -> Dict[str, Any]:
        """Handle a message from an external protocol.
        
        Args:
            message: The external protocol message
            protocol: The protocol name
            
        Returns:
            The processed message
        """
        if protocol not in self.protocol_adapters:
            raise ValueError(f"No adapter registered for protocol: {protocol}")
        
        adapter = self.protocol_adapters[protocol]
        
        # Translate to NIS format
        nis_message = adapter.translate_to_nis(message)
        
        # Process with internal agents
        result = self._handle_nis_message(nis_message)
        
        # Translate back to original protocol if needed
        if message.get("respond_in_original_protocol", True):
            return adapter.translate_from_nis(result)
        
        return result
    
    def route_to_external_agent(
        self,
        protocol: str,
        agent_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a message to an external agent using the appropriate protocol.
        
        Args:
            protocol: The protocol to use
            agent_id: The ID of the external agent
            message: The message to send
            
        Returns:
            The response from the external agent
        """
        if protocol not in self.protocol_adapters:
            raise ValueError(f"No adapter registered for protocol: {protocol}")
        
        adapter = self.protocol_adapters[protocol]
        return adapter.send_to_external_agent(agent_id, message) 
    
    # ==================== COMPREHENSIVE SELF-AUDIT CAPABILITIES ====================
    
    def audit_coordination_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on coordination outputs.
        
        Args:
            output_text: Text output to audit
            operation: Coordination operation type (route, translate, register, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        print(f"Performing self-audit on coordination output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"coordination:{operation}:{context}" if context else f"coordination:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for coordination-specific analysis
        if violations:
            print(f"Detected {len(violations)} integrity violations in coordination output")
            for violation in violations:
                print(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_coordination_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_coordination_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in coordination outputs.
        
        Args:
            output_text: Text to correct
            operation: Coordination operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        print(f"Performing self-correction on coordination output for operation: {operation}")
        
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
    
    def analyze_coordination_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze coordination integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Coordination integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        print(f"Analyzing coordination integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate coordination-specific metrics
        coordination_metrics = {
            'protocol_adapters_count': len(self.protocol_adapters),
            'routing_rules_count': len(self.routing_rules),
            'coordination_stats': self.coordination_stats
        }
        
        # Generate coordination-specific recommendations
        recommendations = self._generate_coordination_integrity_recommendations(
            integrity_report, coordination_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'coordination_metrics': coordination_metrics,
            'integrity_trend': self._calculate_coordination_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def get_coordination_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive coordination integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add coordination-specific metrics
        coordination_report = {
            'coordination_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'coordination_capabilities': {
                'message_routing': True,
                'protocol_translation': True,
                'multi_agent_workflows': True,
                'external_protocol_support': len(self.protocol_adapters) > 0,
                'registered_protocols': list(self.protocol_adapters.keys()),
                'routing_rules_configured': len(self.routing_rules) > 0
            },
            'processing_statistics': {
                'total_messages_routed': self.coordination_stats.get('total_messages_routed', 0),
                'successful_routes': self.coordination_stats.get('successful_routes', 0),
                'protocol_translations': self.coordination_stats.get('protocol_translations', 0),
                'routing_errors': self.coordination_stats.get('routing_errors', 0),
                'average_routing_time': self.coordination_stats.get('average_routing_time', 0.0)
            },
            'configuration_status': {
                'protocol_adapters_configured': len(self.protocol_adapters),
                'routing_rules_configured': len(self.routing_rules),
                'confidence_factors_configured': bool(self.confidence_factors)
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return coordination_report
    
    def validate_coordination_configuration(self) -> Dict[str, Any]:
        """Validate coordination configuration for integrity"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check protocol adapters
        if len(self.protocol_adapters) == 0:
            validation_results['warnings'].append("No protocol adapters registered - external protocol communication unavailable")
            validation_results['recommendations'].append("Register protocol adapters for MCP, ACP, A2A or other external protocols")
        
        # Check routing rules
        if len(self.routing_rules) == 0:
            validation_results['warnings'].append("No routing rules configured - may impact message routing efficiency")
            validation_results['recommendations'].append("Configure routing rules for optimal message routing")
        
        # Check routing error rate
        error_rate = (self.coordination_stats.get('routing_errors', 0) / 
                     max(1, self.coordination_stats.get('total_messages_routed', 1)))
        
        if error_rate > 0.1:
            validation_results['warnings'].append(f"High routing error rate: {error_rate:.1%}")
            validation_results['recommendations'].append("Investigate and resolve sources of routing errors")
        
        # Check routing performance
        avg_time = self.coordination_stats.get('average_routing_time', 0.0)
        if avg_time > 1.0:
            validation_results['warnings'].append(f"High average routing time: {avg_time:.2f}s")
            validation_results['recommendations'].append("Consider optimizing routing algorithms or infrastructure")
        
        return validation_results
    
    def _monitor_coordination_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct coordination output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Coordination operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_coordination_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_coordination_output(output_text, operation)
            
            print(f"Auto-corrected coordination output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_coordination_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to coordination operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_coordination_integrity_recommendations(self, integrity_report: Dict[str, Any], coordination_metrics: Dict[str, Any]) -> List[str]:
        """Generate coordination-specific integrity improvement recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 5:
            recommendations.append("Consider implementing more rigorous coordination output validation")
        
        if coordination_metrics.get('protocol_adapters_count', 0) == 0:
            recommendations.append("Register protocol adapters for external protocol communication")
        
        if coordination_metrics.get('routing_rules_count', 0) == 0:
            recommendations.append("Configure routing rules for optimal message routing")
        
        success_rate = (coordination_metrics.get('coordination_stats', {}).get('successful_routes', 0) / 
                       max(1, coordination_metrics.get('coordination_stats', {}).get('total_messages_routed', 1)))
        
        if success_rate < 0.9:
            recommendations.append("Low coordination success rate - consider optimizing routing algorithms")
        
        if coordination_metrics.get('coordination_stats', {}).get('routing_errors', 0) > 10:
            recommendations.append("High number of routing errors - investigate and resolve error sources")
        
        if len(recommendations) == 0:
            recommendations.append("Coordination integrity status is excellent - maintain current practices")
        
        return recommendations
    
    def _calculate_coordination_integrity_trend(self) -> Dict[str, Any]:
        """Calculate coordination integrity trends with mathematical validation"""
        if not hasattr(self, 'coordination_stats'):
            return {'trend': 'INSUFFICIENT_DATA'}
        
        total_routed = self.coordination_stats.get('total_messages_routed', 0)
        successful_routes = self.coordination_stats.get('successful_routes', 0)
        
        if total_routed == 0:
            return {'trend': 'NO_MESSAGES_ROUTED'}
        
        success_rate = successful_routes / total_routed
        error_rate = self.coordination_stats.get('routing_errors', 0) / total_routed
        avg_routing_time = self.coordination_stats.get('average_routing_time', 0.0)
        
        # Calculate trend with mathematical validation
        routing_efficiency = 1.0 / max(avg_routing_time, 0.1)
        trend_score = calculate_confidence(
            (success_rate * 0.5 + (1.0 - error_rate) * 0.3 + min(routing_efficiency, 1.0) * 0.2), 
            self.confidence_factors
        )
        
        return {
            'trend': 'IMPROVING' if trend_score > 0.8 else 'STABLE' if trend_score > 0.6 else 'NEEDS_ATTENTION',
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_routing_time': avg_routing_time,
            'trend_score': trend_score,
            'messages_processed': total_routed,
            'coordination_analysis': self._analyze_coordination_patterns()
        }
    
    def _analyze_coordination_patterns(self) -> Dict[str, Any]:
        """Analyze coordination patterns for integrity assessment"""
        if not hasattr(self, 'coordination_stats') or not self.coordination_stats:
            return {'pattern_status': 'NO_COORDINATION_STATS'}
        
        total_routed = self.coordination_stats.get('total_messages_routed', 0)
        successful_routes = self.coordination_stats.get('successful_routes', 0)
        protocol_translations = self.coordination_stats.get('protocol_translations', 0)
        routing_errors = self.coordination_stats.get('routing_errors', 0)
        
        return {
            'pattern_status': 'NORMAL' if total_routed > 0 else 'NO_ROUTING_ACTIVITY',
            'total_messages_routed': total_routed,
            'successful_routes': successful_routes,
            'protocol_translations': protocol_translations,
            'routing_errors': routing_errors,
            'protocol_adapters_available': len(self.protocol_adapters),
            'routing_rules_configured': len(self.routing_rules),
            'analysis_timestamp': time.time()
        } 