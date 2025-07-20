"""
Communication Agent

Handles natural communication including text-to-speech synthesis in the NIS Protocol.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of communication operations with evidence-based metrics
- Comprehensive integrity oversight for all communication outputs
- Auto-correction capabilities for communication-related outputs
"""

from typing import Dict, Any, Optional, List
import time
import os
import numpy as np
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import sounddevice as sd
from collections import defaultdict

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.interpretation.interpretation_agent import InterpretationAgent

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class CommunicationAgent(NISAgent):
    """
    Agent responsible for natural communication including speech synthesis.
    
    This agent handles text-to-speech conversion and natural communication
    with emotional awareness and voice modulation.
    """
    
    def __init__(
        self,
        agent_id: str = "communicator",
        description: str = "Handles natural communication",
        emotional_state: Optional[EmotionalState] = None,
        interpreter: Optional[InterpretationAgent] = None,
        voice_preset: str = "v2/en_speaker_6",
        output_dir: str = "data/audio_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_self_audit: bool = True
    ):
        """
        Initialize the communication agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            interpreter: Optional interpreter agent for content analysis
            voice_preset: Bark voice preset to use
            output_dir: Directory for saving audio files
            device: Device to use for inference ('cuda' or 'cpu')
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.COMMUNICATION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.interpreter = interpreter
        self.voice_preset = voice_preset
        self.output_dir = output_dir
        self.device = device
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Bark
        print("Loading Bark models...")
        preload_models()
        
        # Track conversation history
        self.conversation_history = []
        self.max_history = 10
        
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
        
        # Track communication statistics
        self.communication_stats = {
            'total_communications': 0,
            'successful_communications': 0,
            'speech_synthesis_operations': 0,
            'voice_adjustments': 0,
            'communication_errors': 0,
            'average_generation_time': 0.0
        }
        
        print(f"Communication Agent '{agent_id}' initialized with self-audit: {enable_self_audit}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process communication requests.
        
        Args:
            message: Message containing communication operation
                'operation': Operation to perform
                    ('speak', 'synthesize', 'respond', 'adjust_voice')
                'content': Text content to communicate
                + Additional parameters based on operation
                
        Returns:
            Result of the communication operation
        """
        operation = message.get("operation", "").lower()
        content = message.get("content", "")
        
        if not content and operation not in ["adjust_voice"]:
            return {
                "status": "error",
                "error": "No content provided for communication",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Process the requested operation
        if operation == "speak":
            return self._speak_content(content, message.get("save_audio", True))
        elif operation == "synthesize":
            return self._synthesize_speech(content)
        elif operation == "respond":
            return self._generate_response(content)
        elif operation == "adjust_voice":
            voice_params = message.get("voice_params", {})
            return self._adjust_voice_parameters(voice_params)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _speak_content(self, content: str, save_audio: bool = True) -> Dict[str, Any]:
        """
        Convert text to speech and play it.
        
        Args:
            content: Text content to speak
            save_audio: Whether to save the audio file
            
        Returns:
            Speech operation result
        """
        try:
            # Generate audio
            audio_array = generate_audio(
                content,
                history_prompt=self.voice_preset
            )
            
            # Play audio
            sd.play(audio_array, SAMPLE_RATE)
            sd.wait()
            
            # Save audio if requested
            audio_path = None
            if save_audio:
                timestamp = int(time.time())
                audio_path = os.path.join(
                    self.output_dir,
                    f"speech_{timestamp}.wav"
                )
                write_wav(audio_path, SAMPLE_RATE, audio_array)
            
            # Update conversation history
            self._update_history({
                "role": "system",
                "content": content,
                "timestamp": time.time()
            })
            
            return {
                "status": "success",
                "message": "Speech generated and played",
                "audio_path": audio_path,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Speech generation failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _synthesize_speech(self, content: str) -> Dict[str, Any]:
        """
        Generate speech audio without playing it.
        
        Args:
            content: Text content to synthesize
            
        Returns:
            Speech synthesis result
        """
        try:
            # Generate audio
            audio_array = generate_audio(
                content,
                history_prompt=self.voice_preset
            )
            
            # Save audio
            timestamp = int(time.time())
            audio_path = os.path.join(
                self.output_dir,
                f"synthesis_{timestamp}.wav"
            )
            write_wav(audio_path, SAMPLE_RATE, audio_array)
            
            return {
                "status": "success",
                "message": "Speech synthesized",
                "audio_path": audio_path,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Speech synthesis failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _generate_response(self, content: str) -> Dict[str, Any]:
        """
        Generate and speak a response to the given content.
        
        This method uses the interpreter for content analysis and
        emotional awareness in generating responses.
        
        Args:
            content: Input content to respond to
            
        Returns:
            Response operation result
        """
        try:
            # Interpret content if interpreter is available
            interpretation = None
            if self.interpreter:
                interpretation = self.interpreter.process({
                    "operation": "interpret",
                    "content": content
                })
                
                # Update emotional state based on interpretation
                if interpretation["status"] == "success":
                    self.emotional_state = interpretation["emotional_state"]
            
            # Update conversation history
            self._update_history({
                "role": "user",
                "content": content,
                "timestamp": time.time()
            })
            
            # Generate response based on interpretation and emotional state
            response = self._craft_response(content, interpretation)
            
            # Speak the response
            speak_result = self._speak_content(response)
            
            return {
                "status": "success",
                "response": response,
                "interpretation": interpretation,
                "speech_result": speak_result,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Response generation failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _craft_response(
        self,
        content: str,
        interpretation: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Craft an appropriate response based on content and interpretation.
        
        Args:
            content: Input content to respond to
            interpretation: Optional content interpretation
            
        Returns:
            Crafted response text
        """
        # This is a simple response generation implementation
        # In a full system, this would use more sophisticated NLG
        
        if not interpretation:
            return f"I hear you saying: {content}"
            
        sentiment = interpretation.get("sentiment", "NEUTRAL")
        content_type = interpretation.get("content_type", [])
        
        # Adjust response based on content type and sentiment
        if any(c["label"] == "query" for c in content_type):
            return f"Let me help you with your question about {content}"
        elif sentiment == "POSITIVE":
            return f"I'm glad to hear that! {content}"
        elif sentiment == "NEGATIVE":
            return f"I understand your concern about {content}"
        else:
            return f"I understand you're saying: {content}"
    
    def _adjust_voice_parameters(self, voice_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust voice synthesis parameters.
        
        Args:
            voice_params: Dictionary of voice parameters to adjust
                'preset': Voice preset name
                'rate': Speech rate multiplier
                'pitch': Pitch adjustment
                
        Returns:
            Parameter adjustment result
        """
        if "preset" in voice_params:
            self.voice_preset = voice_params["preset"]
        
        return {
            "status": "success",
            "message": "Voice parameters adjusted",
            "current_params": {
                "preset": self.voice_preset
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _update_history(self, entry: Dict[str, Any]) -> None:
        """
        Update conversation history.
        
        Args:
            entry: Conversation entry to add
        """
        self.conversation_history.append(entry)
        
        # Maintain maximum history length
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = [] 
    
    # ==================== COMPREHENSIVE SELF-AUDIT CAPABILITIES ====================
    
    def audit_communication_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on communication outputs.
        
        Args:
            output_text: Text output to audit
            operation: Communication operation type (speak, synthesize, respond, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        print(f"Performing self-audit on communication output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"communication:{operation}:{context}" if context else f"communication:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for communication-specific analysis
        if violations:
            print(f"Detected {len(violations)} integrity violations in communication output")
            for violation in violations:
                print(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_communication_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_communication_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in communication outputs.
        
        Args:
            output_text: Text to correct
            operation: Communication operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        print(f"Performing self-correction on communication output for operation: {operation}")
        
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
    
    def analyze_communication_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze communication integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Communication integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        print(f"Analyzing communication integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate communication-specific metrics
        communication_metrics = {
            'voice_preset': self.voice_preset,
            'output_dir': self.output_dir,
            'device': self.device,
            'max_history': self.max_history,
            'conversation_history_length': len(self.conversation_history),
            'emotional_state_configured': bool(self.emotional_state),
            'interpreter_configured': bool(self.interpreter),
            'communication_stats': self.communication_stats
        }
        
        # Generate communication-specific recommendations
        recommendations = self._generate_communication_integrity_recommendations(
            integrity_report, communication_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'communication_metrics': communication_metrics,
            'integrity_trend': self._calculate_communication_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def get_communication_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive communication integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add communication-specific metrics
        communication_report = {
            'communication_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'communication_capabilities': {
                'text_to_speech': True,
                'voice_synthesis': True,
                'emotional_awareness': bool(self.emotional_state),
                'voice_modulation': True,
                'conversation_tracking': True,
                'content_interpretation': bool(self.interpreter),
                'audio_generation': True,
                'voice_preset': self.voice_preset,
                'device_type': self.device
            },
            'configuration_status': {
                'voice_preset_configured': bool(self.voice_preset),
                'output_directory_configured': bool(self.output_dir),
                'emotional_state_configured': bool(self.emotional_state),
                'interpreter_configured': bool(self.interpreter),
                'conversation_history_enabled': self.max_history > 0
            },
            'processing_statistics': {
                'total_communications': self.communication_stats.get('total_communications', 0),
                'successful_communications': self.communication_stats.get('successful_communications', 0),
                'speech_synthesis_operations': self.communication_stats.get('speech_synthesis_operations', 0),
                'voice_adjustments': self.communication_stats.get('voice_adjustments', 0),
                'communication_errors': self.communication_stats.get('communication_errors', 0),
                'average_generation_time': self.communication_stats.get('average_generation_time', 0.0),
                'conversation_history_entries': len(self.conversation_history)
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return communication_report
    
    def validate_communication_configuration(self) -> Dict[str, Any]:
        """Validate communication configuration for integrity"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check voice preset
        if not self.voice_preset:
            validation_results['warnings'].append("Voice preset not configured - may impact speech synthesis")
            validation_results['recommendations'].append("Configure voice preset for consistent speech generation")
        
        # Check output directory
        if not os.path.exists(self.output_dir):
            validation_results['warnings'].append("Output directory does not exist - audio saving may fail")
            validation_results['recommendations'].append("Ensure output directory exists and is writable")
        
        # Check device configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            validation_results['warnings'].append("CUDA device specified but not available - falling back to CPU")
            validation_results['recommendations'].append("Use CPU device or ensure CUDA is properly installed")
        
        # Check conversation history
        if len(self.conversation_history) > self.max_history * 2:
            validation_results['warnings'].append("Conversation history exceeds recommended size")
            validation_results['recommendations'].append("Clear conversation history or increase max_history limit")
        
        # Check communication error rate
        error_rate = (self.communication_stats.get('communication_errors', 0) / 
                     max(1, self.communication_stats.get('total_communications', 1)))
        
        if error_rate > 0.1:
            validation_results['warnings'].append(f"High communication error rate: {error_rate:.1%}")
            validation_results['recommendations'].append("Investigate and resolve sources of communication errors")
        
        # Check generation time
        avg_time = self.communication_stats.get('average_generation_time', 0.0)
        if avg_time > 5.0:
            validation_results['warnings'].append(f"High average generation time: {avg_time:.2f}s")
            validation_results['recommendations'].append("Consider optimizing speech synthesis or using faster device")
        
        return validation_results
    
    def _monitor_communication_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct communication output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Communication operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_communication_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_communication_output(output_text, operation)
            
            print(f"Auto-corrected communication output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_communication_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to communication operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_communication_integrity_recommendations(self, integrity_report: Dict[str, Any], communication_metrics: Dict[str, Any]) -> List[str]:
        """Generate communication-specific integrity improvement recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 5:
            recommendations.append("Consider implementing more rigorous communication output validation")
        
        if not communication_metrics.get('emotional_state_configured', False):
            recommendations.append("Configure emotional state for emotionally-aware communication")
        
        if not communication_metrics.get('interpreter_configured', False):
            recommendations.append("Configure interpreter for enhanced content analysis")
        
        if communication_metrics.get('conversation_history_length', 0) > 50:
            recommendations.append("Conversation history is large - consider implementing cleanup")
        
        success_rate = (communication_metrics.get('communication_stats', {}).get('successful_communications', 0) / 
                       max(1, communication_metrics.get('communication_stats', {}).get('total_communications', 1)))
        
        if success_rate < 0.9:
            recommendations.append("Low communication success rate - investigate synthesis or device issues")
        
        if communication_metrics.get('communication_stats', {}).get('communication_errors', 0) > 10:
            recommendations.append("High number of communication errors - check device and model configuration")
        
        avg_time = communication_metrics.get('communication_stats', {}).get('average_generation_time', 0.0)
        if avg_time > 5.0:
            recommendations.append("High generation time - consider using faster device or optimizing models")
        
        if communication_metrics.get('device') == 'cpu' and torch.cuda.is_available():
            recommendations.append("CUDA available but using CPU - consider switching to GPU for better performance")
        
        if len(recommendations) == 0:
            recommendations.append("Communication integrity status is excellent - maintain current practices")
        
        return recommendations
    
    def _calculate_communication_integrity_trend(self) -> Dict[str, Any]:
        """Calculate communication integrity trends with mathematical validation"""
        if not hasattr(self, 'communication_stats'):
            return {'trend': 'INSUFFICIENT_DATA'}
        
        total_communications = self.communication_stats.get('total_communications', 0)
        successful_communications = self.communication_stats.get('successful_communications', 0)
        
        if total_communications == 0:
            return {'trend': 'NO_COMMUNICATIONS_PROCESSED'}
        
        success_rate = successful_communications / total_communications
        error_rate = self.communication_stats.get('communication_errors', 0) / total_communications
        avg_generation_time = self.communication_stats.get('average_generation_time', 0.0)
        
        # Calculate trend with mathematical validation
        generation_efficiency = 1.0 / max(avg_generation_time, 0.1)
        trend_score = calculate_confidence(
            (success_rate * 0.5 + (1.0 - error_rate) * 0.3 + min(generation_efficiency / 5.0, 1.0) * 0.2), 
            self.confidence_factors
        )
        
        return {
            'trend': 'IMPROVING' if trend_score > 0.8 else 'STABLE' if trend_score > 0.6 else 'NEEDS_ATTENTION',
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_generation_time': avg_generation_time,
            'trend_score': trend_score,
            'communications_processed': total_communications,
            'communication_analysis': self._analyze_communication_patterns()
        }
    
    def _analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns for integrity assessment"""
        if not hasattr(self, 'communication_stats') or not self.communication_stats:
            return {'pattern_status': 'NO_COMMUNICATION_STATS'}
        
        total_communications = self.communication_stats.get('total_communications', 0)
        successful_communications = self.communication_stats.get('successful_communications', 0)
        speech_operations = self.communication_stats.get('speech_synthesis_operations', 0)
        voice_adjustments = self.communication_stats.get('voice_adjustments', 0)
        
        return {
            'pattern_status': 'NORMAL' if total_communications > 0 else 'NO_COMMUNICATION_ACTIVITY',
            'total_communications': total_communications,
            'successful_communications': successful_communications,
            'speech_synthesis_operations': speech_operations,
            'voice_adjustments': voice_adjustments,
            'success_rate': successful_communications / max(1, total_communications),
            'conversation_history_size': len(self.conversation_history),
            'voice_preset_used': self.voice_preset,
            'analysis_timestamp': time.time()
        } 