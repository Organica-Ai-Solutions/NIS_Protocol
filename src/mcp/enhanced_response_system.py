"""
Enhanced Tool Response System for NIS Protocol
Based on Anthropic's tool optimization research

Implements intelligent response formatting:
- Context-aware response prioritization
- Token-efficient response formats
- Meaningful information filtering
- Semantic identifier resolution
- Response truncation with guidance

Reference: https://www.anthropic.com/engineering/writing-tools-for-agents
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import uuid

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Response format options for different use cases"""
    CONCISE = "concise"      # Essential information only, minimal tokens
    DETAILED = "detailed"    # Full information with context and metadata
    STRUCTURED = "structured" # Machine-readable JSON/XML format
    NATURAL = "natural"      # Human-readable narrative format


class ContextPriority(Enum):
    """Priority levels for response information"""
    CRITICAL = "critical"    # Must always include
    HIGH = "high"           # Include unless severe token constraints
    MEDIUM = "medium"       # Include if space permits
    LOW = "low"             # Optional, exclude under token pressure


@dataclass
class ResponseElement:
    """Individual element within a tool response"""
    key: str
    value: Any
    priority: ContextPriority
    description: Optional[str] = None
    semantic_id: Optional[str] = None  # Human-readable identifier
    token_estimate: int = 0
    
    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = self._estimate_tokens()
    
    def _estimate_tokens(self) -> int:
        """Estimate token count for this element"""
        # Rough estimation: ~4 characters per token
        content = f"{self.key}: {json.dumps(self.value) if not isinstance(self.value, str) else self.value}"
        return max(1, len(content) // 4)


@dataclass
class ResponseMetadata:
    """Metadata for response generation"""
    tool_name: str
    execution_time: float
    token_limit: int
    requested_format: ResponseFormat
    context_hints: List[str] = None
    truncated: bool = False
    truncation_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.context_hints is None:
            self.context_hints = []


class ResponseProcessor(ABC):
    """Abstract base for response processors"""
    
    @abstractmethod
    def process(self, elements: List[ResponseElement], metadata: ResponseMetadata) -> Dict[str, Any]:
        """Process response elements into final format"""
        pass
    
    @abstractmethod
    def estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate token count for response"""
        pass


class ConciseResponseProcessor(ResponseProcessor):
    """Processor for concise responses - essential information only"""
    
    def process(self, elements: List[ResponseElement], metadata: ResponseMetadata) -> Dict[str, Any]:
        """Create concise response with only critical information"""
        response = {}
        token_count = 0
        
        # Sort by priority (critical first)
        sorted_elements = sorted(elements, key=lambda x: (
            0 if x.priority == ContextPriority.CRITICAL else
            1 if x.priority == ContextPriority.HIGH else
            2 if x.priority == ContextPriority.MEDIUM else 3
        ))
        
        for element in sorted_elements:
            # Always include critical elements
            if element.priority == ContextPriority.CRITICAL:
                response[element.key] = self._format_value_concise(element.value)
                token_count += element.token_estimate
            
            # Include high priority if under token limit
            elif element.priority == ContextPriority.HIGH:
                if token_count + element.token_estimate <= metadata.token_limit * 0.8:
                    response[element.key] = self._format_value_concise(element.value)
                    token_count += element.token_estimate
                else:
                    metadata.truncated = True
                    metadata.truncation_reason = "token_limit_reached"
                    break
        
        return response
    
    def _format_value_concise(self, value: Any) -> Any:
        """Format value for concise output"""
        if isinstance(value, dict):
            # Keep only essential keys for objects
            essential_keys = {'id', 'name', 'status', 'result', 'value', 'success', 'error'}
            return {k: v for k, v in value.items() if k in essential_keys}
        elif isinstance(value, list) and len(value) > 5:
            # Truncate long lists
            return value[:5] + ["...truncated"]
        elif isinstance(value, str) and len(value) > 100:
            # Truncate long strings
            return value[:100] + "..."
        else:
            return value
    
    def estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate tokens for response"""
        content = json.dumps(response, separators=(',', ':'))
        return max(1, len(content) // 4)


class DetailedResponseProcessor(ResponseProcessor):
    """Processor for detailed responses with full context"""
    
    def process(self, elements: List[ResponseElement], metadata: ResponseMetadata) -> Dict[str, Any]:
        """Create detailed response with comprehensive information"""
        response = {}
        token_count = 0
        
        # Group elements by priority
        priority_groups = {
            ContextPriority.CRITICAL: [],
            ContextPriority.HIGH: [],
            ContextPriority.MEDIUM: [],
            ContextPriority.LOW: []
        }
        
        for element in elements:
            priority_groups[element.priority].append(element)
        
        # Add elements by priority group
        for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, ContextPriority.MEDIUM, ContextPriority.LOW]:
            for element in priority_groups[priority]:
                if token_count + element.token_estimate <= metadata.token_limit:
                    response[element.key] = self._format_value_detailed(element)
                    token_count += element.token_estimate
                else:
                    metadata.truncated = True
                    metadata.truncation_reason = f"token_limit_reached_at_{priority.value}_priority"
                    break
            
            if metadata.truncated:
                break
        
        # Add metadata section
        if token_count + 50 <= metadata.token_limit:  # Reserve 50 tokens for metadata
            response['_metadata'] = {
                'tool': metadata.tool_name,
                'execution_time': round(metadata.execution_time, 3),
                'format': metadata.requested_format.value,
                'truncated': metadata.truncated
            }
            
            if metadata.truncated:
                response['_metadata']['truncation_reason'] = metadata.truncation_reason
        
        return response
    
    def _format_value_detailed(self, element: ResponseElement) -> Any:
        """Format value for detailed output with context"""
        value = element.value
        
        # Add semantic identifiers where available
        if element.semantic_id and isinstance(value, dict):
            value = dict(value)  # Copy to avoid modifying original
            value['_semantic_id'] = element.semantic_id
        
        # Add descriptions for complex objects
        if element.description and isinstance(value, dict):
            value = dict(value)
            value['_description'] = element.description
        
        return value
    
    def estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate tokens for detailed response"""
        content = json.dumps(response, indent=2)
        return max(1, len(content) // 4)


class StructuredResponseProcessor(ResponseProcessor):
    """Processor for structured machine-readable responses"""
    
    def process(self, elements: List[ResponseElement], metadata: ResponseMetadata) -> Dict[str, Any]:
        """Create structured response optimized for machine consumption"""
        response = {
            'data': {},
            'schema': {},
            'metadata': {
                'tool': metadata.tool_name,
                'timestamp': time.time(),
                'format': 'structured',
                'version': '1.0'
            }
        }
        
        token_count = 100  # Base overhead for structure
        
        for element in sorted(elements, key=lambda x: x.priority.value):
            if token_count + element.token_estimate <= metadata.token_limit:
                response['data'][element.key] = element.value
                response['schema'][element.key] = {
                    'type': self._infer_type(element.value),
                    'priority': element.priority.value
                }
                
                if element.description:
                    response['schema'][element.key]['description'] = element.description
                
                if element.semantic_id:
                    response['schema'][element.key]['semantic_id'] = element.semantic_id
                
                token_count += element.token_estimate
            else:
                metadata.truncated = True
                break
        
        if metadata.truncated:
            response['metadata']['truncated'] = True
            response['metadata']['truncation_reason'] = metadata.truncation_reason
        
        return response
    
    def _infer_type(self, value: Any) -> str:
        """Infer JSON schema type from value"""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'number'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        else:
            return 'unknown'
    
    def estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate tokens for structured response"""
        content = json.dumps(response, separators=(',', ':'))
        return max(1, len(content) // 4)


class NaturalResponseProcessor(ResponseProcessor):
    """Processor for natural language responses"""
    
    def process(self, elements: List[ResponseElement], metadata: ResponseMetadata) -> Dict[str, Any]:
        """Create natural language narrative response"""
        # Sort elements by priority
        sorted_elements = sorted(elements, key=lambda x: (
            0 if x.priority == ContextPriority.CRITICAL else
            1 if x.priority == ContextPriority.HIGH else
            2 if x.priority == ContextPriority.MEDIUM else 3
        ))
        
        narrative_parts = []
        token_count = 0
        
        # Build narrative sections
        for element in sorted_elements:
            narrative = self._element_to_narrative(element)
            estimated_tokens = len(narrative) // 4
            
            if token_count + estimated_tokens <= metadata.token_limit:
                narrative_parts.append(narrative)
                token_count += estimated_tokens
            else:
                metadata.truncated = True
                metadata.truncation_reason = "narrative_length_limit"
                break
        
        # Combine into coherent response
        full_narrative = self._combine_narratives(narrative_parts, metadata)
        
        return {
            'narrative': full_narrative,
            'format': 'natural',
            'truncated': metadata.truncated
        }
    
    def _element_to_narrative(self, element: ResponseElement) -> str:
        """Convert response element to narrative text"""
        if element.description:
            base = element.description
        else:
            base = f"The {element.key.replace('_', ' ')}"
        
        # Format value appropriately
        if isinstance(element.value, bool):
            value_text = "is confirmed" if element.value else "is not confirmed"
        elif isinstance(element.value, (int, float)):
            value_text = f"is {element.value}"
        elif isinstance(element.value, str):
            value_text = f"is '{element.value}'"
        elif isinstance(element.value, list):
            if len(element.value) <= 3:
                value_text = f"includes {', '.join(map(str, element.value))}"
            else:
                value_text = f"includes {len(element.value)} items: {', '.join(map(str, element.value[:2]))} and others"
        elif isinstance(element.value, dict):
            key_count = len(element.value)
            value_text = f"contains {key_count} properties"
        else:
            value_text = f"has value {element.value}"
        
        return f"{base} {value_text}."
    
    def _combine_narratives(self, parts: List[str], metadata: ResponseMetadata) -> str:
        """Combine narrative parts into coherent response"""
        if not parts:
            return "No information available."
        
        # Add introduction
        intro = f"Based on the {metadata.tool_name} operation:"
        
        # Combine parts
        combined = intro + " " + " ".join(parts)
        
        # Add truncation notice if needed
        if metadata.truncated:
            combined += " (Note: Response was truncated due to length limits.)"
        
        return combined
    
    def estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate tokens for natural response"""
        narrative = response.get('narrative', '')
        return max(1, len(narrative) // 4)


class EnhancedResponseSystem:
    """
    Enhanced response system implementing Anthropic's optimization principles.
    
    Features:
    - Context-aware response prioritization
    - Multiple response formats with token efficiency
    - Semantic identifier resolution
    - Intelligent truncation with guidance
    - Response quality metrics
    """
    
    def __init__(self):
        self.processors = {
            ResponseFormat.CONCISE: ConciseResponseProcessor(),
            ResponseFormat.DETAILED: DetailedResponseProcessor(), 
            ResponseFormat.STRUCTURED: StructuredResponseProcessor(),
            ResponseFormat.NATURAL: NaturalResponseProcessor()
        }
        
        # Response optimization settings
        self.default_token_limits = {
            ResponseFormat.CONCISE: 500,
            ResponseFormat.DETAILED: 2000,
            ResponseFormat.STRUCTURED: 1500,
            ResponseFormat.NATURAL: 1000
        }
        
        self.semantic_id_cache = {}  # Cache for UUID -> semantic ID mapping
        
        logger.info("🔧 Enhanced Response System initialized")
    
    def create_response(
        self,
        tool_name: str,
        raw_data: Dict[str, Any],
        response_format: ResponseFormat = ResponseFormat.DETAILED,
        token_limit: Optional[int] = None,
        context_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create optimized response from raw tool data.
        
        Args:
            tool_name: Name of the tool that generated the data
            raw_data: Raw response data from tool execution
            response_format: Desired response format
            token_limit: Maximum tokens for response (None for default)
            context_hints: Hints about what information is most relevant
            
        Returns:
            Optimized response dictionary
        """
        start_time = time.time()
        
        # Determine token limit
        if token_limit is None:
            token_limit = self.default_token_limits[response_format]
        
        # Create metadata
        metadata = ResponseMetadata(
            tool_name=tool_name,
            execution_time=time.time() - start_time,  # Will be updated
            token_limit=token_limit,
            requested_format=response_format,
            context_hints=context_hints or []
        )
        
        # Convert raw data to response elements
        elements = self._create_response_elements(raw_data, context_hints)
        
        # Resolve semantic identifiers
        elements = self._resolve_semantic_identifiers(elements)
        
        # Process with appropriate processor
        processor = self.processors[response_format]
        response = processor.process(elements, metadata)
        
        # Update execution time
        metadata.execution_time = time.time() - start_time
        
        # Add system metadata if not already present
        if '_system' not in response:
            response['_system'] = {
                'processing_time': round(metadata.execution_time, 3),
                'format': response_format.value,
                'token_estimate': processor.estimate_tokens(response),
                'optimized': True
            }
        
        logger.debug(f"Created {response_format.value} response for {tool_name} in {metadata.execution_time:.3f}s")
        
        return response
    
    def _create_response_elements(
        self,
        raw_data: Dict[str, Any],
        context_hints: Optional[List[str]] = None
    ) -> List[ResponseElement]:
        """Convert raw data to prioritized response elements"""
        elements = []
        context_hints = context_hints or []
        
        for key, value in raw_data.items():
            # Determine priority based on key and context
            priority = self._determine_priority(key, value, context_hints)
            
            # Create element
            element = ResponseElement(
                key=key,
                value=value,
                priority=priority,
                description=self._generate_description(key, value),
                semantic_id=self._extract_semantic_id(key, value)
            )
            
            elements.append(element)
        
        return elements
    
    def _determine_priority(
        self,
        key: str,
        value: Any,
        context_hints: List[str]
    ) -> ContextPriority:
        """Determine priority level for a response element"""
        
        # Critical keys that should always be included
        critical_keys = {
            'success', 'error', 'result', 'status', 'output',
            'validation_result', 'physics_compliance', 'reasoning_result'
        }
        
        if key in critical_keys:
            return ContextPriority.CRITICAL
        
        # High priority for context-relevant information
        if context_hints:
            key_lower = key.lower()
            for hint in context_hints:
                if hint.lower() in key_lower or key_lower in hint.lower():
                    return ContextPriority.HIGH
        
        # High priority for meaningful identifiers
        high_priority_patterns = {
            'name', 'id', 'title', 'summary', 'confidence', 'accuracy',
            'performance', 'metrics', 'recommendation', 'analysis'
        }
        
        key_lower = key.lower()
        for pattern in high_priority_patterns:
            if pattern in key_lower:
                return ContextPriority.HIGH
        
        # Medium priority for data and metadata
        medium_priority_patterns = {
            'data', 'metadata', 'parameters', 'config', 'settings',
            'timestamp', 'duration', 'count', 'size'
        }
        
        for pattern in medium_priority_patterns:
            if pattern in key_lower:
                return ContextPriority.MEDIUM
        
        # Everything else is low priority
        return ContextPriority.LOW
    
    def _generate_description(self, key: str, value: Any) -> Optional[str]:
        """Generate human-readable description for response element"""
        # Convert snake_case to human readable
        readable_key = key.replace('_', ' ').title()
        
        if isinstance(value, bool):
            return f"{readable_key} status indicator"
        elif isinstance(value, (int, float)) and 'count' in key.lower():
            return f"Number of {readable_key.lower()}"
        elif isinstance(value, (int, float)) and any(word in key.lower() for word in ['time', 'duration', 'latency']):
            return f"{readable_key} measurement"
        elif isinstance(value, list):
            return f"Collection of {readable_key.lower()}"
        elif isinstance(value, dict):
            return f"{readable_key} information"
        else:
            return f"{readable_key} value"
    
    def _extract_semantic_id(self, key: str, value: Any) -> Optional[str]:
        """Extract or generate semantic identifier for cryptic IDs"""
        if not isinstance(value, str):
            return None
        
        # Check if value looks like a UUID or cryptic ID
        if len(value) == 36 and value.count('-') == 4:  # UUID format
            # Try to resolve from cache or generate meaningful name
            if value in self.semantic_id_cache:
                return self.semantic_id_cache[value]
            else:
                # Generate semantic ID based on context
                semantic = self._generate_semantic_id(key, value)
                self.semantic_id_cache[value] = semantic
                return semantic
        
        # Check for other cryptic patterns
        if len(value) > 20 and value.isalnum():  # Long alphanumeric
            return self._generate_semantic_id(key, value)
        
        return None
    
    def _generate_semantic_id(self, key: str, value: str) -> str:
        """Generate semantic identifier for cryptic ID"""
        # Create meaningful name based on key and value hash
        hash_short = hashlib.md5(value.encode()).hexdigest()[:6]
        
        if 'agent' in key.lower():
            return f"agent_{hash_short}"
        elif 'task' in key.lower():
            return f"task_{hash_short}"
        elif 'session' in key.lower():
            return f"session_{hash_short}"
        elif 'request' in key.lower():
            return f"request_{hash_short}"
        else:
            return f"{key}_{hash_short}"
    
    def _resolve_semantic_identifiers(self, elements: List[ResponseElement]) -> List[ResponseElement]:
        """Resolve cryptic identifiers to semantic names where possible"""
        # This is a placeholder for more sophisticated ID resolution
        # In practice, you might integrate with a registry or database
        
        for element in elements:
            if element.semantic_id:
                # Could enhance value with semantic information
                if isinstance(element.value, str) and len(element.value) > 20:
                    # Keep original value but note semantic ID is available
                    pass
        
        return elements
    
    def add_semantic_mapping(self, cryptic_id: str, semantic_id: str):
        """Add mapping from cryptic ID to semantic identifier"""
        self.semantic_id_cache[cryptic_id] = semantic_id
    
    def get_format_recommendations(self, tool_name: str, data_size: int) -> ResponseFormat:
        """Recommend optimal response format based on context"""
        if data_size < 100:  # Small responses
            return ResponseFormat.CONCISE
        elif data_size < 1000:  # Medium responses
            return ResponseFormat.DETAILED
        elif 'structured' in tool_name.lower() or 'api' in tool_name.lower():
            return ResponseFormat.STRUCTURED
        else:
            return ResponseFormat.NATURAL
    
    def analyze_response_quality(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality metrics for generated response"""
        analysis = {
            'token_efficiency': 0.0,
            'information_density': 0.0,
            'semantic_clarity': 0.0,
            'completeness': 0.0
        }
        
        # Calculate token efficiency (information per token)
        token_count = len(json.dumps(response)) // 4
        info_elements = len([k for k in response.keys() if not k.startswith('_')])
        
        if token_count > 0:
            analysis['token_efficiency'] = info_elements / token_count
        
        # Information density (non-metadata proportion)
        total_keys = len(response.keys())
        metadata_keys = len([k for k in response.keys() if k.startswith('_')])
        
        if total_keys > 0:
            analysis['information_density'] = (total_keys - metadata_keys) / total_keys
        
        # Semantic clarity (presence of human-readable identifiers)
        semantic_elements = 0
        for value in response.values():
            if isinstance(value, dict) and '_semantic_id' in value:
                semantic_elements += 1
        
        if info_elements > 0:
            analysis['semantic_clarity'] = semantic_elements / info_elements
        
        # Completeness (not truncated)
        analysis['completeness'] = 0.0 if response.get('_system', {}).get('truncated') else 1.0
        
        return analysis


# Example usage and testing
def main():
    """Example usage of enhanced response system"""
    system = EnhancedResponseSystem()
    
    # Example raw tool data
    raw_data = {
        'success': True,
        'result': {'value': 42, 'confidence': 0.95},
        'agent_id': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        'execution_time': 1.23,
        'metadata': {'version': '1.0', 'model': 'kan_v2'},
        'debug_info': {'internal_state': 'active', 'memory_usage': '45MB'},
        'recommendations': ['optimize_parameters', 'increase_sample_size']
    }
    
    # Test different formats
    for format_type in ResponseFormat:
        print(f"\n=== {format_type.value.upper()} FORMAT ===")
        
        response = system.create_response(
            tool_name='kan_reason',
            raw_data=raw_data,
            response_format=format_type,
            context_hints=['confidence', 'result']
        )
        
        print(json.dumps(response, indent=2))
        
        # Analyze quality
        quality = system.analyze_response_quality(response)
        print(f"Quality metrics: {quality}")


if __name__ == "__main__":
    main()
