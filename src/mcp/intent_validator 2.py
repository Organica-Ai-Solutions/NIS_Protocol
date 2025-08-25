"""
Intent Validator for MCP UI Integration

Validates and handles UI intents from mcp-ui components.
Provides security layer and intent routing for user interactions.
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from urllib.parse import urlparse


class IntentValidator:
    """
    Validates and handles UI intents from mcp-ui components.
    
    Provides security validation, schema checking, and intent routing
    to ensure safe handling of user interactions from sandboxed UI.
    """
    
    def __init__(self):
        self.intent_handlers: Dict[str, Callable] = {}
        self.intent_schemas = self._get_intent_schemas()
        self.security_rules = self._get_security_rules()
        
    def register_handler(self, intent_type: str, handler: Callable):
        """Register a handler for a specific intent type."""
        self.intent_handlers[intent_type] = handler
        logging.debug(f"Registered handler for intent type: {intent_type}")
        
    def validate_intent(self, intent_type: str, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate an intent against security rules and schema.
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Check if intent type is supported
        if intent_type not in self.intent_schemas:
            errors.append(f"Unknown intent type: {intent_type}")
            return False, errors
            
        # Security validation
        security_errors = self._validate_security(intent_type, payload)
        errors.extend(security_errors)
        
        # Schema validation
        schema_errors = self._validate_schema(intent_type, payload)
        errors.extend(schema_errors)
        
        return len(errors) == 0, errors
        
    async def handle_intent(self, intent_type: str, payload: Dict[str, Any], 
                          message_id: str = None) -> Dict[str, Any]:
        """
        Handle a validated intent.
        
        Args:
            intent_type: Type of intent (tool, intent, prompt, notify, link)
            payload: Intent payload data
            message_id: Optional message ID for async responses
            
        Returns:
            Handler response
        """
        if intent_type not in self.intent_handlers:
            return {"error": f"No handler registered for intent type: {intent_type}"}
            
        try:
            handler = self.intent_handlers[intent_type]
            return await handler(payload, message_id)
        except Exception as e:
            logging.error(f"Error handling intent {intent_type}: {str(e)}")
            return {"error": f"Intent handling failed: {str(e)}"}
            
    def _get_intent_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all supported intent types."""
        return {
            "tool": {
                "type": "object",
                "properties": {
                    "toolName": {"type": "string", "pattern": r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$"},
                    "params": {"type": "object"}
                },
                "required": ["toolName"]
            },
            "intent": {
                "type": "object", 
                "properties": {
                    "intent": {"type": "string", "enum": ["refresh", "navigate", "close", "minimize", "maximize"]},
                    "params": {"type": "object"}
                },
                "required": ["intent"]
            },
            "prompt": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "maxLength": 10000}
                },
                "required": ["prompt"]
            },
            "notify": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "maxLength": 1000},
                    "level": {"type": "string", "enum": ["info", "warning", "error", "success"], "default": "info"}
                },
                "required": ["message"]
            },
            "link": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
        
    def _get_security_rules(self) -> Dict[str, Any]:
        """Get security rules for intent validation."""
        return {
            "tool": {
                "allowed_tools": [
                    # Dataset tools
                    "dataset.search", "dataset.preview", "dataset.analyze", "dataset.list",
                    # Pipeline tools
                    "pipeline.run", "pipeline.status", "pipeline.configure", "pipeline.cancel", 
                    "pipeline.artifacts", "pipeline.list",
                    # Research tools
                    "research.plan", "research.search", "research.synthesize", "research.analyze",
                    # Audit tools
                    "audit.view", "audit.analyze", "audit.compliance", "audit.risk", "audit.report",
                    # Code tools
                    "code.edit", "code.review", "code.analyze", "code.generate", "code.refactor"
                ],
                "parameter_limits": {
                    "max_string_length": 5000,
                    "max_array_items": 100,
                    "max_object_depth": 5
                }
            },
            "intent": {
                "allowed_intents": ["refresh", "navigate", "close", "minimize", "maximize"],
                "navigation_whitelist": [
                    r"^/dashboard.*",
                    r"^/data.*", 
                    r"^/pipeline.*",
                    r"^/research.*",
                    r"^/audit.*",
                    r"^/code.*"
                ]
            },
            "prompt": {
                "forbidden_patterns": [
                    r"<script.*?>.*?</script>",  # No script injection
                    r"javascript:",             # No javascript URLs
                    r"data:.*?base64",         # No base64 data URLs
                    r"file://",                # No file URLs
                    r"\\\\\\\\",               # No UNC paths
                ],
                "max_length": 10000
            },
            "link": {
                "allowed_schemes": ["http", "https"],
                "forbidden_domains": ["localhost", "127.0.0.1", "0.0.0.0"],
                "path_whitelist": [
                    r"^/api/.*",
                    r"^/docs/.*",
                    r"^/dashboard/.*"
                ]
            }
        }
        
    def _validate_security(self, intent_type: str, payload: Dict[str, Any]) -> List[str]:
        """Validate intent against security rules."""
        errors = []
        rules = self.security_rules.get(intent_type, {})
        
        if intent_type == "tool":
            errors.extend(self._validate_tool_security(payload, rules))
        elif intent_type == "intent":
            errors.extend(self._validate_intent_security(payload, rules))
        elif intent_type == "prompt":
            errors.extend(self._validate_prompt_security(payload, rules))
        elif intent_type == "link":
            errors.extend(self._validate_link_security(payload, rules))
            
        return errors
        
    def _validate_tool_security(self, payload: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate tool intent security."""
        errors = []
        
        tool_name = payload.get("toolName", "")
        allowed_tools = rules.get("allowed_tools", [])
        
        if tool_name not in allowed_tools:
            errors.append(f"Tool not allowed: {tool_name}")
            
        # Validate parameters
        params = payload.get("params", {})
        param_errors = self._validate_parameters_security(params, rules.get("parameter_limits", {}))
        errors.extend(param_errors)
        
        return errors
        
    def _validate_intent_security(self, payload: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate generic intent security."""
        errors = []
        
        intent = payload.get("intent", "")
        allowed_intents = rules.get("allowed_intents", [])
        
        if intent not in allowed_intents:
            errors.append(f"Intent not allowed: {intent}")
            
        # Validate navigation URLs
        if intent == "navigate":
            url = payload.get("params", {}).get("url", "")
            if url:
                nav_errors = self._validate_navigation_url(url, rules.get("navigation_whitelist", []))
                errors.extend(nav_errors)
                
        return errors
        
    def _validate_prompt_security(self, payload: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate prompt security."""
        errors = []
        
        prompt = payload.get("prompt", "")
        
        # Check length
        max_length = rules.get("max_length", 10000)
        if len(prompt) > max_length:
            errors.append(f"Prompt too long: {len(prompt)} > {max_length}")
            
        # Check for forbidden patterns
        forbidden_patterns = rules.get("forbidden_patterns", [])
        for pattern in forbidden_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                errors.append(f"Prompt contains forbidden pattern: {pattern}")
                
        return errors
        
    def _validate_link_security(self, payload: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate link click security."""
        errors = []
        
        url = payload.get("url", "")
        if not url:
            return errors
            
        try:
            parsed = urlparse(url)
            
            # Check scheme
            allowed_schemes = rules.get("allowed_schemes", ["http", "https"])
            if parsed.scheme not in allowed_schemes:
                errors.append(f"URL scheme not allowed: {parsed.scheme}")
                
            # Check domain
            forbidden_domains = rules.get("forbidden_domains", [])
            if parsed.hostname and parsed.hostname.lower() in forbidden_domains:
                errors.append(f"Domain not allowed: {parsed.hostname}")
                
            # Check path
            path_whitelist = rules.get("path_whitelist", [])
            if path_whitelist:
                path_allowed = any(re.match(pattern, parsed.path) for pattern in path_whitelist)
                if not path_allowed:
                    errors.append(f"URL path not whitelisted: {parsed.path}")
                    
        except Exception as e:
            errors.append(f"Invalid URL format: {str(e)}")
            
        return errors
        
    def _validate_parameters_security(self, params: Dict[str, Any], limits: Dict[str, Any]) -> List[str]:
        """Validate parameter values for security."""
        errors = []
        
        max_string_length = limits.get("max_string_length", 5000)
        max_array_items = limits.get("max_array_items", 100)
        max_object_depth = limits.get("max_object_depth", 5)
        
        def check_value(value, depth=0):
            if depth > max_object_depth:
                errors.append(f"Object nesting too deep: {depth} > {max_object_depth}")
                return
                
            if isinstance(value, str):
                if len(value) > max_string_length:
                    errors.append(f"String too long: {len(value)} > {max_string_length}")
            elif isinstance(value, list):
                if len(value) > max_array_items:
                    errors.append(f"Array too large: {len(value)} > {max_array_items}")
                for item in value:
                    check_value(item, depth + 1)
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(k, depth + 1)
                    check_value(v, depth + 1)
                    
        check_value(params)
        return errors
        
    def _validate_navigation_url(self, url: str, whitelist: List[str]) -> List[str]:
        """Validate navigation URL against whitelist."""
        errors = []
        
        if not whitelist:
            return errors  # No restrictions if no whitelist
            
        # Check if URL matches any whitelist pattern
        url_allowed = any(re.match(pattern, url) for pattern in whitelist)
        if not url_allowed:
            errors.append(f"Navigation URL not whitelisted: {url}")
            
        return errors
        
    def _validate_schema(self, intent_type: str, payload: Dict[str, Any]) -> List[str]:
        """Validate intent payload against schema."""
        errors = []
        schema = self.intent_schemas.get(intent_type, {})
        
        # Basic schema validation
        required = schema.get("required", [])
        for field in required:
            if field not in payload:
                errors.append(f"Missing required field: {field}")
                
        properties = schema.get("properties", {})
        for field, value in payload.items():
            if field in properties:
                field_schema = properties[field]
                field_errors = self._validate_field_schema(field, value, field_schema)
                errors.extend(field_errors)
                
        return errors
        
    def _validate_field_schema(self, field_name: str, value: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate a single field against its schema."""
        errors = []
        
        expected_type = schema.get("type")
        
        # Type checking
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field_name}' must be string")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}' must be number")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field_name}' must be boolean")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field '{field_name}' must be array")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"Field '{field_name}' must be object")
            
        # String validations
        if isinstance(value, str):
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"Field '{field_name}' too long: {len(value)} > {schema['maxLength']}")
                
            if "pattern" in schema:
                pattern = schema["pattern"]
                if not re.match(pattern, value):
                    errors.append(f"Field '{field_name}' does not match pattern: {pattern}")
                    
            if "enum" in schema and value not in schema["enum"]:
                errors.append(f"Field '{field_name}' must be one of: {schema['enum']}")
                
        return errors
        
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intent types."""
        return list(self.intent_schemas.keys())
        
    def get_intent_schema(self, intent_type: str) -> Dict[str, Any]:
        """Get schema for a specific intent type."""
        return self.intent_schemas.get(intent_type, {})
        
    def get_security_info(self) -> Dict[str, Any]:
        """Get security configuration info."""
        return {
            "supported_intents": self.get_supported_intents(),
            "security_rules": {
                intent_type: {
                    "description": f"Security rules for {intent_type} intents",
                    "rules": list(rules.keys())
                }
                for intent_type, rules in self.security_rules.items()
            }
        }
