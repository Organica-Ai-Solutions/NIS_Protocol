"""
Enhanced Tool Schemas for NIS Protocol
Based on Anthropic's tool optimization research

Implements improved tool design patterns:
- Clear, consistent namespacing (nis_, physics_, kan_, laplace_)
- Unambiguous parameter names
- Comprehensive tool descriptions
- Response format controls
- Token efficiency features

Reference: https://www.anthropic.com/engineering/writing-tools-for-agents
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import json


class ResponseFormat(Enum):
    """Response format options for context efficiency"""
    CONCISE = "concise"      # Essential information only
    DETAILED = "detailed"    # Full information with metadata
    STRUCTURED = "structured" # JSON/XML structured format
    NATURAL = "natural"      # Human-readable text format


class ToolCategory(Enum):
    """Tool categories for clear organization"""
    NIS_CORE = "nis_core"           # Core NIS Protocol operations
    LAPLACE = "laplace"             # Signal processing (Laplace transforms)
    KAN = "kan"                     # Reasoning (KAN networks)  
    PHYSICS = "physics"             # Physics validation (PINN)
    DATASET = "dataset"             # Data management
    PIPELINE = "pipeline"           # Data processing workflows
    RESEARCH = "research"           # Research and analysis
    AUDIT = "audit"                 # System auditing
    CODE = "code"                   # Code analysis and generation


@dataclass
class ToolParameter:
    """Enhanced parameter definition with clear semantics"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    examples: List[Any] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.constraints is None:
            self.constraints = {}


@dataclass
class ToolDefinition:
    """Enhanced tool definition with optimization features"""
    name: str
    category: ToolCategory
    description: str
    parameters: List[ToolParameter]
    
    # Anthropic optimization features
    response_formats: List[ResponseFormat] = None
    max_response_tokens: int = 1000
    supports_pagination: bool = False
    supports_filtering: bool = False
    consolidates_operations: List[str] = None  # Operations this tool combines
    
    # Usage guidance
    usage_examples: List[Dict[str, Any]] = None
    common_workflows: List[str] = None
    error_patterns: List[str] = None
    
    def __post_init__(self):
        if self.response_formats is None:
            self.response_formats = [ResponseFormat.DETAILED]
        if self.consolidates_operations is None:
            self.consolidates_operations = []
        if self.usage_examples is None:
            self.usage_examples = []
        if self.common_workflows is None:
            self.common_workflows = []
        if self.error_patterns is None:
            self.error_patterns = []


class EnhancedToolSchemas:
    """
    Enhanced tool schema system implementing Anthropic's best practices.
    
    Key improvements:
    - Consistent namespacing with clear boundaries
    - Consolidated operations to reduce tool proliferation
    - Response format controls for token efficiency
    - Comprehensive parameter descriptions with examples
    - Built-in pagination and filtering support
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._initialize_enhanced_tools()
    
    def _initialize_enhanced_tools(self):
        """Initialize enhanced tool definitions"""
        
        # 🧠 NIS CORE TOOLS - Always-active system operations
        self.tools.update({
            "nis_status": ToolDefinition(
                name="nis_status",
                category=ToolCategory.NIS_CORE,
                description="Get comprehensive system status including active agents, performance metrics, and health indicators",
                parameters=[
                    ToolParameter(
                        name="detail_level",
                        type="string",
                        description="Level of detail to include: 'summary', 'detailed', or 'full'",
                        required=False,
                        default="detailed",
                        examples=["summary", "detailed", "full"]
                    ),
                    ToolParameter(
                        name="include_agents",
                        type="boolean", 
                        description="Whether to include individual agent status",
                        required=False,
                        default=True
                    ),
                    ToolParameter(
                        name="response_format",
                        type="string",
                        description="Response format: 'concise', 'detailed', 'structured', or 'natural'",
                        required=False,
                        default="detailed",
                        examples=["concise", "detailed", "structured", "natural"]
                    )
                ],
                response_formats=[ResponseFormat.CONCISE, ResponseFormat.DETAILED, ResponseFormat.STRUCTURED],
                max_response_tokens=500,
                usage_examples=[
                    {
                        "description": "Get quick system overview",
                        "parameters": {"detail_level": "summary", "response_format": "concise"}
                    },
                    {
                        "description": "Full system diagnostics",
                        "parameters": {"detail_level": "full", "include_agents": True}
                    }
                ],
                common_workflows=["system_monitoring", "health_check", "debugging"]
            ),
            
            "nis_configure": ToolDefinition(
                name="nis_configure",
                category=ToolCategory.NIS_CORE,
                description="Configure NIS Protocol system settings, agent parameters, and operational modes",
                parameters=[
                    ToolParameter(
                        name="config_section",
                        type="string",
                        description="Configuration section to modify: 'agents', 'performance', 'protocols', or 'logging'",
                        required=True,
                        examples=["agents", "performance", "protocols", "logging"]
                    ),
                    ToolParameter(
                        name="settings",
                        type="object",
                        description="Configuration settings as key-value pairs",
                        required=True,
                        examples=[
                            {"max_concurrent_agents": 5, "timeout_seconds": 30},
                            {"log_level": "INFO", "enable_detailed_tracing": True}
                        ]
                    ),
                    ToolParameter(
                        name="validate_before_apply",
                        type="boolean",
                        description="Validate configuration before applying changes",
                        required=False,
                        default=True
                    )
                ],
                consolidates_operations=["set_agent_config", "update_performance_settings", "configure_logging"],
                usage_examples=[
                    {
                        "description": "Update agent timeout settings",
                        "parameters": {
                            "config_section": "agents",
                            "settings": {"timeout_seconds": 60, "max_retries": 3}
                        }
                    }
                ]
            )
        })
        
        # 🌊 LAPLACE TOOLS - Signal processing and transformation
        self.tools.update({
            "laplace_transform": ToolDefinition(
                name="laplace_transform",
                category=ToolCategory.LAPLACE,
                description="Apply Laplace transform to input signals for frequency domain analysis and preprocessing",
                parameters=[
                    ToolParameter(
                        name="signal_data",
                        type="array",
                        description="Input signal data as array of numerical values",
                        required=True,
                        examples=[[1.0, 2.5, 3.2, 1.8, 0.5], [0, 1, 4, 9, 16, 25]]
                    ),
                    ToolParameter(
                        name="sampling_rate",
                        type="number",
                        description="Sampling rate in Hz for proper frequency domain mapping",
                        required=True,
                        examples=[1000, 44100, 8000],
                        constraints={"min": 1, "max": 1000000}
                    ),
                    ToolParameter(
                        name="transform_type",
                        type="string", 
                        description="Type of Laplace transform: 'standard', 'inverse', or 'bilateral'",
                        required=False,
                        default="standard",
                        examples=["standard", "inverse", "bilateral"]
                    ),
                    ToolParameter(
                        name="response_format",
                        type="string",
                        description="Output format: 'concise' (values only) or 'detailed' (with metadata)",
                        required=False,
                        default="detailed"
                    )
                ],
                response_formats=[ResponseFormat.CONCISE, ResponseFormat.DETAILED],
                max_response_tokens=2000,
                supports_pagination=True,
                usage_examples=[
                    {
                        "description": "Transform audio signal for analysis",
                        "parameters": {
                            "signal_data": [0.1, 0.5, 0.8, 0.3, -0.2],
                            "sampling_rate": 44100,
                            "transform_type": "standard"
                        }
                    }
                ],
                common_workflows=["signal_preprocessing", "frequency_analysis", "noise_filtering"]
            ),
            
            "laplace_analyze_stability": ToolDefinition(
                name="laplace_analyze_stability",
                category=ToolCategory.LAPLACE,
                description="Analyze system stability using Laplace domain analysis of poles and zeros",
                parameters=[
                    ToolParameter(
                        name="transfer_function",
                        type="object",
                        description="Transfer function coefficients with 'numerator' and 'denominator' arrays",
                        required=True,
                        examples=[
                            {"numerator": [1, 2], "denominator": [1, 3, 2]},
                            {"numerator": [5], "denominator": [1, 0, -4]}
                        ]
                    ),
                    ToolParameter(
                        name="stability_criteria",
                        type="array",
                        description="Stability criteria to check: 'routh_hurwitz', 'nyquist', 'bode'",
                        required=False,
                        default=["routh_hurwitz"],
                        examples=[["routh_hurwitz"], ["nyquist", "bode"], ["routh_hurwitz", "nyquist"]]
                    )
                ],
                consolidates_operations=["check_poles", "analyze_zeros", "stability_margins"],
                common_workflows=["control_system_design", "stability_verification", "system_analysis"]
            )
        })
        
        # 🧮 KAN TOOLS - Kolmogorov-Arnold Network reasoning
        self.tools.update({
            "kan_reason": ToolDefinition(
                name="kan_reason",
                category=ToolCategory.KAN,
                description="Perform symbolic reasoning using Kolmogorov-Arnold Networks with interpretable function approximation",
                parameters=[
                    ToolParameter(
                        name="input_data",
                        type="object",
                        description="Input data for reasoning with 'variables' dict and optional 'constraints' array",
                        required=True,
                        examples=[
                            {"variables": {"x": 5, "y": 3}, "constraints": ["x > 0", "y < 10"]},
                            {"variables": {"temperature": 25.5, "pressure": 101.3}}
                        ]
                    ),
                    ToolParameter(
                        name="reasoning_type",
                        type="string",
                        description="Type of reasoning: 'symbolic', 'numerical', 'hybrid', or 'interpretable'",
                        required=False,
                        default="hybrid",
                        examples=["symbolic", "numerical", "hybrid", "interpretable"]
                    ),
                    ToolParameter(
                        name="extract_functions",
                        type="boolean",
                        description="Whether to extract interpretable symbolic functions from the network",
                        required=False,
                        default=True
                    ),
                    ToolParameter(
                        name="confidence_threshold",
                        type="number",
                        description="Minimum confidence threshold for reasoning results (0.0 to 1.0)",
                        required=False,
                        default=0.7,
                        constraints={"min": 0.0, "max": 1.0}
                    )
                ],
                response_formats=[ResponseFormat.CONCISE, ResponseFormat.DETAILED, ResponseFormat.STRUCTURED],
                consolidates_operations=["symbolic_analysis", "function_approximation", "interpretability_extraction"],
                usage_examples=[
                    {
                        "description": "Analyze mathematical relationship",
                        "parameters": {
                            "input_data": {"variables": {"x": [1, 2, 3, 4], "y": [1, 4, 9, 16]}},
                            "reasoning_type": "symbolic",
                            "extract_functions": True
                        }
                    }
                ],
                common_workflows=["pattern_recognition", "symbolic_regression", "interpretable_ai"]
            ),
            
            "kan_optimize": ToolDefinition(
                name="kan_optimize",
                category=ToolCategory.KAN,
                description="Optimize KAN network parameters and spline functions for improved reasoning performance",
                parameters=[
                    ToolParameter(
                        name="optimization_target",
                        type="string",
                        description="What to optimize: 'accuracy', 'interpretability', 'speed', or 'memory'",
                        required=True,
                        examples=["accuracy", "interpretability", "speed", "memory"]
                    ),
                    ToolParameter(
                        name="training_data",
                        type="object",
                        description="Training data with 'inputs' and 'outputs' arrays",
                        required=False,
                        examples=[
                            {"inputs": [[1, 2], [3, 4]], "outputs": [3, 7]}
                        ]
                    ),
                    ToolParameter(
                        name="max_iterations",
                        type="integer",
                        description="Maximum optimization iterations",
                        required=False,
                        default=100,
                        constraints={"min": 1, "max": 10000}
                    )
                ],
                supports_pagination=True,
                common_workflows=["model_tuning", "performance_optimization", "interpretability_enhancement"]
            )
        })
        
        # ⚡ PHYSICS TOOLS - Physics-Informed Neural Networks
        self.tools.update({
            "physics_validate": ToolDefinition(
                name="physics_validate",
                category=ToolCategory.PHYSICS,
                description="Validate outputs against fundamental physics laws using Physics-Informed Neural Networks",
                parameters=[
                    ToolParameter(
                        name="physical_system",
                        type="object",
                        description="Physical system description with 'type', 'parameters', and 'state' information",
                        required=True,
                        examples=[
                            {
                                "type": "mechanical",
                                "parameters": {"mass": 1.0, "velocity": [10, 0, 0]},
                                "state": {"position": [0, 0, 0], "time": 0}
                            },
                            {
                                "type": "thermodynamic",
                                "parameters": {"temperature": 300, "pressure": 101325},
                                "state": {"volume": 0.001, "entropy": 1000}
                            }
                        ]
                    ),
                    ToolParameter(
                        name="conservation_laws",
                        type="array",
                        description="Conservation laws to validate: 'energy', 'momentum', 'mass', 'charge'",
                        required=False,
                        default=["energy", "momentum"],
                        examples=[["energy"], ["momentum", "mass"], ["energy", "momentum", "charge"]]
                    ),
                    ToolParameter(
                        name="tolerance",
                        type="number",
                        description="Numerical tolerance for physics validation (relative error)",
                        required=False,
                        default=0.01,
                        constraints={"min": 1e-10, "max": 0.1}
                    ),
                    ToolParameter(
                        name="auto_correct",
                        type="boolean",
                        description="Whether to automatically correct physics violations when possible",
                        required=False,
                        default=True
                    )
                ],
                consolidates_operations=["check_conservation", "validate_constraints", "auto_correction"],
                usage_examples=[
                    {
                        "description": "Validate projectile motion",
                        "parameters": {
                            "physical_system": {
                                "type": "mechanical",
                                "parameters": {"mass": 0.1, "initial_velocity": [20, 15, 0]},
                                "state": {"position": [100, 50, 0], "time": 5}
                            },
                            "conservation_laws": ["energy", "momentum"]
                        }
                    }
                ],
                common_workflows=["simulation_validation", "physics_checking", "auto_correction"]
            ),
            
            "physics_simulate": ToolDefinition(
                name="physics_simulate",
                category=ToolCategory.PHYSICS,
                description="Run physics simulations with PINN constraints for accurate predictions",
                parameters=[
                    ToolParameter(
                        name="simulation_config",
                        type="object",
                        description="Simulation configuration with 'system_type', 'initial_conditions', and 'time_range'",
                        required=True,
                        examples=[
                            {
                                "system_type": "pendulum",
                                "initial_conditions": {"angle": 0.5, "angular_velocity": 0},
                                "time_range": {"start": 0, "end": 10, "steps": 100}
                            }
                        ]
                    ),
                    ToolParameter(
                        name="output_variables",
                        type="array",
                        description="Variables to output: 'position', 'velocity', 'acceleration', 'energy'",
                        required=False,
                        default=["position", "velocity"],
                        examples=[["position"], ["position", "velocity", "energy"]]
                    ),
                    ToolParameter(
                        name="response_format",
                        type="string",
                        description="Output format: 'concise' (final state), 'detailed' (full trajectory)",
                        required=False,
                        default="detailed"
                    )
                ],
                response_formats=[ResponseFormat.CONCISE, ResponseFormat.DETAILED],
                supports_pagination=True,
                max_response_tokens=5000,
                common_workflows=["system_modeling", "prediction", "design_validation"]
            )
        })
        
        # 📊 CONSOLIDATED DATASET TOOLS - Streamlined data operations
        self.tools.update({
            "dataset_search_and_preview": ToolDefinition(
                name="dataset_search_and_preview",
                category=ToolCategory.DATASET,
                description="Search for datasets and get preview with structure analysis in one operation",
                parameters=[
                    ToolParameter(
                        name="search_query",
                        type="string",
                        description="Search query with keywords, tags, or specific criteria",
                        required=True,
                        examples=["climate change temperature data", "financial stock prices 2023", "medical imaging brain MRI"]
                    ),
                    ToolParameter(
                        name="filters",
                        type="object",
                        description="Search filters for format, size, date range, and quality metrics",
                        required=False,
                        examples=[
                            {"format": "CSV", "min_size": 1000, "max_size": 1000000},
                            {"created_after": "2023-01-01", "quality_score": 0.8}
                        ]
                    ),
                    ToolParameter(
                        name="preview_samples",
                        type="integer",
                        description="Number of sample records to include in preview",
                        required=False,
                        default=5,
                        constraints={"min": 1, "max": 100}
                    ),
                    ToolParameter(
                        name="response_format",
                        type="string",
                        description="Response format: 'concise' (summary only) or 'detailed' (with samples)",
                        required=False,
                        default="detailed"
                    )
                ],
                consolidates_operations=["search_datasets", "preview_structure", "analyze_quality"],
                supports_filtering=True,
                supports_pagination=True,
                usage_examples=[
                    {
                        "description": "Find and preview climate datasets",
                        "parameters": {
                            "search_query": "climate temperature precipitation",
                            "filters": {"format": "CSV", "min_size": 5000},
                            "preview_samples": 3
                        }
                    }
                ],
                common_workflows=["data_discovery", "dataset_evaluation", "data_preparation"]
            )
        })
        
        # 🔄 CONSOLIDATED PIPELINE TOOLS - Streamlined processing workflows  
        self.tools.update({
            "pipeline_execute_workflow": ToolDefinition(
                name="pipeline_execute_workflow",
                category=ToolCategory.PIPELINE,
                description="Execute complete data processing workflow from ingestion to output with quality validation",
                parameters=[
                    ToolParameter(
                        name="workflow_config",
                        type="object",
                        description="Workflow configuration with 'steps', 'input_source', and 'output_target'",
                        required=True,
                        examples=[
                            {
                                "steps": ["ingest", "clean", "transform", "validate"],
                                "input_source": {"type": "file", "path": "/data/raw.csv"},
                                "output_target": {"type": "database", "table": "processed_data"}
                            }
                        ]
                    ),
                    ToolParameter(
                        name="processing_options",
                        type="object",
                        description="Processing options including parallelization, error handling, and quality thresholds",
                        required=False,
                        examples=[
                            {"parallel_workers": 4, "error_tolerance": 0.05, "quality_threshold": 0.9}
                        ]
                    ),
                    ToolParameter(
                        name="monitor_progress",
                        type="boolean",
                        description="Whether to provide progress updates during processing",
                        required=False,
                        default=True
                    ),
                    ToolParameter(
                        name="response_format",
                        type="string",
                        description="Response format: 'concise' (summary) or 'detailed' (full report)",
                        required=False,
                        default="detailed"
                    )
                ],
                consolidates_operations=["ingest_data", "clean_data", "transform_data", "validate_quality", "export_results"],
                max_response_tokens=3000,
                common_workflows=["etl_pipeline", "data_processing", "batch_operations"]
            )
        })
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get definition for a specific tool"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> Dict[str, ToolDefinition]:
        """Get all tools in a specific category"""
        return {name: tool for name, tool in self.tools.items() if tool.category == category}
    
    def get_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """Convert to MCP-compatible tool definitions"""
        mcp_tools = []
        
        for tool_name, tool_def in self.tools.items():
            # Build parameter schema
            properties = {}
            required = []
            
            for param in tool_def.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                
                if param.examples:
                    properties[param.name]["examples"] = param.examples
                
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                
                if param.constraints:
                    properties[param.name].update(param.constraints)
                
                if param.required:
                    required.append(param.name)
            
            # Create MCP tool definition
            mcp_tool = {
                "name": tool_name,
                "description": self._create_enhanced_description(tool_def),
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            mcp_tools.append(mcp_tool)
        
        return mcp_tools
    
    def _create_enhanced_description(self, tool_def: ToolDefinition) -> str:
        """Create enhanced tool description with usage guidance"""
        description = tool_def.description
        
        # Add consolidation info
        if tool_def.consolidates_operations:
            description += f"\n\nConsolidates: {', '.join(tool_def.consolidates_operations)}"
        
        # Add format options
        if len(tool_def.response_formats) > 1:
            formats = [f.value for f in tool_def.response_formats]
            description += f"\n\nSupported formats: {', '.join(formats)}"
        
        # Add efficiency features
        features = []
        if tool_def.supports_pagination:
            features.append("pagination")
        if tool_def.supports_filtering:
            features.append("filtering")
        
        if features:
            description += f"\n\nEfficiency features: {', '.join(features)}"
        
        # Add usage examples
        if tool_def.usage_examples:
            description += "\n\nExample usage:"
            for example in tool_def.usage_examples[:2]:  # Limit to 2 examples
                description += f"\n- {example['description']}"
        
        return description
    
    def generate_tool_documentation(self) -> str:
        """Generate comprehensive tool documentation"""
        doc = "# NIS Protocol Enhanced Tool Documentation\n\n"
        
        # Group by category
        for category in ToolCategory:
            tools = self.get_tools_by_category(category)
            if not tools:
                continue
            
            doc += f"## {category.value.upper().replace('_', ' ')} Tools\n\n"
            
            for tool_name, tool_def in tools.items():
                doc += f"### `{tool_name}`\n\n"
                doc += f"{tool_def.description}\n\n"
                
                # Parameters
                doc += "**Parameters:**\n\n"
                for param in tool_def.parameters:
                    required_marker = " *(required)*" if param.required else ""
                    doc += f"- `{param.name}` ({param.type}){required_marker}: {param.description}\n"
                    
                    if param.examples:
                        doc += f"  - Examples: {param.examples}\n"
                    
                    if param.default is not None:
                        doc += f"  - Default: `{param.default}`\n"
                
                # Features
                features = []
                if tool_def.supports_pagination:
                    features.append("Pagination")
                if tool_def.supports_filtering:
                    features.append("Filtering")
                if len(tool_def.response_formats) > 1:
                    features.append("Multiple response formats")
                
                if features:
                    doc += f"\n**Features:** {', '.join(features)}\n"
                
                # Usage examples
                if tool_def.usage_examples:
                    doc += "\n**Usage Examples:**\n\n"
                    for example in tool_def.usage_examples:
                        doc += f"- {example['description']}\n"
                        doc += f"  ```json\n  {json.dumps(example['parameters'], indent=2)}\n  ```\n"
                
                doc += "\n---\n\n"
        
        return doc
    
    def validate_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a tool call against the schema"""
        if tool_name not in self.tools:
            return False, [f"Unknown tool: {tool_name}"]
        
        tool_def = self.tools[tool_name]
        errors = []
        
        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in parameters:
                errors.append(f"Missing required parameter: {param.name}")
        
        # Check parameter types and constraints
        for param_name, param_value in parameters.items():
            # Find parameter definition
            param_def = next((p for p in tool_def.parameters if p.name == param_name), None)
            if not param_def:
                errors.append(f"Unknown parameter: {param_name}")
                continue
            
            # Basic type checking
            if param_def.type == "string" and not isinstance(param_value, str):
                errors.append(f"Parameter '{param_name}' must be string")
            elif param_def.type == "number" and not isinstance(param_value, (int, float)):
                errors.append(f"Parameter '{param_name}' must be number")
            elif param_def.type == "boolean" and not isinstance(param_value, bool):
                errors.append(f"Parameter '{param_name}' must be boolean")
            elif param_def.type == "array" and not isinstance(param_value, list):
                errors.append(f"Parameter '{param_name}' must be array")
            elif param_def.type == "object" and not isinstance(param_value, dict):
                errors.append(f"Parameter '{param_name}' must be object")
            
            # Check constraints
            if param_def.constraints:
                if "min" in param_def.constraints and param_value < param_def.constraints["min"]:
                    errors.append(f"Parameter '{param_name}' below minimum: {param_def.constraints['min']}")
                if "max" in param_def.constraints and param_value > param_def.constraints["max"]:
                    errors.append(f"Parameter '{param_name}' above maximum: {param_def.constraints['max']}")
        
        return len(errors) == 0, errors


# Example usage and testing
def main():
    """Example usage of enhanced tool schemas"""
    schemas = EnhancedToolSchemas()
    
    # Get MCP definitions
    mcp_tools = schemas.get_mcp_tool_definitions()
    print(f"Generated {len(mcp_tools)} MCP tool definitions")
    
    # Test validation
    is_valid, errors = schemas.validate_tool_call(
        "nis_status", 
        {"detail_level": "summary", "include_agents": True}
    )
    print(f"Validation result: {is_valid}, Errors: {errors}")
    
    # Generate documentation
    doc = schemas.generate_tool_documentation()
    print(f"Generated documentation ({len(doc)} characters)")
    
    return schemas


if __name__ == "__main__":
    main()
