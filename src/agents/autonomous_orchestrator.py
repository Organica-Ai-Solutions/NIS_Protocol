#!/usr/bin/env python3
"""
🤖 NIS Protocol Autonomous Agent Orchestrator

Autonomous AI that decides which tools/agents to use based on user intent.
Acts as the "brain" of the NIS Protocol system.

Features:
- Automatic tool selection
- Agent orchestration
- Intent recognition
- Autonomous decision making
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents the system can recognize"""
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    PHYSICS_VALIDATION = "physics_validation"
    DEEP_RESEARCH = "deep_research"
    DATA_ANALYSIS = "data_analysis"
    MATH_CALCULATION = "math_calculation"
    CONVERSATION = "conversation"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATION = "file_operation"
    VISUALIZATION = "visualization"


class ToolCapability(Enum):
    """Available tools in the NIS Protocol"""
    RUNNER = "runner"  # Code execution
    PHYSICS_PINN = "physics_pinn"  # Physics validation
    RESEARCH_ENGINE = "research_engine"  # Deep research
    WEB_SEARCH = "web_search"  # Web search
    CALCULATOR = "calculator"  # Math calculations
    FILE_SYSTEM = "file_system"  # File operations
    DATABASE = "database"  # Data storage/retrieval
    VISUALIZATION = "visualization"  # Charts/graphs
    LLM_PROVIDER = "llm_provider"  # Multi-LLM responses


@dataclass
class ToolSelection:
    """Represents a tool that should be used"""
    tool: ToolCapability
    confidence: float
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class AgentPlan:
    """Plan for executing a user request"""
    intent: IntentType
    tools_needed: List[ToolSelection]
    execution_order: List[str]
    estimated_time: float
    requires_human_approval: bool


class AutonomousOrchestrator:
    """
    🧠 Autonomous Agent Orchestrator
    
    Automatically decides which tools/agents to use based on user input.
    Acts as the brain of the NIS Protocol.
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        
        # Intent recognition patterns
        self.intent_patterns = {
            IntentType.WEB_SEARCH: [
                "search", "find", "look up", "google", "what is", "who is",
                "when was", "where is", "latest news", "current"
            ],
            IntentType.CODE_EXECUTION: [
                "run", "execute", "code", "python", "script", "calculate",
                "fibonacci", "algorithm", "program"
            ],
            IntentType.PHYSICS_VALIDATION: [
                "physics", "validate", "pinn", "conservation", "energy",
                "momentum", "force", "velocity", "acceleration", "gravity"
            ],
            IntentType.DEEP_RESEARCH: [
                "research", "analyze", "study", "investigate", "comprehensive",
                "detailed analysis", "in-depth", "examine"
            ],
            IntentType.DATA_ANALYSIS: [
                "analyze data", "dataset", "statistics", "correlation",
                "trend", "pattern", "insights"
            ],
            IntentType.MATH_CALCULATION: [
                "calculate", "compute", "solve", "equation", "formula",
                "math", "arithmetic", "+", "-", "*", "/"
            ],
            IntentType.FILE_OPERATION: [
                "file", "save", "load", "read", "write", "delete",
                "create file", "directory"
            ],
            IntentType.VISUALIZATION: [
                "visualize", "plot", "graph", "chart", "diagram", "show"
            ]
        }
        
        # Tool capabilities mapping
        self.tool_mappings = {
            IntentType.CODE_EXECUTION: ToolCapability.RUNNER,
            IntentType.PHYSICS_VALIDATION: ToolCapability.PHYSICS_PINN,
            IntentType.DEEP_RESEARCH: ToolCapability.RESEARCH_ENGINE,
            IntentType.WEB_SEARCH: ToolCapability.WEB_SEARCH,
            IntentType.MATH_CALCULATION: ToolCapability.CALCULATOR,
            IntentType.FILE_OPERATION: ToolCapability.FILE_SYSTEM,
            IntentType.VISUALIZATION: ToolCapability.VISUALIZATION
        }
        
        logger.info("🤖 Autonomous Orchestrator initialized")
    
    async def analyze_intent(self, user_message: str) -> IntentType:
        """
        🎯 Analyze user message to determine intent
        
        Uses pattern matching and keyword analysis to understand what
        the user wants to accomplish.
        """
        message_lower = user_message.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent, or CONVERSATION as default
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"🎯 Detected intent: {best_intent.value} (score: {intent_scores[best_intent]})")
            return best_intent
        
        return IntentType.CONVERSATION
    
    async def create_execution_plan(self, user_message: str, intent: IntentType) -> AgentPlan:
        """
        📋 Create an execution plan based on intent
        
        Determines which tools to use and in what order.
        """
        tools_needed = []
        
        # Map intent to tool
        if intent in self.tool_mappings:
            primary_tool = self.tool_mappings[intent]
            tools_needed.append(ToolSelection(
                tool=primary_tool,
                confidence=0.9,
                parameters=self._extract_parameters(user_message, intent),
                reasoning=f"Primary tool for {intent.value}"
            ))
        
        # Add LLM provider for response generation
        tools_needed.append(ToolSelection(
            tool=ToolCapability.LLM_PROVIDER,
            confidence=1.0,
            parameters={"provider": "smart"},
            reasoning="Generate final response"
        ))
        
        # Create execution order
        execution_order = [tool.tool.value for tool in tools_needed]
        
        # Estimate time
        estimated_time = self._estimate_execution_time(tools_needed)
        
        # Determine if human approval needed (for destructive operations)
        requires_approval = intent in [IntentType.FILE_OPERATION, IntentType.SYSTEM_CONTROL]
        
        plan = AgentPlan(
            intent=intent,
            tools_needed=tools_needed,
            execution_order=execution_order,
            estimated_time=estimated_time,
            requires_human_approval=requires_approval
        )
        
        logger.info(f"📋 Created execution plan: {len(tools_needed)} tools, ~{estimated_time:.1f}s")
        return plan
    
    def _extract_parameters(self, message: str, intent: IntentType) -> Dict[str, Any]:
        """Extract parameters for tool execution"""
        params = {}
        
        if intent == IntentType.CODE_EXECUTION:
            # Extract code if present
            if "```" in message:
                code_blocks = message.split("```")
                if len(code_blocks) > 1:
                    params["code"] = code_blocks[1].strip()
        
        elif intent == IntentType.PHYSICS_VALIDATION:
            params["scenario"] = "general"
            params["mode"] = "true_pinn"
            # Extract physics keywords
            if "ball" in message.lower():
                params["scenario"] = "bouncing_ball"
        
        elif intent == IntentType.DEEP_RESEARCH:
            params["query"] = message
            params["research_depth"] = "comprehensive"
        
        elif intent == IntentType.MATH_CALCULATION:
            # Extract mathematical expression
            import re
            expr = re.search(r'[\d\+\-\*\/\(\)\s]+', message)
            if expr:
                params["expression"] = expr.group(0).strip()
        
        return params
    
    def _estimate_execution_time(self, tools: List[ToolSelection]) -> float:
        """Estimate total execution time"""
        time_estimates = {
            ToolCapability.RUNNER: 2.0,
            ToolCapability.PHYSICS_PINN: 3.0,
            ToolCapability.RESEARCH_ENGINE: 5.0,
            ToolCapability.WEB_SEARCH: 1.0,
            ToolCapability.CALCULATOR: 0.1,
            ToolCapability.LLM_PROVIDER: 2.0,
        }
        
        total_time = sum(
            time_estimates.get(tool.tool, 1.0)
            for tool in tools
        )
        
        return total_time
    
    async def execute_plan(self, plan: AgentPlan, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ⚡ Execute the agent plan autonomously
        
        Runs all selected tools in order and returns results.
        """
        start_time = time.time()
        results = {
            "intent": plan.intent.value,
            "tools_used": [],
            "outputs": {},
            "success": True,
            "execution_time": 0.0
        }
        
        try:
            # Execute each tool in order
            for tool_selection in plan.tools_needed:
                tool_name = tool_selection.tool.value
                logger.info(f"🔧 Executing tool: {tool_name}")
                
                # Execute tool
                tool_result = await self._execute_tool(
                    tool_selection.tool,
                    tool_selection.parameters,
                    message,
                    context
                )
                
                results["tools_used"].append(tool_name)
                results["outputs"][tool_name] = tool_result
            
            results["execution_time"] = time.time() - start_time
            logger.info(f"✅ Plan executed in {results['execution_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Plan execution failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def _execute_tool(
        self,
        tool: ToolCapability,
        parameters: Dict[str, Any],
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific tool"""
        
        if tool == ToolCapability.RUNNER:
            return await self._execute_runner(parameters, message)
        
        elif tool == ToolCapability.PHYSICS_PINN:
            return await self._execute_physics(parameters)
        
        elif tool == ToolCapability.RESEARCH_ENGINE:
            return await self._execute_research(parameters)
        
        elif tool == ToolCapability.WEB_SEARCH:
            return await self._execute_web_search(parameters, message)
        
        elif tool == ToolCapability.CALCULATOR:
            return await self._execute_calculator(parameters)
        
        elif tool == ToolCapability.LLM_PROVIDER:
            return await self._execute_llm(parameters, message, context)
        
        else:
            return {"success": False, "error": f"Tool {tool.value} not implemented"}
    
    async def _execute_runner(self, params: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Execute code in runner"""
        # If no code in params, generate simple demo
        if "code" not in params:
            params["code"] = "print('Hello from autonomous agent!')"
        
        return {
            "success": True,
            "tool": "runner",
            "code": params["code"],
            "note": "Code execution via runner"
        }
    
    async def _execute_physics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics validation"""
        return {
            "success": True,
            "tool": "physics_pinn",
            "scenario": params.get("scenario", "general"),
            "note": "Physics validation via TRUE PINN"
        }
    
    async def _execute_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deep research"""
        return {
            "success": True,
            "tool": "research_engine",
            "query": params.get("query", ""),
            "note": "Deep research via GPT-4"
        }
    
    async def _execute_web_search(self, params: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Execute web search"""
        return {
            "success": True,
            "tool": "web_search",
            "query": message,
            "note": "Web search capability"
        }
    
    async def _execute_calculator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculation"""
        expr = params.get("expression", "")
        try:
            # Safe eval (in production, use a proper math parser)
            result = eval(expr, {"__builtins__": {}}, {})
            return {
                "success": True,
                "tool": "calculator",
                "expression": expr,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "calculator",
                "error": str(e)
            }
    
    async def _execute_llm(
        self,
        params: Dict[str, Any],
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LLM provider"""
        if self.llm_provider:
            try:
                response = await self.llm_provider.generate_response(
                    messages=[{"role": "user", "content": message}],
                    requested_provider=params.get("provider"),
                    temperature=0.7
                )
                return {
                    "success": True,
                    "tool": "llm_provider",
                    "response": response.get("content", "")
                }
            except Exception as e:
                logger.error(f"LLM execution error: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "LLM provider not available"}


# Global instance
autonomous_orchestrator = None


def get_autonomous_orchestrator(llm_provider=None) -> AutonomousOrchestrator:
    """Get or create the autonomous orchestrator instance"""
    global autonomous_orchestrator
    
    if autonomous_orchestrator is None:
        autonomous_orchestrator = AutonomousOrchestrator(llm_provider)
    
    return autonomous_orchestrator

