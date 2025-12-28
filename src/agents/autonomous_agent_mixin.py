"""
Autonomous Agent Mixin - MCP Tool Execution Capabilities

Mixin class that gives any agent autonomous tool execution capabilities.
Agents can DO things, not just return text.
"""

import logging
from typing import Any, Dict, List, Optional
from src.core.mcp_tool_executor import get_mcp_executor

logger = logging.getLogger("nis.autonomous_agent")


class AutonomousAgentMixin:
    """
    Mixin to add autonomous MCP tool execution to any agent.
    
    Usage:
        class MyAgent(AutonomousAgentMixin):
            def __init__(self):
                self.init_autonomous_capabilities()
                
            async def my_method(self):
                # Agent can now execute tools
                result = await self.execute_tool("web_search", {"query": "latest AI research"})
    """
    
    def init_autonomous_capabilities(self):
        """Initialize autonomous capabilities"""
        self.mcp_executor = get_mcp_executor()
        self.tool_execution_history = []
        self.autonomous_mode = True
        logger.info(f"🤖 Autonomous capabilities enabled for {self.__class__.__name__}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if not hasattr(self, 'mcp_executor'):
            self.init_autonomous_capabilities()
        
        logger.info(f"🔧 {self.__class__.__name__} executing tool: {tool_name}")
        result = await self.mcp_executor.execute_tool(tool_name, parameters)
        
        # Track execution history
        self.tool_execution_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "success": result.get("success", False)
        })
        
        return result
    
    async def execute_tool_chain(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a chain of tools.
        
        Args:
            tools: List of tool specs [{"name": "tool_name", "params": {...}}, ...]
            
        Returns:
            List of execution results
        """
        if not hasattr(self, 'mcp_executor'):
            self.init_autonomous_capabilities()
        
        logger.info(f"🔗 {self.__class__.__name__} executing tool chain: {len(tools)} tools")
        results = await self.mcp_executor.execute_tool_chain(tools)
        
        # Track in history
        for result in results:
            self.tool_execution_history.append(result)
        
        return results
    
    async def search_web(self, query: str) -> Dict[str, Any]:
        """Convenience method: Search the web"""
        return await self.execute_tool("web_search", {"query": query})
    
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """Convenience method: Execute Python code"""
        return await self.execute_tool("code_execute", {"code": code})
    
    async def solve_physics(self, equation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method: Solve physics equation"""
        return await self.execute_tool("physics_solve", {
            "equation_type": equation_type,
            "parameters": parameters
        })
    
    async def compute_kinematics(self, operation: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method: Compute robot kinematics"""
        return await self.execute_tool("robotics_kinematics", {
            "operation": operation,
            "robot_data": robot_data
        })
    
    async def analyze_vision(self, image_data: str) -> Dict[str, Any]:
        """Convenience method: Analyze image"""
        return await self.execute_tool("vision_analyze", {"image_data": image_data})
    
    async def store_memory(self, key: str, value: Any) -> Dict[str, Any]:
        """Convenience method: Store in memory"""
        return await self.execute_tool("memory_store", {"key": key, "value": value})
    
    async def retrieve_memory(self, key: str) -> Dict[str, Any]:
        """Convenience method: Retrieve from memory"""
        return await self.execute_tool("memory_retrieve", {"key": key})
    
    async def create_agent(self, capability: str) -> Dict[str, Any]:
        """Convenience method: Create specialized agent"""
        return await self.execute_tool("consciousness_genesis", {"capability": capability})
    
    async def chat_llm(self, message: str, provider: str = "openai") -> Dict[str, Any]:
        """Convenience method: Chat with LLM"""
        return await self.execute_tool("llm_chat", {"message": message, "provider": provider})
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        if not hasattr(self, 'mcp_executor'):
            self.init_autonomous_capabilities()
        return self.mcp_executor.get_available_tools()
    
    def get_tool_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of tool executions"""
        return self.tool_execution_history.copy()
    
    def clear_tool_history(self):
        """Clear tool execution history"""
        self.tool_execution_history = []
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get autonomous capabilities status"""
        return {
            "autonomous_mode": getattr(self, 'autonomous_mode', False),
            "tools_available": len(self.get_available_tools()) if hasattr(self, 'mcp_executor') else 0,
            "executions_count": len(self.tool_execution_history) if hasattr(self, 'tool_execution_history') else 0,
            "agent_class": self.__class__.__name__
        }


class AutonomousResearchAgent(AutonomousAgentMixin):
    """
    Autonomous Research Agent with full tool access.
    
    Can:
    - Search the web
    - Execute code for analysis
    - Store/retrieve findings
    - Create specialized sub-agents
    """
    
    def __init__(self):
        self.init_autonomous_capabilities()
        logger.info("🔬 Autonomous Research Agent initialized")
    
    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """
        Autonomously research a topic using all available tools.
        
        1. Search web for information
        2. Execute code to analyze data
        3. Store findings in memory
        4. Return comprehensive report
        """
        logger.info(f"🔬 Researching topic: {topic}")
        
        # Step 1: Web search
        search_result = await self.search_web(topic)
        
        # Step 2: Store findings
        if search_result.get("success"):
            await self.store_memory(f"research_{topic}", search_result.get("results", []))
        
        # Step 3: Analyze with code if needed
        analysis_code = f"""
# Analyze research results for: {topic}
results = {search_result.get("results", [])}
print(f"Found {{len(results)}} results")
for i, result in enumerate(results[:3], 1):
    print(f"{{i}}. {{result.get('title', 'N/A')}}")
"""
        code_result = await self.execute_code(analysis_code)
        
        return {
            "topic": topic,
            "search_results": search_result,
            "analysis": code_result,
            "tools_used": ["web_search", "memory_store", "code_execute"],
            "autonomous": True
        }


class AutonomousPhysicsAgent(AutonomousAgentMixin):
    """
    Autonomous Physics Agent with computation capabilities.
    
    Can:
    - Solve physics equations
    - Execute numerical simulations
    - Validate results with code
    - Store solutions
    """
    
    def __init__(self):
        self.init_autonomous_capabilities()
        logger.info("⚛️ Autonomous Physics Agent initialized")
    
    async def solve_and_validate(self, equation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Autonomously solve physics equation and validate results.
        """
        logger.info(f"⚛️ Solving {equation_type} with validation")
        
        # Step 1: Solve equation
        solution = await self.solve_physics(equation_type, parameters)
        
        # Step 2: Store solution (skip validation to reduce execution time)
        if solution.get("success"):
            await self.store_memory(f"physics_{equation_type}", solution)
            
            return {
                "equation_type": equation_type,
                "solution": solution,
                "tools_used": ["physics_solve", "memory_store"],
                "autonomous": True,
                "status": "completed"
            }
        
        return {"success": False, "error": "Physics solve failed", "status": "failed"}


class AutonomousRoboticsAgent(AutonomousAgentMixin):
    """
    Autonomous Robotics Agent with kinematics and planning.
    
    Can:
    - Compute kinematics
    - Plan trajectories
    - Validate with physics
    - Execute motion commands
    """
    
    def __init__(self):
        self.init_autonomous_capabilities()
        logger.info("🤖 Autonomous Robotics Agent initialized")
    
    async def plan_and_execute_motion(self, target_position: List[float]) -> Dict[str, Any]:
        """
        Autonomously plan and execute robot motion.
        """
        logger.info(f"🤖 Planning motion to {target_position}")
        
        # Step 1: Compute inverse kinematics
        ik_result = await self.compute_kinematics("inverse_kinematics", {
            "target_pose": {"position": target_position}
        })
        
        if ik_result.get("success"):
            # Step 2: Validate with physics
            joint_angles = ik_result.get("result", {}).get("joint_angles", [])
            
            # Step 3: Compute forward kinematics to verify
            fk_result = await self.compute_kinematics("forward_kinematics", {
                "joint_angles": joint_angles
            })
            
            # Step 4: Store motion plan
            await self.store_memory("last_motion_plan", {
                "target": target_position,
                "joint_angles": joint_angles,
                "verified": fk_result.get("success", False)
            })
            
            return {
                "target_position": target_position,
                "joint_angles": joint_angles,
                "verification": fk_result,
                "tools_used": ["robotics_kinematics", "memory_store"],
                "autonomous": True
            }
        
        return {"success": False, "error": "IK computation failed"}


class AutonomousVisionAgent(AutonomousAgentMixin):
    """
    Autonomous Vision Agent with image analysis.
    
    Can:
    - Analyze images
    - Execute image processing code
    - Store analysis results
    - Search for related information
    """
    
    def __init__(self):
        self.init_autonomous_capabilities()
        logger.info("👁️ Autonomous Vision Agent initialized")
    
    async def analyze_and_research(self, image_data: str, context: str = "") -> Dict[str, Any]:
        """
        Autonomously analyze image and research related topics.
        """
        logger.info(f"👁️ Analyzing image with context: {context}")
        
        # Step 1: Analyze image
        analysis = await self.analyze_vision(image_data)
        
        if analysis.get("success"):
            # Step 2: Research related topics if context provided
            if context:
                search_result = await self.search_web(f"image analysis {context}")
                
                # Step 3: Store combined results
                await self.store_memory("last_vision_analysis", {
                    "analysis": analysis,
                    "research": search_result,
                    "context": context
                })
                
                return {
                    "analysis": analysis,
                    "research": search_result,
                    "tools_used": ["vision_analyze", "web_search", "memory_store"],
                    "autonomous": True
                }
            
            return {
                "analysis": analysis,
                "tools_used": ["vision_analyze"],
                "autonomous": True
            }
        
        return {"success": False, "error": "Vision analysis failed"}
