"""
MCP Tool Executor - Autonomous Agent Capabilities

Provides real tool execution for agents to DO things, not just return text.
All agents can use these tools to be truly autonomous.
"""

import asyncio
import logging
import httpx
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger("nis.mcp_tool_executor")


class MCPToolExecutor:
    """
    Centralized MCP tool executor for autonomous agent capabilities.
    
    Agents use this to execute real actions:
    - code_execute: Run Python code
    - web_search: Search the web
    - physics_solve: Solve physics equations
    - robotics_kinematics: Compute robot kinematics
    - vision_analyze: Analyze images
    - memory_store/retrieve: Persistent memory
    - consciousness_genesis: Create specialized agents
    - llm_chat: Call LLM providers
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.available_tools = [
            "code_execute",
            "web_search", 
            "physics_solve",
            "robotics_kinematics",
            "vision_analyze",
            "memory_store",
            "memory_retrieve",
            "consciousness_genesis",
            "llm_chat",
            "file_read",
            "file_write",
            "file_list",
            "file_exists",
            "db_query",
            "db_schema",
            "db_tables"
        ]
        logger.info(f"🔧 MCP Tool Executor initialized with {len(self.available_tools)} tools")
    
    def _execute_file_operation(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute file operation locally (no HTTP call needed).
        
        Args:
            tool_name: file_read, file_write, file_list, or file_exists
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            from src.tools.file_operations import get_file_operations_tool
            file_ops = get_file_operations_tool()
            
            if tool_name == "file_read":
                return file_ops.file_read(
                    file_path=parameters.get("file_path", ""),
                    encoding=parameters.get("encoding", "utf-8")
                )
            elif tool_name == "file_write":
                return file_ops.file_write(
                    file_path=parameters.get("file_path", ""),
                    content=parameters.get("content", ""),
                    encoding=parameters.get("encoding", "utf-8"),
                    create_dirs=parameters.get("create_dirs", True)
                )
            elif tool_name == "file_list":
                return file_ops.file_list(
                    directory=parameters.get("directory", "."),
                    pattern=parameters.get("pattern", "*")
                )
            elif tool_name == "file_exists":
                return file_ops.file_exists(
                    file_path=parameters.get("file_path", "")
                )
            else:
                return {"success": False, "error": f"Unknown file operation: {tool_name}"}
                
        except Exception as e:
            logger.error(f"File operation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_database_operation(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute database operation locally (no HTTP call needed).
        
        Args:
            tool_name: db_query, db_schema, or db_tables
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            from src.tools.database_query import get_database_query_tool
            db_tool = get_database_query_tool()
            
            if tool_name == "db_query":
                return db_tool.db_query(
                    db_path=parameters.get("db_path", ""),
                    query=parameters.get("query", ""),
                    params=parameters.get("params")
                )
            elif tool_name == "db_schema":
                return db_tool.db_schema(
                    db_path=parameters.get("db_path", ""),
                    table_name=parameters.get("table_name")
                )
            elif tool_name == "db_tables":
                return db_tool.db_tables(
                    db_path=parameters.get("db_path", "")
                )
            else:
                return {"success": False, "error": f"Unknown database operation: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": self.available_tools
            }
        
        try:
            # Handle file operations locally (no HTTP call)
            if tool_name in ["file_read", "file_write", "file_list", "file_exists"]:
                return self._execute_file_operation(tool_name, parameters)
            
            # Handle database operations locally (no HTTP call)
            if tool_name in ["db_query", "db_schema", "db_tables"]:
                return self._execute_database_operation(tool_name, parameters)
            
            # Route to appropriate tool endpoint
            if tool_name == "code_execute":
                return await self._execute_code(parameters)
            elif tool_name == "web_search":
                return await self._web_search(parameters)
            elif tool_name == "physics_solve":
                return await self._physics_solve(parameters)
            elif tool_name == "robotics_kinematics":
                return await self._robotics_kinematics(parameters)
            elif tool_name == "vision_analyze":
                return await self._vision_analyze(parameters)
            elif tool_name == "memory_store":
                return await self._memory_store(parameters)
            elif tool_name == "memory_retrieve":
                return await self._memory_retrieve(parameters)
            elif tool_name == "consciousness_genesis":
                return await self._consciousness_genesis(parameters)
            elif tool_name == "llm_chat":
                return await self._llm_chat(parameters)
            else:
                return {"success": False, "error": f"Tool not implemented: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in sandboxed runner"""
        code = params.get("code", "")
        if not code:
            return {"success": False, "error": "No code provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://nis-runner-cpu:8001/execute",
                    json={"code_content": code},
                    timeout=10.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "error": result.get("error", ""),
                        "tool": "code_execute"
                    }
                else:
                    return {"success": False, "error": f"Runner returned {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Code execution failed: {str(e)}"}
    
    async def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search the web"""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "No query provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/research/query",
                    json={"query": query},
                    timeout=10.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "results": result.get("results", []),
                        "query": query,
                        "tool": "web_search"
                    }
                else:
                    return {"success": False, "error": f"Search failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Web search failed: {str(e)}"}
    
    async def _physics_solve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve physics equations"""
        equation_type = params.get("equation_type", "heat-equation")
        parameters = params.get("parameters", {})
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/physics/solve/{equation_type}",
                    json=parameters,
                    timeout=60.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "solution": result.get("solution", {}),
                        "equation_type": equation_type,
                        "tool": "physics_solve"
                    }
                else:
                    return {"success": False, "error": f"Physics solve failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Physics solve failed: {str(e)}"}
    
    async def _robotics_kinematics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute robot kinematics"""
        operation = params.get("operation", "forward_kinematics")
        robot_data = params.get("robot_data", {})
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/robotics/{operation}",
                    json=robot_data,
                    timeout=10.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "result": result,
                        "operation": operation,
                        "tool": "robotics_kinematics"
                    }
                else:
                    return {"success": False, "error": f"Kinematics failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Kinematics failed: {str(e)}"}
    
    async def _vision_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an image"""
        image_data = params.get("image_data", "") or params.get("image_url", "")
        if not image_data:
            return {"success": False, "error": "No image data provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/vision/analyze",
                    json={"image_data": image_data},
                    timeout=15.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "analysis": result.get("analysis", {}),
                        "tool": "vision_analyze"
                    }
                else:
                    return {"success": False, "error": f"Vision analysis failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Vision analysis failed: {str(e)}"}
    
    async def _memory_store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in persistent memory"""
        key = params.get("key", "")
        value = params.get("value")
        
        if not key:
            return {"success": False, "error": "No key provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/memory/store",
                    json={"key": key, "value": value},
                    timeout=5.0
                )
                if response.status_code == 200:
                    return {
                        "success": True,
                        "key": key,
                        "tool": "memory_store"
                    }
                else:
                    return {"success": False, "error": f"Memory store failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Memory store failed: {str(e)}"}
    
    async def _memory_retrieve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data from persistent memory"""
        key = params.get("key", "")
        
        if not key:
            return {"success": False, "error": "No key provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/memory/retrieve/{key}",
                    timeout=5.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "key": key,
                        "value": result.get("value"),
                        "tool": "memory_retrieve"
                    }
                else:
                    return {"success": False, "error": f"Memory retrieve failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Memory retrieve failed: {str(e)}"}
    
    async def _consciousness_genesis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specialized agent"""
        capability = params.get("capability", "")
        
        if not capability:
            return {"success": False, "error": "No capability specified"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v4/consciousness/genesis",
                    json={"capability": capability},
                    timeout=10.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "agent": result.get("agent", {}),
                        "capability": capability,
                        "tool": "consciousness_genesis"
                    }
                else:
                    return {"success": False, "error": f"Genesis failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Genesis failed: {str(e)}"}
    
    async def _llm_chat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to LLM provider"""
        message = params.get("message", "")
        provider = params.get("provider", "openai")
        
        if not message:
            return {"success": False, "error": "No message provided"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json={"message": message, "provider": provider},
                    timeout=30.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "provider": provider,
                        "tool": "llm_chat"
                    }
                else:
                    return {"success": False, "error": f"LLM chat failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"LLM chat failed: {str(e)}"}
    
    async def execute_tool_chain(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a chain of tools sequentially.
        
        Args:
            tools: List of tool specifications [{"name": "tool_name", "params": {...}}, ...]
            
        Returns:
            List of execution results
        """
        results = []
        
        for tool_spec in tools:
            tool_name = tool_spec.get("name")
            params = tool_spec.get("params", {})
            
            logger.info(f"🔧 Executing tool: {tool_name}")
            result = await self.execute_tool(tool_name, params)
            results.append({
                "tool": tool_name,
                "result": result,
                "success": result.get("success", False)
            })
            
            # Stop chain if a tool fails
            if not result.get("success", False):
                logger.warning(f"⚠️ Tool chain stopped at {tool_name}: {result.get('error')}")
                break
        
        return results
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return self.available_tools.copy()
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description of a specific tool"""
        descriptions = {
            "code_execute": "Execute Python code in sandboxed environment - run calculations, data analysis, algorithms",
            "web_search": "Search the web for real-time information - research, fact-checking, current events",
            "physics_solve": "Solve physics equations using neural networks - heat, wave, laplace equations",
            "robotics_kinematics": "Compute robot kinematics - forward/inverse kinematics, trajectory planning",
            "vision_analyze": "Analyze images - object detection, scene understanding, visual reasoning",
            "memory_store": "Store data in persistent memory - save state, cache results, remember context",
            "memory_retrieve": "Retrieve data from persistent memory - load state, recall information",
            "consciousness_genesis": "Create specialized agents - spawn new agents with specific capabilities",
            "llm_chat": "Call LLM providers - reasoning, analysis, text generation"
        }
        return descriptions.get(tool_name)


# Global executor instance
_global_executor: Optional[MCPToolExecutor] = None


def get_mcp_executor() -> MCPToolExecutor:
    """Get or create global MCP tool executor"""
    global _global_executor
    if _global_executor is None:
        _global_executor = MCPToolExecutor()
    return _global_executor


async def execute_autonomous_task(task_description: str, available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Execute an autonomous task using available MCP tools.
    
    This is the main entry point for autonomous agent behavior.
    The agent decides which tools to use and executes them.
    
    Args:
        task_description: Natural language description of the task
        available_tools: Optional list of tools to restrict to
        
    Returns:
        Task execution result with tool usage trace
    """
    executor = get_mcp_executor()
    
    # For now, return the executor and task description
    # In a full implementation, this would use an LLM to plan and execute
    return {
        "status": "autonomous_execution_ready",
        "task": task_description,
        "available_tools": available_tools or executor.get_available_tools(),
        "executor": executor
    }
