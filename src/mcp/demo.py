"""
MCP + Deep Agents + mcp-ui Integration Demo

Demonstrates the complete integration between Deep Agents, MCP server,
and mcp-ui for interactive agent experiences.
"""

import asyncio
import json
import logging
from typing import Dict, Any

from ..core.agent import NISAgent
from ..memory.memory_manager import MemoryManager
from .server import MCPServer


class MCPDemo:
    """
    Demonstration of the complete MCP + Deep Agents + mcp-ui stack.
    
    Shows how to:
    1. Initialize the MCP server with Deep Agents
    2. Handle tool requests and generate UI resources
    3. Process UI intents and maintain interactive sessions
    """
    
    def __init__(self):
        self.server = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup demo logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def initialize_server(self) -> MCPServer:
        """Initialize the MCP server with all components."""
        logging.info("Initializing MCP Server with Deep Agents...")
        
        # Create core components (simplified for demo)
        agent = NISAgent(agent_id="mcp_demo")  # Using real NIS agent
        memory_manager = MemoryManager()
        
        # Initialize MCP server
        self.server = MCPServer(agent, memory_manager)
        
        # Start the server
        await self.server.start_server()
        
        logging.info("MCP Server initialized successfully!")
        return self.server
        
    async def demo_dataset_search(self) -> Dict[str, Any]:
        """Demonstrate dataset search with UI generation."""
        logging.info("Demo: Dataset Search with mcp-ui")
        
        request = {
            "type": "tool",
            "tool_name": "dataset.search",
            "parameters": {
                "query": "weather data",
                "filters": {
                    "format": "csv",
                    "size_min": 1000
                },
                "limit": 10
            },
            "request_id": "demo_dataset_search",
            "user_id": "demo_user"
        }
        
        response = await self.server.handle_request(request)
        
        logging.info("Dataset search completed")
        logging.info(f"Success: {response['success']}")
        if response.get('ui_resource'):
            logging.info("Generated interactive data grid UI")
            
        return response
        
    async def demo_pipeline_run(self) -> Dict[str, Any]:
        """Demonstrate pipeline execution with progress monitoring."""
        logging.info("Demo: Pipeline Run with Progress Monitor")
        
        request = {
            "type": "tool",
            "tool_name": "pipeline.run",
            "parameters": {
                "pipeline_id": "demo_pipeline",
                "config": {
                    "input_dataset": "weather_data_2024",
                    "output_path": "/output/processed_weather.csv",
                    "parameters": {
                        "clean_missing": True,
                        "aggregate_hourly": True
                    }
                },
                "async": True
            },
            "request_id": "demo_pipeline_run"
        }
        
        response = await self.server.handle_request(request)
        
        logging.info("Pipeline execution started")
        logging.info(f"Run ID: {response.get('data', {}).get('run_id')}")
        if response.get('ui_resource'):
            logging.info("Generated progress monitor UI")
            
        return response
        
    async def demo_research_plan(self) -> Dict[str, Any]:
        """Demonstrate research planning with tree view."""
        logging.info("Demo: Research Plan Generation")
        
        request = {
            "type": "tool",
            "tool_name": "research.plan",
            "parameters": {
                "goal": "Analyze climate change impact on agricultural yields",
                "constraints": {
                    "timeline": "3 months",
                    "resources": ["academic papers", "government data", "satellite imagery"],
                    "scope": "North American crops"
                },
                "depth": "detailed"
            },
            "request_id": "demo_research_plan"
        }
        
        response = await self.server.handle_request(request)
        
        logging.info("Research plan generated")
        if response.get('ui_resource'):
            logging.info("Generated interactive research plan tree")
            
        return response
        
    async def demo_ui_intent_handling(self) -> Dict[str, Any]:
        """Demonstrate UI intent handling."""
        logging.info("Demo: UI Intent Handling")
        
        # Simulate a tool intent from UI
        intent_request = {
            "type": "intent",
            "intent_type": "tool",
            "payload": {
                "toolName": "dataset.preview",
                "params": {
                    "dataset_id": "weather_data_sample",
                    "sample_size": 50,
                    "include_schema": True
                }
            },
            "message_id": "ui_msg_123"
        }
        
        response = await self.server.handle_request(intent_request)
        
        logging.info("UI intent processed")
        if response.get('result', {}).get('ui_resource'):
            logging.info("Generated tabbed data preview UI")
            
        return response
        
    async def demo_code_review(self) -> Dict[str, Any]:
        """Demonstrate code review with review panel."""
        logging.info("Demo: Code Review with mcp-ui")
        
        request = {
            "type": "tool", 
            "tool_name": "code.review",
            "parameters": {
                "file_path": "/src/agents/dataset_agent.py",
                "review_type": "comprehensive",
                "standards": ["PEP8", "security", "performance"]
            },
            "request_id": "demo_code_review"
        }
        
        response = await self.server.handle_request(request)
        
        logging.info("Code review completed")
        if response.get('ui_resource'):
            logging.info("Generated code review panel UI")
            
        return response
        
    async def demo_audit_timeline(self) -> Dict[str, Any]:
        """Demonstrate audit trail viewing."""
        logging.info("Demo: Audit Timeline View")
        
        request = {
            "type": "tool",
            "tool_name": "audit.view",
            "parameters": {
                "timeframe": {
                    "start": "2025-01-19T00:00:00Z",
                    "end": "2025-01-19T23:59:59Z"
                },
                "level": "detailed"
            },
            "request_id": "demo_audit_view"
        }
        
        response = await self.server.handle_request(request)
        
        logging.info("Audit timeline generated")
        if response.get('ui_resource'):
            logging.info("Generated audit timeline UI")
            
        return response
        
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        logging.info("=" * 60)
        logging.info("MCP + Deep Agents + mcp-ui Integration Demo")
        logging.info("=" * 60)
        
        # Initialize server
        await self.initialize_server()
        
        # Show server capabilities
        server_info = self.server.get_server_info()
        logging.info(f"Server has {server_info['capabilities']['tools']} tools available")
        logging.info(f"Skills: {', '.join(server_info['skills'])}")
        
        # Run demos
        demos = [
            ("Dataset Search", self.demo_dataset_search),
            ("Pipeline Execution", self.demo_pipeline_run),
            ("Research Planning", self.demo_research_plan),
            ("UI Intent Handling", self.demo_ui_intent_handling),
            ("Code Review", self.demo_code_review),
            ("Audit Timeline", self.demo_audit_timeline)
        ]
        
        results = {}
        for demo_name, demo_func in demos:
            logging.info("-" * 40)
            try:
                result = await demo_func()
                results[demo_name] = {
                    "success": result.get("success", False),
                    "has_ui": "ui_resource" in result,
                    "data_type": type(result.get("data", {})).__name__
                }
                logging.info(f"✓ {demo_name} completed successfully")
            except Exception as e:
                logging.error(f"✗ {demo_name} failed: {str(e)}")
                results[demo_name] = {"success": False, "error": str(e)}
                
        # Summary
        logging.info("=" * 60)
        logging.info("Demo Summary:")
        successful_demos = sum(1 for r in results.values() if r.get("success"))
        ui_demos = sum(1 for r in results.values() if r.get("has_ui"))
        
        logging.info(f"Successful demos: {successful_demos}/{len(demos)}")
        logging.info(f"UI resources generated: {ui_demos}")
        
        for demo_name, result in results.items():
            status = "✓" if result.get("success") else "✗"
            ui_indicator = " [UI]" if result.get("has_ui") else ""
            logging.info(f"  {status} {demo_name}{ui_indicator}")
            
        logging.info("=" * 60)
        
        return results
        
    def export_server_schemas(self) -> str:
        """Export all server schemas for documentation."""
        if not self.server:
            return "Server not initialized"
            
        schemas = self.server.schemas.export_schemas("json")
        return schemas
        
    def get_ui_component_examples(self) -> Dict[str, str]:
        """Get examples of UI components that can be generated."""
        if not self.server:
            return {}
            
        examples = {}
        
        # Generate sample UI resources
        ui_gen = self.server.ui_generator
        
        # Data Grid example
        sample_data = [
            {"id": "ds_001", "name": "Weather Data", "format": "CSV", "size": "2.5MB"},
            {"id": "ds_002", "name": "Traffic Logs", "format": "JSON", "size": "850KB"}
        ]
        examples["Data Grid"] = ui_gen.create_data_grid(sample_data, "Sample Datasets")
        
        # Tabbed Viewer example
        sample_tabs = {
            "Schema": {"columns": ["timestamp", "temperature", "humidity"]},
            "Statistics": {"rows": 10000, "null_values": 12},
            "Sample": [{"timestamp": "2025-01-19", "temperature": 22.5}]
        }
        examples["Tabbed Viewer"] = ui_gen.create_tabbed_viewer(sample_tabs)
        
        # Progress Monitor example
        examples["Progress Monitor"] = ui_gen.create_progress_monitor(
            "run_123", "running", 45, ["Starting pipeline...", "Processing data..."]
        )
        
        return examples


async def run_demo():
    """Main demo runner."""
    demo = MCPDemo()
    results = await demo.run_complete_demo()
    return results


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo())
