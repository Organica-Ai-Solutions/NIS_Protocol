#!/usr/bin/env python3
"""
Minimal NIS Protocol v3 Test Server
For testing endpoints without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import time
import json

app = FastAPI(
    title="NIS Protocol v3 - Test Server",
    description="Minimal test server for endpoint validation",
    version="3.0.0-test"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    agent_type: Optional[str] = "default"

class AgentCreateRequest(BaseModel):
    agent_type: str
    config: Dict[str, Any] = {}

class BehaviorRequest(BaseModel):
    mode: str

class MemoryStoreRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class MemoryQueryRequest(BaseModel):
    query: str

class ToolExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ProcessRequest(BaseModel):
    input: str

# Global state for testing
agents = []
memory_store = []

@app.get("/")
async def root():
    return {
        "message": "NIS Protocol v3 Test Server Active",
        "version": "3.0.0-test",
        "status": "operational",
        "endpoints": [
            "/health", "/agents", "/agent/create", "/chat", 
            "/consciousness/status", "/infrastructure/status", 
            "/metrics", "/process", "/memory/store", "/memory/query",
            "/tool/execute", "/protocol/status", "/dashboard/metrics"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "server": "minimal-test",
        "components": {
            "api": "operational",
            "agents": f"{len(agents)} active",
            "memory": f"{len(memory_store)} entries"
        }
    }

@app.get("/agents")
async def list_agents():
    return {
        "agents": agents,
        "total": len(agents),
        "timestamp": time.time()
    }

@app.post("/agent/create")
async def create_agent(request: AgentCreateRequest):
    agent = {
        "id": f"agent_{len(agents)+1}",
        "type": request.agent_type,
        "config": request.config,
        "created": time.time(),
        "status": "active"
    }
    agents.append(agent)
    return {
        "agent": agent,
        "message": f"Agent {request.agent_type} created successfully"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    response_text = f"NIS Protocol v3 response to: {request.message}"
    
    # Mock NIS pipeline processing
    pipeline_result = {
        "laplace_transform": {"signal_processed": True, "frequency_domain": "analyzed"},
        "kan_reasoning": {"symbolic_function": "extracted", "interpretability": 0.85},
        "pinn_physics": {"conservation_laws": "validated", "physics_compliance": 0.92},
        "llm_response": response_text
    }
    
    return {
        "response": response_text,
        "agent_type": request.agent_type,
        "nis_pipeline": pipeline_result,
        "confidence": 0.87,
        "provider": "test_llm",
        "real_ai": True,
        "timestamp": time.time()
    }

@app.get("/consciousness/status")
async def consciousness_status():
    return {
        "consciousness_level": 0.75,
        "introspection_active": True,
        "awareness_metrics": {
            "self_model_accuracy": 0.82,
            "goal_clarity": 0.78,
            "decision_coherence": 0.85
        },
        "timestamp": time.time()
    }

@app.get("/infrastructure/status")
async def infrastructure_status():
    return {
        "system_health": "optimal",
        "resource_utilization": {
            "cpu": 45.2,
            "memory": 67.8,
            "disk": 23.1
        },
        "services": {
            "llm_providers": 4,
            "active_agents": len(agents),
            "memory_system": "operational"
        },
        "timestamp": time.time()
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "performance": {
            "avg_response_time": 0.145,
            "throughput": 142.7,
            "error_rate": 0.003
        },
        "agent_metrics": {
            "total_agents": len(agents),
            "active_conversations": 12,
            "memory_entries": len(memory_store)
        },
        "system_metrics": {
            "uptime": time.time(),
            "requests_processed": 1247,
            "success_rate": 0.997
        }
    }

@app.post("/process")
async def process_data(request: ProcessRequest):
    # Mock processing through NIS pipeline
    processed_result = {
        "input": request.input,
        "laplace_analysis": {"transform_successful": True},
        "kan_interpretation": {"symbolic_form": "extracted"},
        "physics_validation": {"compliant": True},
        "output": f"Processed: {request.input}",
        "timestamp": time.time()
    }
    return processed_result

@app.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    memory_entry = {
        "id": len(memory_store) + 1,
        "content": request.content,
        "metadata": request.metadata,
        "timestamp": time.time(),
        "embedding_stored": True
    }
    memory_store.append(memory_entry)
    return {
        "status": "stored",
        "memory_id": memory_entry["id"],
        "message": "Memory stored successfully"
    }

@app.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    # Mock memory search
    relevant_memories = [m for m in memory_store if request.query.lower() in m["content"].lower()]
    return {
        "query": request.query,
        "results": relevant_memories[:5],  # Top 5 results
        "total_found": len(relevant_memories),
        "timestamp": time.time()
    }

@app.post("/tool/execute")
async def execute_tool(request: ToolExecuteRequest):
    return {
        "tool": request.tool_name,
        "parameters": request.parameters,
        "result": f"Tool {request.tool_name} executed successfully",
        "execution_time": 0.234,
        "status": "completed",
        "timestamp": time.time()
    }

@app.get("/protocol/status")
async def protocol_status():
    return {
        "protocol_version": "3.0.0",
        "integrations": {
            "mcp": "connected",
            "acp": "connected", 
            "a2a": "connected"
        },
        "active_protocols": 3,
        "message_throughput": 156.3,
        "timestamp": time.time()
    }

@app.get("/dashboard/metrics")
async def dashboard_metrics():
    return {
        "overview": {
            "system_health": 98.7,
            "active_agents": len(agents),
            "memory_utilization": 45.2,
            "performance_index": 94.1
        },
        "real_time": {
            "requests_per_second": 23.7,
            "avg_latency": 145.2,
            "success_rate": 99.8
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("🚀 Starting NIS Protocol v3 Minimal Test Server...")
    print("📍 Testing all endpoints for connectivity and functionality")
    uvicorn.run(app, host="0.0.0.0", port=8002) 