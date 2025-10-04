#!/usr/bin/env python3
"""
Example 3: FastAPI Integration

This example shows how to integrate NIS Protocol into a FastAPI application.
Run with: uvicorn examples.03_fastapi_integration:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nis_protocol import NISCore

# Initialize FastAPI
app = FastAPI(
    title="NIS Protocol API",
    description="Example integration of NIS Protocol with FastAPI",
    version="1.0.0"
)

# Initialize NIS Core
nis = NISCore()


class ChatRequest(BaseModel):
    message: str
    provider: str | None = None


class AutonomousRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "NIS Protocol API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "autonomous": "/autonomous",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "nis_protocol": "active",
        "version": "3.2.1"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Simple chat endpoint using specific LLM provider
    
    Example:
        POST /chat
        {
            "message": "Hello, world!",
            "provider": "openai"
        }
    """
    try:
        response = nis.get_llm_response(
            request.message,
            provider=request.provider
        )
        
        return {
            "success": True,
            "response": response.get("content", ""),
            "provider": request.provider or "auto"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autonomous")
async def autonomous(request: AutonomousRequest):
    """
    Autonomous processing endpoint - system decides what to do!
    
    Example:
        POST /autonomous
        {
            "message": "Calculate fibonacci(10)"
        }
    """
    try:
        result = await nis.process_autonomously(request.message)
        
        return {
            "success": result["success"],
            "intent": result["intent"],
            "tools_used": result["tools_used"],
            "reasoning": result["reasoning"],
            "response": result["response"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def info():
    """Get NIS Protocol information"""
    from nis_protocol import get_info
    return get_info()


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          NIS Protocol FastAPI Integration Example             ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Starting server on http://localhost:8000
    
    Endpoints:
      - http://localhost:8000/          (Root)
      - http://localhost:8000/docs      (API Documentation)
      - http://localhost:8000/chat      (Simple chat)
      - http://localhost:8000/autonomous (Autonomous mode)
      - http://localhost:8000/health    (Health check)
      - http://localhost:8000/info      (System info)
    
    Test with curl:
      curl -X POST http://localhost:8000/chat \\
        -H "Content-Type: application/json" \\
        -d '{"message": "Hello!"}'
      
      curl -X POST http://localhost:8000/autonomous \\
        -H "Content-Type: application/json" \\
        -d '{"message": "Calculate 2+2"}'
    
    Press CTRL+C to stop
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

