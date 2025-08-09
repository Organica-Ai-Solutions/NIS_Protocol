"""
NIS Protocol v3 - Ultra Simple Version
Just FastAPI basics to establish working baseline
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Create FastAPI application
app = FastAPI(
    title="NIS Protocol v3 - Ultra Simple",
    description="Ultra simple baseline - step 1 of incremental build",
    version="3.0.0-ultra-simple"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
startup_time = time.time()

class HealthResponse(BaseModel):
    status: str
    uptime: float
    message: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: float

@app.get("/")
async def root():
    """Root endpoint"""
    uptime = time.time() - startup_time
    return {
        "system": "NIS Protocol v3",
        "version": "3.0.0-ultra-simple", 
        "status": "operational",
        "uptime_seconds": uptime,
        "message": "Ultra simple version working!",
        "endpoints": ["/", "/health", "/chat", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        message="Ultra simple version is healthy!"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Basic chat endpoint"""
    return ChatResponse(
        response=f"Hello {request.user_id}! You said: '{request.message}'. This is the ultra-simple version working!",
        user_id=request.user_id,
        timestamp=time.time()
    )

@app.get("/test")
async def test():
    """Test endpoint to verify everything works"""
    return {
        "test": "success",
        "timestamp": time.time(),
        "message": "Ultra simple NIS Protocol v3 is working!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 