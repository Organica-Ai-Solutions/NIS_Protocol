#!/usr/bin/env python3
"""
Minimal FastAPI test to isolate numpy serialization issue
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from fastapi.responses import StreamingResponse
import json

app = FastAPI(title="Minimal Test")

class TestChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    agent_type: Optional[str] = "default"
    provider: Optional[str] = None

@app.post("/test/chat")
async def test_chat(request: TestChatRequest):
    """Minimal chat endpoint"""
    return {"response": f"Echo: {request.message}", "status": "success"}

@app.post("/test/stream")
async def test_stream(request: TestChatRequest):
    """Minimal streaming endpoint"""
    async def generate():
        yield f"data: " + json.dumps({"type": "content", "data": f"Echo: {request.message}"}) + "\n\n"
        yield f"data: " + json.dumps({"type": "done"}) + "\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
