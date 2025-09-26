from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

app = FastAPI(title="NIS Protocol v3.2 - Simplified")

@app.get("/health")
async def health():
    return JSONResponse(content={
        "status": "healthy", 
        "message": "NIS Protocol server running (simplified mode)",
        "timestamp": 1234567890
    })

@app.post("/chat")
async def chat(request: dict):
    message = request.get("message", "")
    user_id = request.get("user_id", "anonymous")
    
    return JSONResponse(content={
        "response": f"Received your message: {message[:100]}...",
        "user_id": user_id,
        "timestamp": 1234567890,
        "confidence": 0.95,
        "provider": "simplified",
        "model": "fallback"
    })

@app.get("/consciousness/status")
async def consciousness_status():
    return JSONResponse(content={
        "consciousness_level": 0.95,
        "introspection_active": True,
        "awareness_metrics": {
            "self_awareness": 0.8,
            "environmental_awareness": 0.9,
            "goal_clarity": 0.85
        },
        "system_metrics": {
            "active_conversations": 1,
            "uptime": 1000
        },
        "cognitive_state": {
            "attention_focus": "high",
            "learning_mode": "active"
        },
        "timestamp": 1234567890
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting simplified NIS Protocol server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
