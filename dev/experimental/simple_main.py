"""
NIS Protocol v3 - Simplified Main Application
Based on working implementations - start simple, then expand
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Core imports that we know work
from src.utils.env_config import EnvironmentConfig
from src.cognitive_agents.cognitive_system import CognitiveSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nis_main")

# Global state
cognitive_system = None
startup_time = None

class ProcessRequest(BaseModel):
    text: str
    generate_speech: bool = False

class ProcessResponse(BaseModel):
    response_text: str
    processing_time: float
    timestamp: float

class HealthResponse(BaseModel):
    status: str
    uptime: float
    components: Dict[str, str]
    metrics: Dict[str, float]

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    timestamp: float
    confidence: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simple application lifespan manager."""
    global cognitive_system, startup_time
    
    logger.info("🚀 Starting NIS Protocol v3 - Simplified Mode")
    startup_time = time.time()
    
    try:
        # Initialize environment configuration
        env_config = EnvironmentConfig()
        logger.info("✅ Environment configuration loaded")
        
        # Initialize cognitive system (core functionality)
        try:
            cognitive_system = CognitiveSystem()
            logger.info("✅ Cognitive system initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Cognitive system initialization failed: {e}")
            logger.info("🔄 Continuing in basic mode")
        
        # Store global state
        app.state.startup_time = startup_time
        app.state.cognitive_system = cognitive_system
        app.state.mode = "simplified"
        
        logger.info("🎉 NIS Protocol v3 ready in simplified mode!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        logger.exception("Startup error details:")
        raise
    
    # Cleanup
    logger.info("🛑 Shutting down NIS Protocol v3")

# Create FastAPI application
app = FastAPI(
    title="NIS Protocol v3 - Simplified",
    description="Neural Intelligence Synthesis Protocol v3 - Simplified Working Version",
    version="3.0.0-simplified",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== CORE ENDPOINTS ===============

@app.get("/")
async def system_info():
    """System information endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    return {
        "system": "NIS Protocol v3",
        "version": "3.0.0-simplified",
        "status": "operational",
        "mode": "simplified",
        "uptime_seconds": uptime,
        "endpoints": {
            "health": "/health",
            "process": "/process", 
            "chat": "/chat",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    uptime = time.time() - startup_time if startup_time else 0
    
    components = {
        "cognitive_system": "healthy" if cognitive_system else "unavailable",
        "api": "healthy"
    }
    
    metrics = {
        "uptime_seconds": uptime,
        "memory_usage_mb": 0.0,  # Placeholder
        "active_connections": 1.0,
        "response_time_ms": 0.0
    }
    
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        components=components,
        metrics=metrics
    )

@app.post("/process", response_model=ProcessResponse)
async def process_input(request: ProcessRequest):
    """Process text input through the NIS system"""
    start_time = time.time()
    
    try:
        if cognitive_system:
            # Use the cognitive system
            response = cognitive_system.process_input(
                text=request.text,
                generate_speech=request.generate_speech
            )
            response_text = getattr(response, 'response_text', str(response))
        else:
            # Fallback response
            response_text = f"I received your message: '{request.text}'. The cognitive system is currently initializing."
        
        processing_time = time.time() - start_time
        
        return ProcessResponse(
            response_text=response_text,
            processing_time=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        return ProcessResponse(
            response_text="I'm experiencing some technical difficulties. Please try again.",
            processing_time=time.time() - start_time,
            timestamp=time.time()
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat interface"""
    conversation_id = request.conversation_id or f"conv_{int(time.time())}"
    
    try:
        if cognitive_system:
            # Process through cognitive system
            response = cognitive_system.process_input(
                text=request.message,
                generate_speech=False
            )
            response_text = getattr(response, 'response_text', str(response))
            confidence = 0.8
        else:
            response_text = "Hello! I'm in simplified mode. How can I help you?"
            confidence = 0.5
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            conversation_id=conversation_id,
            timestamp=time.time(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response="I apologize, but I'm having some difficulties right now. Please try again.",
            user_id=request.user_id,
            conversation_id=conversation_id,
            timestamp=time.time(),
            confidence=0.1
        )

@app.get("/metrics")
async def system_metrics():
    """System metrics endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "uptime_seconds": uptime,
        "cognitive_system_status": "active" if cognitive_system else "unavailable",
        "endpoints_available": 5,
        "mode": "simplified",
        "health": "good"
    }

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 