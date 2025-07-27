"""
NIS Protocol v3 - Step 1: Add Cognitive System
Building on the working ultra-simple baseline
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nis_step1")

# Try to import cognitive system - graceful fallback if it fails
try:
    from src.utils.env_config import EnvironmentConfig
    from src.cognitive_agents.cognitive_system import CognitiveSystem
    COGNITIVE_AVAILABLE = True
    logger.info("✅ Cognitive system imports successful")
except Exception as e:
    logger.warning(f"⚠️ Cognitive system imports failed: {e}")
    COGNITIVE_AVAILABLE = False

# Global state
startup_time = None
cognitive_system = None

class HealthResponse(BaseModel):
    status: str
    uptime: float
    components: Dict[str, str]
    message: str

class ProcessRequest(BaseModel):
    text: str
    generate_speech: bool = False

class ProcessResponse(BaseModel):
    response_text: str
    processing_time: float
    timestamp: float

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: float
    confidence: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with cognitive system initialization"""
    global startup_time, cognitive_system
    
    logger.info("🚀 Starting NIS Protocol v3 - Step 1 (Cognitive System)")
    startup_time = time.time()
    
    # Initialize cognitive system if available
    if COGNITIVE_AVAILABLE:
        try:
            env_config = EnvironmentConfig()
            logger.info("✅ Environment configuration loaded")
            
            cognitive_system = CognitiveSystem()
            logger.info("✅ Cognitive system initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Cognitive system initialization failed: {e}")
            logger.info("🔄 Continuing without cognitive system")
    
    app.state.startup_time = startup_time
    app.state.cognitive_system = cognitive_system
    
    logger.info("🎉 NIS Protocol v3 Step 1 ready!")
    yield
    
    logger.info("🛑 Shutting down NIS Protocol v3 Step 1")

# Create FastAPI application
app = FastAPI(
    title="NIS Protocol v3 - Step 1",
    description="Step 1: Adding cognitive system to working baseline",
    version="3.0.0-step1",
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

@app.get("/")
async def root():
    """Root endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    return {
        "system": "NIS Protocol v3",
        "version": "3.0.0-step1", 
        "status": "operational",
        "uptime_seconds": uptime,
        "cognitive_system": "available" if cognitive_system else "unavailable",
        "message": "Step 1: Cognitive system integration",
        "endpoints": ["/", "/health", "/process", "/chat", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    
    components = {
        "api": "healthy",
        "cognitive_system": "active" if cognitive_system else "unavailable"
    }
    
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        components=components,
        message="Step 1 system healthy!"
    )

@app.post("/process", response_model=ProcessResponse)
async def process_input(request: ProcessRequest):
    """Process text input through cognitive system"""
    start_time = time.time()
    
    try:
        if cognitive_system:
            # Use actual cognitive system
            response = cognitive_system.process_input(
                text=request.text,
                generate_speech=request.generate_speech
            )
            response_text = getattr(response, 'response_text', str(response))
        else:
            # Fallback response
            response_text = f"[Step 1 Fallback] I received: '{request.text}'. Cognitive system is initializing..."
        
        processing_time = time.time() - start_time
        
        return ProcessResponse(
            response_text=response_text,
            processing_time=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return ProcessResponse(
            response_text="I'm experiencing technical difficulties. Please try again.",
            processing_time=time.time() - start_time,
            timestamp=time.time()
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat with cognitive system"""
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
            response_text = f"Hello {request.user_id}! I'm running in Step 1 mode. You said: '{request.message}'. My cognitive system is still initializing."
            confidence = 0.5
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            timestamp=time.time(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response="I'm having some difficulties. Please try again.",
            user_id=request.user_id,
            timestamp=time.time(),
            confidence=0.1
        )

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "test": "success",
        "step": 1,
        "cognitive_system": "available" if cognitive_system else "unavailable",
        "timestamp": time.time(),
        "message": "Step 1 NIS Protocol v3 working!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 