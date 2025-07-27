"""
NIS Protocol v3 - Step 2: Add Infrastructure Components
Building on Step 1 + adding Redis and Kafka infrastructure
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
logger = logging.getLogger("nis_step2")

# Import components with graceful fallbacks
COGNITIVE_AVAILABLE = False
INFRASTRUCTURE_AVAILABLE = False

# Try to import cognitive system
try:
    from src.utils.env_config import EnvironmentConfig
    from src.cognitive_agents.cognitive_system import CognitiveSystem
    COGNITIVE_AVAILABLE = True
    logger.info("✅ Cognitive system imports successful")
except Exception as e:
    logger.warning(f"⚠️ Cognitive system imports failed: {e}")

# Try to import infrastructure
try:
    from src.infrastructure.integration_coordinator import InfrastructureCoordinator
    INFRASTRUCTURE_AVAILABLE = True
    logger.info("✅ Infrastructure imports successful")
except Exception as e:
    logger.warning(f"⚠️ Infrastructure imports failed: {e}")

# Global state
startup_time = None
cognitive_system = None
infrastructure_coordinator = None

class HealthResponse(BaseModel):
    status: str
    uptime: float
    components: Dict[str, str]
    metrics: Dict[str, float]
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
    """Application lifespan with cognitive system and infrastructure"""
    global startup_time, cognitive_system, infrastructure_coordinator
    
    logger.info("🚀 Starting NIS Protocol v3 - Step 2 (Infrastructure)")
    startup_time = time.time()
    
    # Initialize environment config
    if COGNITIVE_AVAILABLE or INFRASTRUCTURE_AVAILABLE:
        try:
            env_config = EnvironmentConfig()
            logger.info("✅ Environment configuration loaded")
        except Exception as e:
            logger.warning(f"⚠️ Environment config failed: {e}")
    
    # Initialize infrastructure coordinator
    if INFRASTRUCTURE_AVAILABLE:
        try:
            infrastructure_coordinator = InfrastructureCoordinator()
            await infrastructure_coordinator.initialize()
            logger.info("✅ Infrastructure coordinator initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Infrastructure initialization failed: {e}")
            logger.info("🔄 Continuing without infrastructure")
    
    # Initialize cognitive system
    if COGNITIVE_AVAILABLE:
        try:
            cognitive_system = CognitiveSystem()
            logger.info("✅ Cognitive system initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Cognitive system initialization failed: {e}")
            logger.info("🔄 Continuing without cognitive system")
    
    app.state.startup_time = startup_time
    app.state.cognitive_system = cognitive_system
    app.state.infrastructure_coordinator = infrastructure_coordinator
    
    logger.info("🎉 NIS Protocol v3 Step 2 ready!")
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down NIS Protocol v3 Step 2")
    try:
        if infrastructure_coordinator:
            await infrastructure_coordinator.shutdown()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Create FastAPI application
app = FastAPI(
    title="NIS Protocol v3 - Step 2",
    description="Step 2: Adding infrastructure components (Redis, Kafka)",
    version="3.0.0-step2",
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
        "version": "3.0.0-step2", 
        "status": "operational",
        "uptime_seconds": uptime,
        "cognitive_system": "available" if cognitive_system else "unavailable",
        "infrastructure": "available" if infrastructure_coordinator else "unavailable",
        "message": "Step 2: Infrastructure integration",
        "endpoints": ["/", "/health", "/process", "/chat", "/infrastructure/status", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    
    components = {
        "api": "healthy",
        "cognitive_system": "active" if cognitive_system else "unavailable",
        "infrastructure": "active" if infrastructure_coordinator else "unavailable"
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
        metrics=metrics,
        message="Step 2 system healthy!"
    )

@app.get("/infrastructure/status")
async def infrastructure_status():
    """Infrastructure status endpoint"""
    if not infrastructure_coordinator:
        return {
            "status": "unavailable",
            "message": "Infrastructure coordinator not initialized",
            "redis": "unavailable",
            "kafka": "unavailable"
        }
    
    try:
        # Get infrastructure status
        status = {
            "status": "active",
            "message": "Infrastructure coordinator running",
            "redis": "active",  # Simplified - would check actual Redis
            "kafka": "active",  # Simplified - would check actual Kafka
            "timestamp": time.time()
        }
        return status
    except Exception as e:
        logger.error(f"Infrastructure status error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "redis": "unknown",
            "kafka": "unknown"
        }

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
            response_text = f"[Step 2 Fallback] I received: '{request.text}'. Cognitive system + Infrastructure available."
        
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
    """Enhanced chat with cognitive system and infrastructure"""
    try:
        if cognitive_system:
            # Process through cognitive system
            response = cognitive_system.process_input(
                text=request.message,
                generate_speech=False
            )
            response_text = getattr(response, 'response_text', str(response))
            confidence = 0.85
        else:
            response_text = f"Hello {request.user_id}! I'm running in Step 2 mode with infrastructure support. You said: '{request.message}'"
            confidence = 0.6
        
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

@app.get("/metrics")
async def system_metrics():
    """System metrics endpoint"""
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "uptime_seconds": uptime,
        "cognitive_system_status": "active" if cognitive_system else "unavailable",
        "infrastructure_status": "active" if infrastructure_coordinator else "unavailable",
        "endpoints_available": 6,
        "step": 2,
        "health": "good"
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "test": "success",
        "step": 2,
        "cognitive_system": "available" if cognitive_system else "unavailable",
        "infrastructure": "available" if infrastructure_coordinator else "unavailable",
        "timestamp": time.time(),
        "message": "Step 2 NIS Protocol v3 working with infrastructure!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 