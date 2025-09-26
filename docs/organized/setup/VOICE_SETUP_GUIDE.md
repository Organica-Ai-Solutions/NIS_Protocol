# 🎙️ NIS Protocol Voice Setup Guide

**Complete setup guide for voice conversation features**

> **Updated**: 2025-01-19  
> **Version**: v3.2.1 - Voice-Enabled AI Operating System

---

## 🎯 **Overview**

This guide will help you set up the complete NIS Protocol voice conversation system, including Microsoft VibeVoice integration, wake word detection, and real-time voice streaming capabilities.

---

## 📋 **Prerequisites**

### **System Requirements**
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 5GB free space for models
- **CPU**: 4+ cores recommended for real-time processing
- **OS**: Linux, macOS, or Windows with WSL2
- **Network**: Stable internet for model downloads

### **Dependencies**
```bash
# Python 3.9+ with pip
python3 --version

# Docker and Docker Compose
docker --version
docker-compose --version

# Git
git --version
```

---

## 🚀 **Installation Steps**

### **Step 1: Clone NIS Protocol**

```bash
# Clone the repository
git clone <repository-url>
cd NIS_Protocol

# Verify structure
ls -la
```

### **Step 2: Install Voice Dependencies**

```bash
# Run the voice installation script
python scripts/install_vibevoice.py

# Expected output:
# 🎙️ Installing Microsoft VibeVoice for NIS Protocol
# ✅ Audio processing dependencies installed
# ✅ VibeVoice model configuration created
# ✅ All VibeVoice dependencies available
# 🎉 VibeVoice installation complete!
```

### **Step 3: Start the System**

```bash
# Start all services
./start.sh

# Wait for services to initialize (30-60 seconds)
# Check status
curl http://localhost:8000/health
```

### **Step 4: Verify Voice System**

```bash
# Test communication status
curl http://localhost:8000/communication/status

# Expected response:
# {
#   "status": "operational",
#   "vibevoice_available": true,
#   "capabilities": [...],
#   "streaming_features": {...}
# }
```

---

## 🔧 **Configuration**

### **VibeVoice Configuration**

Edit `configs/vibevoice_config.py`:

```python
# VibeVoice Configuration for NIS Protocol
MODEL_NAME = "microsoft/VibeVoice-1.5B"
LOCAL_MODEL_PATH = "models/vibevoice/VibeVoice-1.5B"
SAMPLE_RATE = 24000
MAX_SPEAKERS = 4
MAX_DURATION_MINUTES = 90
CHUNK_SIZE_MS = 50
STREAMING_ENABLED = True

# Speaker voice profiles - Customize as needed
SPEAKER_PROFILES = {
    "consciousness": {"voice_id": 0, "pitch": 0.8, "speed": 0.95},
    "physics": {"voice_id": 1, "pitch": 1.0, "speed": 1.0},
    "research": {"voice_id": 2, "pitch": 1.1, "speed": 1.05},
    "coordination": {"voice_id": 3, "pitch": 1.05, "speed": 1.0}
}
```

### **Performance Optimization**

Edit voice performance settings in `main.py`:

```python
# Audio buffer optimization
AUDIO_BUFFER_CONFIG = {
    "target_latency_ms": 200,    # Lower for faster response
    "max_latency_ms": 500,       # Adjust based on hardware
    "chunk_size_ms": 20,         # 20ms for low latency
    "adaptive_quality": True     # Enable adaptive quality
}

# STT optimization  
STT_CONFIG = {
    "model_size": "base",        # Use "small" for faster processing
    "device": "cpu",             # Use "cuda" if GPU available
    "compute_type": "int8"       # Quantization for speed
}
```

---

## 🧪 **Testing**

### **Test 1: Basic Speech Synthesis**

```bash
# Test consciousness agent voice
curl -X POST http://localhost:8000/communication/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is the NIS Protocol consciousness agent speaking.",
    "speaker": "consciousness",
    "emotion": "thoughtful"
  }' \
  --output test_consciousness.wav

# Play the audio file
# macOS: open test_consciousness.wav
# Linux: aplay test_consciousness.wav
# Windows: start test_consciousness.wav
```

### **Test 2: Multi-Agent Dialogue**

```bash
# Create a conversation between agents
curl -X POST http://localhost:8000/communication/agent_dialogue \
  -H "Content-Type: application/json" \
  -d '{
    "agents_content": {
      "consciousness": "System awareness is at 94.2 percent",
      "physics": "Energy conservation laws are validated",
      "research": "Analysis of 15 research papers complete",
      "coordination": "All agents are synchronized and operational"
    },
    "dialogue_style": "conversation"
  }' | jq '.'

# Expected: Multi-speaker dialogue with seamless voice transitions
```

### **Test 3: Real-Time Voice Chat**

Open the web interface and test voice features:

```bash
# Open browser to NIS Protocol interface
open http://localhost:8000

# In the chat interface, type: /voice
# This should activate voice conversation mode
```

### **Test 4: WebSocket Streaming**

Test real-time streaming with a simple script:

```javascript
// Save as test_voice_stream.html and open in browser
const ws = new WebSocket('ws://localhost:8000/communication/stream');

ws.onopen = () => {
  console.log('Voice streaming connected');
  
  // Start conversation
  ws.send(JSON.stringify({
    "type": "start_conversation",
    "agents_content": {
      "consciousness": "Testing real-time voice streaming",
      "physics": "Physics validation systems online"
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data.type);
  
  if (data.type === "audio_chunk") {
    // Real-time audio chunk received
    console.log(`Audio chunk from ${data.speaker}: ${data.text_chunk}`);
  }
};
```

---

## 🔧 **Troubleshooting**

### **Issue: Audio Dependencies Missing**

```bash
# Error: ModuleNotFoundError: No module named 'soundfile'
# Solution: Install audio dependencies
pip install soundfile librosa resampy
pip install diffusers transformers[torch] accelerate
```

### **Issue: VibeVoice Model Not Found**

```bash
# Error: Model not found at models/vibevoice/VibeVoice-1.5B
# Solution: Download model manually
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/VibeVoice-1.5B',
    local_dir='models/vibevoice/VibeVoice-1.5B',
    local_dir_use_symlinks=False
)
"
```

### **Issue: High Latency**

```bash
# Problem: Voice responses take >1 second
# Solutions:
# 1. Reduce audio buffer size
# 2. Use smaller Whisper model
# 3. Enable GPU acceleration (if available)
# 4. Optimize chunk sizes

# Check current performance
curl http://localhost:8000/communication/status | jq '.streaming_features'
```

### **Issue: Wake Word Not Detecting**

```bash
# Test wake word detection directly
python -c "
from src.services.wake_word_service import get_wake_word_detector
detector = get_wake_word_detector()
result = detector.detect_wake_word('hey nis how are you today')
print('Detection result:', result)
"

# Expected: {"detected": True, "phrase": "hey nis", ...}
```

### **Issue: WebSocket Connection Fails**

```bash
# Check WebSocket endpoints
curl -I http://localhost:8000/voice-chat
curl -I http://localhost:8000/communication/stream

# Check nginx configuration
docker-compose logs nginx

# Restart services if needed
./stop.sh && ./start.sh
```

---

## 🎛️ **Advanced Configuration**

### **Custom Voice Profiles**

Create custom speaker characteristics:

```python
# Add to configs/vibevoice_config.py
CUSTOM_SPEAKER_PROFILES = {
    "assistant": {
        "voice_id": 4,
        "pitch": 0.9,
        "speed": 1.0,
        "emotion_default": "friendly",
        "description": "General purpose assistant voice"
    },
    "narrator": {
        "voice_id": 5,
        "pitch": 0.7,
        "speed": 0.9,
        "emotion_default": "calm",
        "description": "Documentary-style narrator voice"
    }
}
```

### **Performance Tuning**

```python
# High-performance configuration for production
PRODUCTION_CONFIG = {
    "audio_buffer": {
        "target_latency_ms": 100,     # Aggressive latency target
        "chunk_size_ms": 10,          # Smaller chunks
        "concurrent_sessions": 16,    # More concurrent users
        "gpu_acceleration": True      # Enable GPU if available
    },
    "model_optimization": {
        "quantization": "int8",       # Model quantization
        "batch_processing": True,     # Batch similar requests
        "model_caching": True,        # Cache loaded models
        "preload_speakers": True      # Preload speaker embeddings
    }
}
```

### **Integration with External Services**

```python
# Connect voice system to external services
EXTERNAL_INTEGRATIONS = {
    "twilio": {
        "enabled": False,
        "phone_integration": True,
        "sms_responses": True
    },
    "discord": {
        "enabled": False,
        "bot_integration": True,
        "voice_channels": True
    },
    "teams": {
        "enabled": False,
        "meeting_integration": True,
        "real_time_transcription": True
    }
}
```

---

## 🔒 **Security Configuration**

### **Voice Data Protection**

```python
# Security settings for voice data
VOICE_SECURITY = {
    "data_retention": {
        "audio_files": "none",          # Don't store audio
        "transcriptions": "session",    # Only during session
        "voice_prints": "disabled"      # No biometric storage
    },
    "encryption": {
        "websocket_tls": True,
        "audio_encryption": "AES-256",
        "key_rotation": "24h"
    },
    "access_control": {
        "authentication_required": True,
        "rate_limiting": True,
        "session_timeout": "30m"
    }
}
```

### **Privacy Settings**

```python
# Privacy protection for voice interactions
PRIVACY_CONFIG = {
    "anonymization": {
        "remove_identifiers": True,
        "voice_fingerprinting": False,
        "location_tracking": False
    },
    "compliance": {
        "gdpr_compliant": True,
        "hipaa_ready": True,
        "audit_logging": True
    }
}
```

---

## 📊 **Monitoring & Analytics**

### **Performance Monitoring**

```bash
# Monitor voice system performance
curl http://localhost:8000/communication/status | jq '.streaming_features'

# Check audio buffer status
# (Available through WebSocket performance_stats messages)

# Monitor resource usage
docker stats nis_protocol_backend
```

### **Quality Metrics**

```python
# Monitor voice quality metrics
QUALITY_MONITORING = {
    "audio_quality": {
        "bit_rate": "monitor",
        "sample_rate": "monitor", 
        "latency": "track",
        "dropouts": "alert"
    },
    "speech_quality": {
        "intelligibility": "measure",
        "naturalness": "track",
        "emotion_accuracy": "validate"
    },
    "system_performance": {
        "cpu_usage": "monitor",
        "memory_usage": "track",
        "network_bandwidth": "measure"
    }
}
```

---

## 🚀 **Production Deployment**

### **Docker Production Configuration**

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  nis-voice-backend:
    image: nis-protocol:voice-prod
    environment:
      - NODE_ENV=production
      - VOICE_OPTIMIZATION=aggressive
      - AUDIO_QUALITY=high
      - CONCURRENT_SESSIONS=32
    volumes:
      - voice-models:/app/models/vibevoice:ro
      - voice-cache:/app/cache/audio
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/communication/status"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### **Load Balancing**

```nginx
# nginx configuration for voice load balancing
upstream voice_backend {
    least_conn;
    server nis-voice-1:8000 max_fails=3 fail_timeout=30s;
    server nis-voice-2:8000 max_fails=3 fail_timeout=30s;
    server nis-voice-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location /communication/ {
        proxy_pass http://voice_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /voice-chat {
        proxy_pass http://voice_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## ✅ **Verification Checklist**

### **Installation Verification**

- [ ] NIS Protocol backend running on port 8000
- [ ] VibeVoice dependencies installed successfully
- [ ] Voice model downloaded and accessible
- [ ] Communication status endpoint returns "operational"
- [ ] Basic speech synthesis test produces audio
- [ ] Multi-agent dialogue test works correctly

### **WebSocket Verification**

- [ ] `/voice-chat` WebSocket connection establishes
- [ ] `/communication/stream` WebSocket connection establishes
- [ ] Audio data can be sent and received
- [ ] Real-time latency is <500ms
- [ ] Wake word detection responds correctly

### **Integration Verification**

- [ ] Voice commands route to correct NIS agents
- [ ] Agent responses have distinct voice characteristics
- [ ] Consciousness vocalization includes system metrics
- [ ] Physics explanations are technically accurate
- [ ] Research findings are properly narrated

### **Performance Verification**

- [ ] End-to-end latency <500ms (target: 350ms)
- [ ] Audio quality is clear and intelligible
- [ ] No audio dropouts or artifacts
- [ ] System handles concurrent voice sessions
- [ ] Resource usage within acceptable limits

---

## 🔄 **Next Steps**

### **Basic Usage**
1. Test voice synthesis with different agents
2. Try multi-agent conversations
3. Experiment with real-time voice chat
4. Explore wake word detection

### **Advanced Features**
1. Customize voice profiles for your use case
2. Integrate with external communication systems
3. Implement custom voice commands
4. Set up production monitoring

### **Development**
1. Build custom voice-enabled applications
2. Extend agent voice capabilities
3. Create domain-specific voice interactions
4. Contribute to voice system improvements

---

**🎙️ Your NIS Protocol voice conversation system is now ready for enterprise-grade voice interactions with multi-agent intelligence!**
