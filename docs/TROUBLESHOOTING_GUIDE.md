# 🔧 NIS Protocol v3 Troubleshooting Guide

## 🔑 **API Key Issues (Most Common)**

### **❌ "No API key provided" or LLM connection errors**

**🔍 Symptoms:**
- System starts but fails on `/process` requests
- Error messages about missing API keys
- LLM provider authentication failures

**✅ Solution:**
```bash
# 1. Check if .env file exists
ls -la .env

# 2. Verify API keys are set
cat .env | grep API_KEY

# 3. Create/fix .env file
cat > .env << EOF
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
DEEPSEEK_API_KEY=your_actual_deepseek_key_here
GOOGLE_API_KEY=your_actual_google_key_here
EOF

# 4. Restart the system
./stop.sh
./start.sh
```

**🔗 Get API keys from:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **DeepSeek**: https://platform.deepseek.com/
- **Google**: https://makersuite.google.com/app/apikey

### **❌ "Invalid API key" errors**

**🔍 Check API key validity:**
```bash
# Test OpenAI key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Anthropic key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json"
```

**✅ Solutions:**
1. **Regenerate keys** from provider website
2. **Check key format** (no extra spaces/characters)
3. **Verify billing** is set up with provider
4. **Check rate limits** - you might be hitting usage limits

---

## 🎯 **Quick Diagnostics**

### **System Health Check**
```bash
# Run comprehensive system check
python utilities/final_100_test.py

# Check individual components
python -c "from src.cognitive_agents.cognitive_system import CognitiveSystem; print('✅ Cognitive System OK')"
python -c "from src.agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent; print('✅ Consciousness Agent OK')"
python -c "from src.infrastructure.integration_coordinator import InfrastructureCoordinator; print('✅ Infrastructure OK')"
```

### **Quick Status Dashboard**
```python
# diagnostic_dashboard.py - Run this for instant system overview
from src.cognitive_agents.cognitive_system import CognitiveSystem
from src.agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
import datetime

def quick_diagnosis():
    print("🩺 NIS Protocol Quick Diagnosis")
    print("=" * 50)
    print(f"📅 Timestamp: {datetime.datetime.now()}")
    
    try:
        # Test cognitive system
        cognitive_system = CognitiveSystem()
        response = cognitive_system.process_input("Test input", generate_speech=False)
        print("✅ Cognitive System: HEALTHY")
        print(f"   └─ Response Confidence: {response.confidence:.2f}")
    except Exception as e:
        print(f"❌ Cognitive System: ERROR - {e}")
    
    try:
        # Test consciousness agent
        consciousness = EnhancedConsciousAgent()
        state = consciousness.get_current_state()
        print("✅ Consciousness Agent: HEALTHY")
        print(f"   └─ Awareness Level: {state.awareness_level}")
    except Exception as e:
        print(f"❌ Consciousness Agent: ERROR - {e}")
    
    print("\n🔍 Run detailed diagnostics with specific error codes below")

if __name__ == "__main__":
    quick_diagnosis()
```

## 🚨 **Common Issues & Solutions**

### **🧠 Cognitive System Issues**

#### **Error: `ModuleNotFoundError: No module named 'src'`**
**Symptoms:**
- Import errors when running examples
- Python can't find NIS Protocol modules

**Solution:**
```bash
# Add to Python path (temporary)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to your script
import sys
sys.path.append('.')

# Permanent solution: Install in development mode
pip install -e .
```

#### **Error: `CognitiveSystem initialization failed`**
**Symptoms:**
- Cognitive system won't start
- Missing dependencies or configuration

**Diagnostic:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.cognitive_agents.cognitive_system import CognitiveSystem
# This will show detailed error messages
```

**Solutions:**
```bash
# Check all dependencies
pip install -r requirements.txt

# Install missing deep learning dependencies
pip install torch torchvision transformers

# Check for configuration issues
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

#### **Error: `Low confidence responses (< 0.5)`**
**Symptoms:**
- System responses have very low confidence
- Inconsistent or poor quality outputs

**Diagnostic:**
```python
from src.agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent

consciousness = EnhancedConsciousAgent()
state = consciousness.get_current_state()
print(f"System confidence: {state.confidence}")
print(f"Active agents: {state.active_agents}")
```

**Solutions:**
```python
# Enable ensemble mode for higher confidence
cognitive_system = CognitiveSystem()
response = cognitive_system.process_input(
    "Your question",
    context={"ensemble_mode": True, "confidence_threshold": 0.7}
)

# Check for model loading issues
print("Checking model availability...")
# If models aren't loading properly, confidence will be low
```

### **🤖 Agent Communication Issues**

#### **Error: `Agent timeout or no response`**
**Symptoms:**
- Agents not responding
- Long processing times
- Stuck operations

**Diagnostic:**
```python
import asyncio
from src.core.registry import Registry

async def check_agent_health():
    registry = Registry()
    agents = registry.get_all_agents()
    
    for agent_name, agent in agents.items():
        try:
            # Simple health check
            response = await agent.process({"test": "ping"})
            print(f"✅ {agent_name}: RESPONSIVE")
        except Exception as e:
            print(f"❌ {agent_name}: ERROR - {e}")

# Run the check
asyncio.run(check_agent_health())
```

**Solutions:**
```python
# Restart stuck agents
from src.core.registry import Registry

registry = Registry()
registry.restart_agent("stuck_agent_name")

# Increase timeout values
import os
os.environ["NIS_AGENT_TIMEOUT"] = "60"  # 60 seconds

# Check resource usage
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

### **💾 Memory System Issues**

#### **Error: `Redis connection failed`**
**Symptoms:**
- Memory operations failing
- Cache miss errors
- "Connection refused" messages

**Diagnostic:**
```bash
# Check if Redis is running
redis-cli ping
# Should return "PONG"

# Check Redis status
systemctl status redis-server  # Linux
brew services list | grep redis  # macOS
```

**Solutions:**
```bash
# Start Redis
sudo systemctl start redis-server  # Linux
brew services start redis  # macOS

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine

# Test connection
python -c "import redis; r = redis.Redis(); print(r.ping())"
```

#### **Error: `Vector store corruption`**
**Symptoms:**
- Memory retrieval failures
- Inconsistent search results
- Index errors

**Diagnostic:**
```python
from src.memory.vector_store import VectorStore

try:
    vector_store = VectorStore()
    test_result = vector_store.similarity_search("test query", k=1)
    print(f"✅ Vector store OK: {len(test_result)} results")
except Exception as e:
    print(f"❌ Vector store error: {e}")
```

**Solutions:**
```python
# Rebuild vector store
from src.memory.vector_store import VectorStore

vector_store = VectorStore()
vector_store.rebuild_index()  # This may take time

# Clear corrupted cache
vector_store.clear_cache()

# Reset to backup
vector_store.restore_from_backup()
```

### **🔗 LLM Provider Issues**

#### **Error: `OpenAI API rate limit exceeded`**
**Symptoms:**
- "Rate limit exceeded" errors
- 429 HTTP status codes
- Slow or failed responses

**Diagnostic:**
```python
import openai
from datetime import datetime

try:
    # Test API connection
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("✅ OpenAI API: WORKING")
except openai.error.RateLimitError as e:
    print(f"❌ Rate limit exceeded: {e}")
except openai.error.APIError as e:
    print(f"❌ API error: {e}")
```

**Solutions:**
```python
# Enable automatic retries with backoff
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_openai_with_retry(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Use multiple providers
from src.llm.llm_manager import LLMManager

llm_manager = LLMManager()
llm_manager.enable_fallback_providers(["anthropic", "google", "local"])
```

#### **Error: `Anthropic API authentication failed`**
**Symptoms:**
- 401 Unauthorized errors
- Authentication failures
- Invalid API key messages

**Solutions:**
```bash
# Check API key format
echo $ANTHROPIC_API_KEY | cut -c1-10
# Should start with "sk-ant-"

# Set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Or create .env file
echo "ANTHROPIC_API_KEY=your-api-key-here" >> .env
```

### **⚡ Performance Issues**

#### **Issue: Slow response times (> 5 seconds)**
**Diagnostic:**
```python
import time
from src.cognitive_agents.cognitive_system import CognitiveSystem

def benchmark_performance():
    cognitive_system = CognitiveSystem()
    
    test_queries = [
        "What is 2+2?",
        "Explain quantum computing",
        "Analyze this data pattern"
    ]
    
    for query in test_queries:
        start_time = time.time()
        response = cognitive_system.process_input(query)
        end_time = time.time()
        
        print(f"Query: {query[:30]}...")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Confidence: {response.confidence:.2f}")
        print("-" * 40)

benchmark_performance()
```

**Solutions:**
```python
# Enable caching
from src.infrastructure.caching_system import CachingSystem

caching = CachingSystem()
caching.enable_response_cache(ttl=3600)  # 1 hour cache

# Optimize model loading
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("🚀 CUDA optimization enabled")

# Use faster models for simple queries
cognitive_system = CognitiveSystem()
cognitive_system.configure_performance_mode("fast")  # vs "quality"
```

#### **Issue: High memory usage**
**Diagnostic:**
```python
import psutil
import gc
from src.memory.memory_manager import MemoryManager

def check_memory_usage():
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent}% used")
    print(f"Available: {memory.available / 1024**3:.1f} GB")
    
    # Python memory
    import sys
    objects_count = len(gc.get_objects())
    print(f"Python objects: {objects_count:,}")
    
    # NIS memory
    memory_manager = MemoryManager()
    nis_memory = memory_manager.get_memory_stats()
    print(f"NIS Cache size: {nis_memory.get('cache_size', 'Unknown')}")

check_memory_usage()
```

**Solutions:**
```python
# Clean up memory
import gc
from src.memory.memory_manager import MemoryManager

# Force garbage collection
gc.collect()

# Clear NIS caches
memory_manager = MemoryManager()
memory_manager.cleanup_expired_cache()
memory_manager.optimize_memory_usage()

# Reduce model size
import torch
torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
```

## 🔍 **Advanced Diagnostics**

### **Network Connectivity Issues**
```bash
# Test external API connectivity
curl -I https://api.openai.com/v1/models
curl -I https://api.anthropic.com/v1/complete

# Test internal services
curl -I http://localhost:6379  # Redis
curl -I http://localhost:9092  # Kafka (if using)

# DNS resolution test
nslookup api.openai.com
nslookup api.anthropic.com
```

### **Configuration Validation**
```python
# config_validator.py
import os
import json
from pathlib import Path

def validate_configuration():
    """Comprehensive configuration validation"""
    
    print("🔧 Configuration Validation")
    print("=" * 40)
    
    # Check environment variables
    required_env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY"
    ]
    
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: SET ({value[:10]}...)")
        else:
            print(f"❌ {var}: NOT SET")
    
    # Check configuration files
    config_files = [
        "config/llm_config.json",
        "config/protocol_config.json"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                print(f"✅ {config_file}: VALID")
            except json.JSONDecodeError as e:
                print(f"❌ {config_file}: INVALID JSON - {e}")
        else:
            print(f"⚠️ {config_file}: NOT FOUND")
    
    # Check model files
    model_paths = [
        "models/",
        "src/agents/"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            file_count = len(list(Path(path).rglob("*.py")))
            print(f"✅ {path}: {file_count} files")
        else:
            print(f"❌ {path}: NOT FOUND")

validate_configuration()
```

### **Performance Profiling**
```python
# performance_profiler.py
import cProfile
import pstats
from src.cognitive_agents.cognitive_system import CognitiveSystem

def profile_nis_performance():
    """Profile NIS Protocol performance"""
    
    def test_function():
        cognitive_system = CognitiveSystem()
        
        # Run multiple test cases
        test_cases = [
            "Simple math: 2+2",
            "Scientific question: How does photosynthesis work?",
            "Complex analysis: Analyze market trends in renewable energy"
        ]
        
        for case in test_cases:
            response = cognitive_system.process_input(case)
            print(f"Processed: {case[:30]}... (confidence: {response.confidence:.2f})")
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    test_function()
    
    profiler.disable()
    
    # Save and display results
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.print_stats(20)  # Top 20 slowest functions
    
    # Save to file for analysis
    stats.dump_stats('nis_performance_profile.prof')
    print("\n📊 Profile saved to: nis_performance_profile.prof")
    print("View with: python -m pstats nis_performance_profile.prof")

profile_nis_performance()
```

## 📋 **Recovery Procedures**

### **Complete System Reset**
```bash
#!/bin/bash
# complete_reset.sh - Full NIS Protocol reset

echo "🔄 Starting complete NIS Protocol reset..."

# Stop all services
echo "1. Stopping services..."
pkill -f "python.*nis"
docker stop $(docker ps -q --filter "name=nis")

# Clear caches
echo "2. Clearing caches..."
redis-cli FLUSHALL
rm -rf __pycache__/
find . -name "*.pyc" -delete

# Reset configurations
echo "3. Resetting configurations..."
cp config/default_config.json config/protocol_config.json

# Reinstall dependencies
echo "4. Reinstalling dependencies..."
pip install -r requirements.txt --force-reinstall

# Rebuild indexes
echo "5. Rebuilding indexes..."
python -c "from src.memory.vector_store import VectorStore; VectorStore().rebuild_index()"

# Run health check
echo "6. Running health check..."
python utilities/final_100_test.py

echo "✅ Reset complete!"
```

### **Backup and Restore**
```python
# backup_restore.py
import shutil
import datetime
from pathlib import Path

def create_backup():
    """Create complete system backup"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/nis_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup configurations
    shutil.copytree("config/", backup_dir / "config")
    
    # Backup models (if any custom models)
    if Path("models/").exists():
        shutil.copytree("models/", backup_dir / "models")
    
    # Backup memory data
    if Path("memory_data/").exists():
        shutil.copytree("memory_data/", backup_dir / "memory_data")
    
    print(f"✅ Backup created: {backup_dir}")
    return backup_dir

def restore_backup(backup_path):
    """Restore from backup"""
    
    backup_dir = Path(backup_path)
    if not backup_dir.exists():
        print(f"❌ Backup not found: {backup_path}")
        return False
    
    # Restore configurations
    if (backup_dir / "config").exists():
        shutil.rmtree("config/")
        shutil.copytree(backup_dir / "config", "config/")
    
    # Restore models
    if (backup_dir / "models").exists():
        shutil.rmtree("models/")
        shutil.copytree(backup_dir / "models", "models/")
    
    # Restore memory data
    if (backup_dir / "memory_data").exists():
        shutil.rmtree("memory_data/")
        shutil.copytree(backup_dir / "memory_data", "memory_data/")
    
    print(f"✅ Restored from: {backup_path}")
    return True

# Usage
if __name__ == "__main__":
    # Create backup
    backup_path = create_backup()
    
    # To restore later:
    # restore_backup("backups/nis_backup_20240119_143022")
```

## 🆘 **Emergency Contacts & Resources**

### **Getting Help**
- 📚 **Documentation**: [docs/README.md](docs/README.md)
- 🐛 **Issue Tracker**: GitHub Issues
- 💬 **Community**: Check project README for community links
- 📧 **Support**: Check project documentation for support channels

### **Escalation Matrix**
1. **🟢 Low Priority**: Self-service using this guide
2. **🟡 Medium Priority**: Check GitHub Issues and documentation
3. **🔴 High Priority**: Create detailed issue with diagnostic output
4. **🚨 Critical Priority**: Emergency contact (if available in project)

### **Diagnostic Information to Include**
When reporting issues, include:
```bash
# System information
python --version
pip freeze | grep -E "(torch|transformers|anthropic|openai)"
uname -a  # Linux/macOS
systeminfo  # Windows

# NIS Protocol diagnostics
python utilities/final_100_test.py
python -c "from diagnostic_dashboard import quick_diagnosis; quick_diagnosis()"

# Logs (last 50 lines)
tail -50 nis_protocol.log
```

This troubleshooting guide covers:
- ✅ **Quick Diagnostics**: Instant system health checks
- ✅ **Common Issues**: Real-world problems and solutions
- ✅ **Performance Optimization**: Speed and memory improvements
- ✅ **Recovery Procedures**: Complete system reset and backup/restore
- ✅ **Advanced Diagnostics**: Deep system analysis tools

Perfect for production deployment and ensuring smooth operation in your AWS MAP program! 