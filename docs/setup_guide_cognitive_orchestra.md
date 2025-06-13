# NIS Protocol v2.0 Setup Guide: Cognitive Orchestra with Web Search

## Overview

This guide will help you set up the NIS Protocol v2.0 with the Cognitive Orchestra and integrated web search capabilities. The system provides advanced AI research capabilities specifically designed for archaeological and cultural heritage applications.

## Prerequisites

- Python 3.9 or higher
- Git
- Internet connection for API access
- At least 8GB RAM recommended
- 10GB free disk space

## Quick Start

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/your-org/NIS-Protocol.git
cd NIS-Protocol

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create your `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# =============================================================================
# LLM PROVIDERS - Cognitive Orchestra Configuration
# =============================================================================

# OpenAI Configuration (Primary for Creativity, Perception)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_ORGANIZATION=org-your-organization-id

# Anthropic Configuration (Primary for Consciousness, Reasoning, Cultural Intelligence)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# DeepSeek Configuration (Primary for Memory, Reasoning, Execution)
DEEPSEEK_API_KEY=your-deepseek-key-here

# Google Gemini Configuration (Web Search Integration)
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CSE_ID=your-custom-search-engine-id

# =============================================================================
# WEB SEARCH & RESEARCH CAPABILITIES
# =============================================================================

# Serper API (Alternative web search)
SERPER_API_KEY=your-serper-api-key-here

# Tavily API (Research-focused search)
TAVILY_API_KEY=tvly-your-tavily-key-here

# Bing Search API
BING_SEARCH_API_KEY=your-bing-api-key-here

# =============================================================================
# COGNITIVE ORCHESTRA SETTINGS
# =============================================================================

# Orchestra Configuration
COGNITIVE_ORCHESTRA_ENABLED=true
PARALLEL_PROCESSING_ENABLED=true
MAX_CONCURRENT_FUNCTIONS=6
HARMONY_THRESHOLD=0.7

# Function-specific providers (optional overrides)
CONSCIOUSNESS_PROVIDER=anthropic
REASONING_PROVIDER=anthropic
CREATIVITY_PROVIDER=openai
CULTURAL_PROVIDER=anthropic
ARCHAEOLOGICAL_PROVIDER=anthropic
EXECUTION_PROVIDER=bitnet

# =============================================================================
# ARCHAEOLOGICAL DOMAIN CONFIGURATION
# =============================================================================

ARCHAEOLOGICAL_DOMAIN_ENABLED=true
CULTURAL_SENSITIVITY_MODE=strict
INDIGENOUS_RIGHTS_PROTECTION=true

# =============================================================================
# MONITORING AND LOGGING
# =============================================================================

LOG_LEVEL=INFO
PERFORMANCE_TRACKING=true
COGNITIVE_METRICS_ENABLED=true
```

### 3. API Key Setup Guide

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account or sign in
3. Navigate to API Keys section
4. Create new secret key
5. Copy key to `OPENAI_API_KEY` in `.env`

#### Anthropic API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create account or sign in
3. Navigate to API Keys
4. Generate new key
5. Copy key to `ANTHROPIC_API_KEY` in `.env`

#### Google API Setup
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or select existing
3. Enable Custom Search API
4. Create API key in Credentials
5. Set up Custom Search Engine at [CSE Control Panel](https://cse.google.com/)
6. Copy API key to `GOOGLE_API_KEY` and CSE ID to `GOOGLE_CSE_ID`

#### Serper API Key
1. Visit [Serper.dev](https://serper.dev/)
2. Sign up for account
3. Get API key from dashboard
4. Copy to `SERPER_API_KEY` in `.env`

#### Tavily API Key
1. Visit [Tavily](https://tavily.com/)
2. Create account
3. Get API key from dashboard
4. Copy to `TAVILY_API_KEY` in `.env`

### 4. Test Installation

Run the demonstration scripts to verify setup:

```bash
# Test Cognitive Orchestra
python examples/cognitive_orchestra_demo.py

# Test Web Search Integration
python examples/web_search_demo.py

# Test Combined System
python examples/enhanced_llm_config_demo.py
```

## Detailed Configuration

### Cognitive Orchestra Configuration

The Cognitive Orchestra uses different LLM providers for specialized cognitive functions:

```python
# Function-Provider Mapping
CONSCIOUSNESS_PROVIDER=anthropic    # Deep self-reflection and awareness
REASONING_PROVIDER=anthropic        # Logical analysis and inference
CREATIVITY_PROVIDER=openai          # Novel idea generation
CULTURAL_PROVIDER=anthropic         # Cultural sensitivity and intelligence
ARCHAEOLOGICAL_PROVIDER=anthropic   # Domain expertise
EXECUTION_PROVIDER=bitnet           # Fast inference and action
```

### Web Search Provider Configuration

Configure multiple search providers for redundancy and comprehensive coverage:

```bash
# Primary: Google Custom Search (high quality, configurable)
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-cse-id

# Secondary: Serper (fast Google results via API)
SERPER_API_KEY=your-key

# Tertiary: Tavily (research-focused, academic sources)
TAVILY_API_KEY=your-key

# Quaternary: Bing (Microsoft search engine)
BING_SEARCH_API_KEY=your-key
```

### Domain-Specific Settings

Configure research domains for specialized functionality:

```bash
# Archaeological Research
ARCHAEOLOGICAL_DOMAIN_ENABLED=true
CULTURAL_SENSITIVITY_MODE=strict
INDIGENOUS_RIGHTS_PROTECTION=true

# Academic Source Prioritization
ACADEMIC_SOURCES_PRIORITY=high
PREFERRED_DOMAINS=jstor.org,cambridge.org,academia.edu

# Cultural Intelligence
CULTURAL_CONTEXT_AWARENESS=high
INDIGENOUS_CONSULTATION_REQUIRED=true
```

## Advanced Setup

### Infrastructure Services (Optional)

For production deployments, set up supporting infrastructure:

#### Redis (Caching)
```bash
# Install Redis
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server

# Configure in .env
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Kafka (Event Streaming)
```bash
# Download and start Kafka
wget https://downloads.apache.org/kafka/2.13-3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Configure in .env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Local LLM Setup (Optional)

For privacy-focused deployments, set up local LLM inference:

#### BitNet Setup
```bash
# Clone BitNet repository
git clone https://github.com/microsoft/BitNet.git
cd BitNet

# Follow BitNet installation instructions
# Configure path in .env
BITNET_EXECUTABLE_PATH=/path/to/bitnet
BITNET_MODEL_PATH=/path/to/model.bin
```

#### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1:70b
ollama pull codellama:34b

# Configure in .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b
```

## Usage Examples

### Basic Cognitive Orchestra Usage

```python
#!/usr/bin/env python3
import asyncio
from src.llm.cognitive_orchestra import CognitiveOrchestra, CognitiveFunction

async def main():
    # Initialize the orchestra
    orchestra = CognitiveOrchestra()
    
    # Execute a consciousness analysis
    result = await orchestra.execute_function(
        function=CognitiveFunction.CONSCIOUSNESS,
        prompt="Analyze the ethical implications of AI in archaeology",
        context={"domain": "archaeological_ethics"}
    )
    
    print(f"Consciousness Analysis: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Web Search Integration Usage

```python
#!/usr/bin/env python3
import asyncio
from src.agents.research import WebSearchAgent, ResearchDomain

async def main():
    # Initialize search agent
    search_agent = WebSearchAgent()
    
    # Conduct archaeological research
    results = await search_agent.research(
        query="recent Mayan archaeological discoveries",
        domain=ResearchDomain.ARCHAEOLOGICAL
    )
    
    print(f"Found {results['total_results']} sources")
    for result in results['top_results'][:3]:
        print(f"- {result.title}: {result.url}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Combined System Usage

```python
#!/usr/bin/env python3
import asyncio
from src.llm.cognitive_orchestra import CognitiveOrchestra, CognitiveFunction
from src.agents.research import WebSearchAgent, ResearchDomain

async def main():
    # Initialize both systems
    orchestra = CognitiveOrchestra()
    search_agent = WebSearchAgent()
    
    # Step 1: Research with web search
    search_results = await search_agent.research(
        query="drone surveys archaeological sites cultural heritage",
        domain=ResearchDomain.ARCHAEOLOGICAL
    )
    
    # Step 2: Analyze with cognitive orchestra
    analysis = await orchestra.execute_function(
        function=CognitiveFunction.ARCHAEOLOGICAL,
        prompt=f"Analyze these drone survey findings: {search_results}",
        context={"research_data": search_results}
    )
    
    # Step 3: Cultural sensitivity check
    cultural_check = await orchestra.execute_function(
        function=CognitiveFunction.CULTURAL,
        prompt="Evaluate cultural sensitivity of drone surveys at heritage sites",
        context={"analysis": analysis, "search_data": search_results}
    )
    
    print("Combined Analysis Complete:")
    print(f"Web Sources: {search_results['total_results']}")
    print(f"Archaeological Analysis: {analysis}")
    print(f"Cultural Assessment: {cultural_check}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Error: Invalid API key
# Solution: Verify API key format and permissions
export OPENAI_API_KEY="sk-your-actual-key-here"
python -c "import openai; print(openai.api_key)"
```

#### Import Errors
```bash
# Error: Module not found
# Solution: Ensure virtual environment is activated and dependencies installed
source venv/bin/activate
pip install -r requirements.txt
```

#### Network Connectivity
```bash
# Error: Connection timeout
# Solution: Check internet connection and firewall settings
curl -I https://api.openai.com/v1/models
curl -I https://api.anthropic.com/v1/messages
```

#### Rate Limiting
```bash
# Error: Rate limit exceeded
# Solution: Implement exponential backoff or upgrade API plan
# Check current usage in provider dashboards
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All system components will now provide detailed logs
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### API Response Times
```python
# Monitor API response times
import time
start_time = time.time()
# ... API call ...
print(f"Response time: {time.time() - start_time:.2f}s")
```

## Production Deployment

### Docker Setup

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  nis-protocol:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    depends_on:
      - redis
      - kafka
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
    depends_on:
      - zookeeper
  
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
```

### Monitoring Setup

Set up monitoring with Prometheus and Grafana:

```yaml
# monitoring/docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Security Best Practices

### API Key Management
1. Never commit API keys to version control
2. Use environment variables or secure key management
3. Rotate keys regularly
4. Monitor API usage for anomalies

### Network Security
1. Use HTTPS for all API communications
2. Implement rate limiting
3. Set up proper firewall rules
4. Monitor network traffic

### Data Privacy
1. Implement data retention policies
2. Anonymize sensitive research data
3. Follow GDPR and other privacy regulations
4. Secure data transmission and storage

## Support and Resources

### Documentation
- [Cognitive Orchestra Architecture](./cognitive_orchestra_architecture.md)
- [Web Search Integration](./web_search_integration.md)
- [API Reference](./api_reference.md)

### Community
- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share experiences
- Wiki: Community-contributed documentation

### Professional Support
- Enterprise support available
- Custom integration services
- Training and consultation

## Next Steps

1. **Complete Setup**: Follow this guide to set up your environment
2. **Run Examples**: Test the system with provided examples
3. **Explore Features**: Try different cognitive functions and search domains
4. **Integrate**: Incorporate into your archaeological workflows
5. **Contribute**: Share improvements and extensions with the community

The NIS Protocol v2.0 with Cognitive Orchestra represents a significant advancement in AI-powered archaeological research. With proper setup and configuration, it provides powerful tools for cultural heritage preservation and research while maintaining the highest standards of cultural sensitivity and academic rigor.