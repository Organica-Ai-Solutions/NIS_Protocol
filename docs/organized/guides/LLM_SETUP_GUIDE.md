# LLM Provider Setup Guide - NIS Protocol v3

## Overview

The NIS Protocol v3 uses multiple LLM providers through an environment-based configuration system. This guide will help you set up API keys and configure the system for optimal performance.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit your `.env` file** with your actual API keys
3. **Test the configuration:**
   ```bash
   python test_env_simple.py
   ```

## Supported LLM Providers

### OpenAI (GPT-4, GPT-4o)
- **recommended for:** General reasoning, creativity, perception
- **API Key:** Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Environment Variables:**
  ```bash
  OPENAI_API_KEY=sk-your-key-here
  OPENAI_ENABLED=true
  OPENAI_ORGANIZATION=your-org-id  # Optional
  ```

### Anthropic (Claude)
- **recommended for:** Consciousness, reasoning, cultural intelligence, archaeological analysis
- **API Key:** Get from [Anthropic Console](https://console.anthropic.com/)
- **Environment Variables:**
  ```bash
  ANTHROPIC_API_KEY=sk-ant-your-key-here
  ANTHROPIC_ENABLED=true
  ```

### DeepSeek
- **recommended for:** Memory, execution, cost-effective reasoning
- **API Key:** Get from [DeepSeek Platform](https://platform.deepseek.com/)
- **Environment Variables:**
  ```bash
  DEEPSEEK_API_KEY=sk-your-key-here
  DEEPSEEK_ENABLED=true
  ```

### BitNet (Local)
- **recommended for:** Execution, local deployment, privacy
- **Setup:** Requires local model installation
- **Environment Variables:**
  ```bash
  BITNET_ENABLED=true
  BITNET_MODEL_PATH=./models/bitnet/model.bin
  BITNET_EXECUTABLE_PATH=bitnet
  ```

## Configuration Sections

### Basic Configuration

```bash
# Default provider (openai, anthropic, deepseek, bitnet, mock)
DEFAULT_LLM_PROVIDER=anthropic

# Fallback to mock if no providers configured
FALLBACK_TO_MOCK=true
```

### Cognitive Function Mapping

The system systematically assigns the recommended provider for each cognitive function:

```bash
# Primary providers for different cognitive functions
CONSCIOUSNESS_PROVIDER=anthropic     # Meta-cognition, self-reflection
REASONING_PROVIDER=anthropic         # Logical analysis, problem solving
CREATIVITY_PROVIDER=openai           # Creative ideas, creative thinking
CULTURAL_PROVIDER=anthropic          # Cultural intelligence, sensitivity
ARCHAEOLOGICAL_PROVIDER=anthropic    # Archaeological expertise
EXECUTION_PROVIDER=bitnet            # Action selection, implementation
MEMORY_PROVIDER=deepseek             # Information storage, retrieval
PERCEPTION_PROVIDER=openai           # Pattern recognition, processing (implemented) (implemented)
```

### Performance Configuration

```bash
# LLM Request Settings
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=1
LLM_TIMEOUT=30
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=3600

# Cognitive Orchestra Settings
COGNITIVE_ORCHESTRA_ENABLED=true
COGNITIVE_PARALLEL_PROCESSING=true
COGNITIVE_MAX_CONCURRENT=6
COGNITIVE_HARMONY_THRESHOLD=0.7
```

## Provider Recommendations

### For Development/Testing
- Enable **mock provider** only:
  ```bash
  DEFAULT_LLM_PROVIDER=mock
  FALLBACK_TO_MOCK=true
  ```

### For Research/Specialized Use
- **Primary:** Anthropic Claude (reasoning and consciousness)
- **Secondary:** OpenAI GPT-4 (creativity and perception)
- **Cost-effective:** DeepSeek (memory and execution)

### For Production
- **Multi-provider setup** with all providers enabled
- **Load balancing** across providers
- **Fallback chain** configured

## Setup Methods

### Method 1: Manual Setup

1. Copy `.env.example` to `.env`
2. Edit `.env` with your API keys
3. Set `ENABLED=true` for each provider you want to use
4. Test with `python test_env_simple.py`

### Method 2: Interactive Setup (Recommended)

```bash
python scripts/setup_env_example.py
```

This script will:
- Guide you through provider setup
- Help you choose the recommended default provider
- Configure infrastructure settings
- Test the configuration

### Method 3: Programmatic Setup

```python
from src.utils.env_config import EnvironmentConfig

env = EnvironmentConfig()
config = env.get_llm_config()

# Check configured providers
manager = LLMManager()
providers = manager.get_configured_providers()
print(f"Configured providers: {providers}")
```

## Testing Your Setup

### Basic Test
```bash
python test_env_simple.py
```

### Comprehensive Test
```bash
python scripts/test_env_config.py
```

### Test Specific Provider
```python
from src.llm.llm_manager import LLMManager
from src.llm.base_llm_provider import LLMMessage, LLMRole

async def test_provider():
    manager = LLMManager()
    provider = manager.get_provider("anthropic")  # or "openai", "deepseek"
    
    messages = [LLMMessage(role=LLMRole.USER, content="Hello!")]
    response = await provider.generate(messages)
    print(response.content)
```

## Cognitive Orchestra

The NIS Protocol v3 includes a **Cognitive Orchestra** that systematically selects the appropriate LLM for each task:

- **Consciousness tasks** → Anthropic Claude (for self-reflection)
- **Creative tasks** → OpenAI GPT-4 (for creative ideas)
- **Reasoning tasks** → Anthropic Claude (for logical analysis)
- **Memory tasks** → DeepSeek (cost-effective, good memory)
- **Execution tasks** → BitNet (fast local execution)

This happens systematically - you don't need to manually assign providers.

## Troubleshooting

### Common Issues

**"No LLM providers configured"**
- Check that at least one provider has `ENABLED=true`
- Verify API keys are set and valid
- Ensure API keys don't contain placeholder values

**"API key is required"**
- Set the API key environment variable
- Remove placeholder values like `your_openai_api_key_here`

**"ImportError: attempted relative import"**
- Run tests from the project root directory
- Use `python test_env_simple.py` not `python scripts/test_env_config.py`

**Mock provider always used**
- Enable at least one real provider with `ENABLED=true`
- Check API keys are properly set
- Test with `python test_env_simple.py`

### Environment Variable Priority

1. **Environment variables** (highest priority)
2. **`.env` file** values
3. **JSON configuration** files (backward compatibility)
4. **Default values** (lowest priority)

### Security recommended Practices

- **Never commit** `.env` file to version control
- **Use different keys** for development/production
- **Rotate API keys** regularly
- **Monitor usage** and costs
- **Use least privilege** - only enable needed providers

## Provider Comparison

| Provider | Cost | Speed | Reasoning | Creativity | Local |
|----------|------|-------|-----------|------------|-------|
| OpenAI | High | Fast | Excellent | Excellent | No |
| Anthropic | High | Medium | Outstanding | Good | No |
| DeepSeek | Low | Fast | Good | Fair | No |
| BitNet | Free | Very Fast | Fair | Poor | Yes |
| Mock | Free | fast | None | None | Yes |

## Integration Examples

### Basic Agent Usage
```python
from src.agents.reasoning.enhanced_reasoning_agent import EnhancedReasoningAgent

agent = EnhancedReasoningAgent()
result = await agent.reason("What are the implications of AI consciousness?")
# systematically uses recommended provider for reasoning (Anthropic)
```

### Cognitive Function Usage
```python
from src.llm.llm_manager import LLMManager

manager = LLMManager()

# Generate with specific cognitive function
response = await manager.generate_with_function(
    messages=[LLMMessage(role=LLMRole.USER, content="Create a poem")],
    function="creativity"  # systematically uses OpenAI
)
```

### Multi-Provider Comparison
```python
from src.llm.cognitive_orchestra import CognitiveOrchestra

orchestra = CognitiveOrchestra(manager)
responses = await orchestra.parallel_generate(
    context=context,
    providers=["anthropic", "openai", "deepseek"]
)
# Returns fused response from multiple providers
```

## Next Steps

1. **Set up your providers** using this guide
2. **Test the configuration** with provided scripts
3. **Explore the agents** in `src/agents/`
4. **Run the examples** in `examples/`
5. **Read the full documentation** in `docs/`

## Support

- **Test scripts:** `test_env_simple.py`, `scripts/test_env_config.py`
- **Setup script:** `scripts/setup_env_example.py`
- **Documentation:** `docs/` directory
- **Examples:** `examples/` directory

---

**Remember:** The system works with mock providers for testing even without API keys. You can start developing immediately and add real providers when ready for production use. 