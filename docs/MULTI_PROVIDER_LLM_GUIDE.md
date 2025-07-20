# NIS Protocol Multi-Provider LLM System

## 🎯 **Overview**

The NIS Protocol now supports **4 different LLM providers** in a fully modular architecture. Developers can choose their preferred provider(s) or use multiple providers for different cognitive agents.

## 🤖 **Supported Providers**

| Provider | Type | Models | Features | Recommended For |
|----------|------|--------|----------|----------|
| **OpenAI** | API | GPT-4o, GPT-4 Turbo | Chat, Embeddings, Function calling | General purpose, high quality |
| **Anthropic Claude** | API | Claude-3.5 Sonnet, Opus, Haiku | Long context, Constitutional AI | Safety-focused, reasoning |
| **DeepSeek** | API | DeepSeek-Chat, DeepSeek-Coder | Code generation, Math reasoning | Technical tasks, coding |
| **BitNet 2** | Local | Custom models | 1-bit quantization, CPU efficient | Privacy, offline, low resource |

## 🚀 **Quick Start**

### 1. Check Current Status
```bash
python examples/multi_provider_demo.py
```

### 2. Choose Your Provider
All providers are **disabled by default**. Enable the one you want to use:

```json
{
  "providers": {
    "openai": {
      "enabled": true,  // ← Set to true
      "api_key": "sk-your-actual-key-here"
    }
  }
}
```

### 3. Test Your Setup
The demo will show which provider is active and test it with a sample request.

## 📝 **Configuration Guide**

### OpenAI Configuration
```json
{
  "providers": {
    "openai": {
      "enabled": true,
      "api_key": "sk-your-openai-api-key",
      "api_base": "https://api.openai.com/v1",
      "organization": "your-org-id",  // Optional
      "models": {
        "chat": {
          "name": "gpt-4o",
          "max_tokens": 4096,
          "temperature": 0.7
        },
        "embedding": {
          "name": "text-embedding-3-small",
          "dimensions": 1536
        }
      }
    }
  }
}
```

**Setup Steps:**
1. Get API key: https://platform.openai.com/api-keys
2. Copy your API key
3. Edit `config/llm_config.json`
4. Set `"enabled": true` and paste your key

### Anthropic Claude Configuration
```json
{
  "providers": {
    "anthropic": {
      "enabled": true,
      "api_key": "sk-ant-your-anthropic-key",
      "api_base": "https://api.anthropic.com/v1",
      "version": "2023-06-01",
      "models": {
        "chat": {
          "name": "claude-3-5-sonnet-20241022",
          "max_tokens": 4096,
          "temperature": 0.7
        }
      }
    }
  }
}
```

**Setup Steps:**
1. Get API key: https://console.anthropic.com/
2. Copy your API key
3. Edit `config/llm_config.json`
4. Set `"enabled": true` and paste your key

### DeepSeek Configuration
```json
{
  "providers": {
    "deepseek": {
      "enabled": true,
      "api_key": "your-deepseek-api-key",
      "api_base": "https://api.deepseek.com/v1",
      "models": {
        "chat": {
          "name": "deepseek-chat",
          "max_tokens": 4096,
          "temperature": 0.7
        }
      }
    }
  }
}
```

**Setup Steps:**
1. Get API key: https://platform.deepseek.com/
2. Copy your API key
3. Edit `config/llm_config.json`
4. Set `"enabled": true` and paste your key

### BitNet 2 Configuration
```json
{
  "providers": {
    "bitnet": {
      "enabled": true,
      "model_path": "./models/bitnet/model.bin",
      "executable_path": "bitnet",
      "context_length": 4096,
      "cpu_threads": 8,
      "models": {
        "chat": {
          "name": "bitnet-b1.58-3b",
          "max_tokens": 2048,
          "temperature": 0.7
        }
      }
    }
  }
}
```

**Setup Steps:**
1. Install BitNet 2: https://github.com/microsoft/BitNet
2. Download a BitNet model
3. Update `model_path` to your model location
4. Set `"enabled": true`

## 🧠 **Agent-Specific Providers**

You can configure different providers for different cognitive agents:

```json
{
  "agent_llm_config": {
    "default_provider": "openai",
    "perception_agent": {
      "provider": "openai",       // Fast, good for perception
      "model": "gpt-4o"
    },
    "memory_agent": {
      "provider": "anthropic",    // Long context for memory
      "model": "claude-3-5-sonnet-20241022"
    },
    "emotional_agent": {
      "provider": "deepseek",     // Good emotional understanding
      "model": "deepseek-chat"
    },
    "executive_agent": {
      "provider": "openai",       // Strategic reasoning
      "model": "gpt-4o"
    },
    "motor_agent": {
      "provider": "bitnet",       // Local execution
      "model": "bitnet-b1.58-3b"
    }
  }
}
```

## 🔧 **Development Mode**

When no providers are configured, the system defaults to the **Mock Provider**:

- ✅ Works without API keys
- 🎭 Generates contextual responses for each agent type
- 🧪 Well-suited for development and testing
- 📚 Responses are themed around archaeological research

```json
{
  "agent_llm_config": {
    "fallback_to_mock": true  // Enable mock fallback (default)
  }
}
```

## 🚀 **Usage Examples**

### Basic Usage
```python
from src.llm.llm_manager import LLMManager
from src.llm.base_llm_provider import LLMMessage, LLMRole

# Initialize manager (auto-selects suitable available provider)
llm_manager = LLMManager()

# Get default provider
provider = llm_manager.get_provider()

# Generate response
messages = [
    LLMMessage(role=LLMRole.USER, content="Hello!")
]
response = await provider.generate(messages)
print(response.content)
```

### Specific Provider
```python
# Request specific provider
openai_provider = llm_manager.get_provider("openai")
claude_provider = llm_manager.get_provider("anthropic")
```

### Agent-Specific Provider
```python
# Get provider configured for specific agent
perception_llm = llm_manager.get_agent_llm("perception_agent")
memory_llm = llm_manager.get_agent_llm("memory_agent")
```

## 📊 **Provider Comparison**

### Performance Characteristics

| Provider | Speed | Quality | Context | Cost | Privacy |
|----------|-------|---------|---------|------|---------|
| OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 128k | $$$ | Cloud |
| Anthropic | ⚡⚡ | ⭐⭐⭐⭐⭐ | 200k | $$$ | Cloud |
| DeepSeek | ⚡⚡⚡ | ⭐⭐⭐⭐ | 64k | $ | Cloud |
| BitNet 2 | ⚡ | ⭐⭐⭐ | 4k | Free | Local |

### Use Case Recommendations

**🎯 General Purpose:** OpenAI GPT-4o
- Good overall balance of speed, quality, and features
- Excellent for most cognitive agents
- Strong function calling capabilities

**🧠 Complex Reasoning:** Anthropic Claude
- Strong performance for complex analytical tasks
- Well-suited for memory and executive agents
- Excellent safety alignment

**💻 Code & Technical:** DeepSeek
- Specialized in technical tasks
- Great for development and engineering
- Cost-effective option

**🔒 Privacy & Local:** BitNet 2
- Full privacy (no data sent to cloud)
- Works offline
- Lower resource requirements
- Good for sensitive applications

## 🔍 **Troubleshooting**

### Provider Not Working?
1. **Check API Key**: Make sure it's valid and not a placeholder
2. **Check Enabled Status**: Ensure `"enabled": true`
3. **Check Quotas**: Verify you have API credits/quota
4. **Check Network**: Ensure internet connection for API providers

### Getting "Mock Provider" When You Want Real Provider?
- Check that your chosen provider is enabled and configured
- Verify API keys are set correctly
- Look at the logs for specific error messages

### Different Agents Using Wrong Providers?
- Check `agent_llm_config` section
- Ensure provider names are spelled correctly
- Set `default_provider` for fallback behavior

## 🎯 **Recommended Practices**

### For Development
- Keep `fallback_to_mock: true` during development
- Test with mock provider first
- Configure real providers only when needed

### For Production
- Choose one primary provider for consistency
- Set up multiple providers for redundancy
- Monitor API usage and costs
- Use local providers (BitNet) for sensitive data

### For Cost Optimization
- Use DeepSeek for cost-sensitive applications
- Use BitNet 2 for high-volume, local processing
- Cache responses to reduce API calls
- Choose models based on task complexity

### For Performance
- Use OpenAI for general tasks requiring speed
- Use Anthropic for complex reasoning tasks
- Use local models (BitNet) for latency-sensitive applications
- Configure different providers per agent type

## 📚 **Integration Examples**

The multi-provider system is already integrated throughout NIS Protocol:

- **🧠 Consciousness Agents**: Use providers for self-reflection
- **🎯 Goal Generation**: Use providers for autonomous goal creation
- **📚 Memory System**: Use providers for semantic understanding
- **😊 Emotional Processing**: Use providers for sentiment analysis
- **🎭 Agent Communication**: Use providers for natural language

Choose your provider and start building AGI with the cognitive architecture that works well for your needs!

## 🔗 **Related Documentation**

- [AGI System Overview](AGI_SYSTEM_OVERVIEW.md)
- [Tech Stack Integration](TECH_STACK_INTEGRATION.md)
- [Getting Started Guide](getting_started/Quick_Start_Guide.md)
- [API Reference](API_Reference.md) 