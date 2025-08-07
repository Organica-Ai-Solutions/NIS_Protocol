# 🌙 Kimi K2 API Setup Guide

## What is Kimi K2?

**Kimi** is a powerful AI model developed by **Moonshot AI** that excels at:
- **🔗 Ultra-long context**: Up to 128K tokens (200+ pages of text)
- **🌐 Multilingual**: Excellent Chinese and English understanding
- **📄 Document analysis**: Superior at processing lengthy documents
- **💬 Conversation continuity**: Maintains context across extended dialogues

## 🚀 Available Models

| Model | Context Window | Best For |
|-------|----------------|----------|
| `moonshot-v1-8k` | 8,000 tokens | General conversations |
| `moonshot-v1-32k` | 32,000 tokens | Long documents, detailed analysis |
| `moonshot-v1-128k` | 128,000 tokens | Extensive documents, research papers |

## 🔑 How to Enable Real Kimi K2 API

### Step 1: Get Your API Key
1. Visit: https://platform.moonshot.cn/
2. Create an account
3. Generate an API key
4. Copy your key (starts with `sk-...`)

### Step 2: Configure NIS Protocol
Add your Kimi API key to your `.env` file:

```bash
# Edit or create .env file
echo "KIMI_API_KEY=sk-your_actual_kimi_key_here" >> .env
```

### Step 3: Restart System
```bash
./stop.sh && ./start.sh
```

## 🎯 How to Use Kimi K2

### In the Chat Console
1. Go to: http://localhost/console
2. Select **"🌙 Kimi K2 (Long Context)"** from the provider dropdown
3. Ask any question or upload long documents

### Example Use Cases

**📚 Long Document Analysis:**
```
Provider: Kimi K2
Message: "Please analyze this 50-page research paper and summarize the key findings..."
```

**🌐 Multilingual Tasks:**
```
Provider: Kimi K2  
Message: "请帮我翻译这个英文文档并解释其中的技术概念"
```

**🔄 Extended Conversations:**
```
Provider: Kimi K2
Message: "Let's have a detailed discussion about quantum computing. Start with the basics..."
[Kimi will remember the entire conversation context]
```

## 🧠 Multimodel with Kimi

When you select **"🧠 Multimodel (Consensus)"**, Kimi is automatically included for:
- Long context tasks
- Multilingual requests  
- Document analysis
- Tasks requiring extensive memory

## 🌟 Kimi Strengths in NIS Protocol

### Long Context Tasks
- **Document Processing**: Upload PDFs, research papers
- **Code Review**: Analyze large codebases
- **Creative Writing**: Maintain story consistency across chapters

### Multilingual Capabilities
- **Translation**: Chinese ↔ English with technical accuracy
- **Cultural Context**: Understanding regional nuances
- **Code Comments**: Multilingual documentation

### Research & Analysis
- **Academic Papers**: Deep analysis of research findings
- **Business Reports**: Comprehensive document review
- **Technical Documentation**: Understanding complex specifications

## 🔧 Advanced Configuration

### Custom Model Selection
Add to your `.env` file:
```bash
KIMI_API_KEY=sk-your_key_here
KIMI_MODEL=moonshot-v1-128k  # For maximum context
KIMI_API_BASE=https://api.moonshot.cn/v1
```

### Integration with Other Providers
Kimi works seamlessly with:
- **OpenAI**: Creative + Long Context
- **Anthropic**: Safety + Extended Memory  
- **DeepSeek**: Math + Document Analysis
- **Google**: Research + Multilingual

## 🎨 Example Responses

### Without API Key (Mock Mode):
```
🌙 Kimi K2 Response (Mock Mode)

I'm Kimi, powered by Moonshot AI, specializing in long-context 
understanding and multilingual capabilities.

Note: Add your Kimi API key to enable real responses!
```

### With API Key (Real Mode):
```
🌙 Based on my analysis of your 128K token document...

[Detailed, context-aware response maintaining full document memory]

Context used: 95,432 tokens
Model: moonshot-v1-128k
```

## 🚨 Troubleshooting

### Common Issues

**❌ "API key not configured"**
- Check your `.env` file exists
- Verify `KIMI_API_KEY=sk-...` format
- Restart Docker containers

**❌ "Rate limit exceeded"**
- Kimi has usage limits on free tier
- Consider upgrading your Moonshot account
- Try spacing out requests

**❌ "Context too long"**
- Switch to `moonshot-v1-128k` model
- Break large documents into sections
- Use document chunking strategies

## 💡 Pro Tips

1. **Use Kimi for Research**: Perfect for analyzing academic papers and reports
2. **Multilingual Projects**: Ideal for Chinese-English translation tasks  
3. **Long Conversations**: Great for extended technical discussions
4. **Document Q&A**: Upload documents and ask specific questions
5. **Combine with Multimodel**: Get consensus from multiple AI perspectives

---

**🌙 Kimi K2 is now integrated into your NIS Protocol v3.2 system!**

Ready to experience ultra-long context AI conversations? 🚀