# 📬 NIS Protocol v3.2 - Complete Multimodal Postman Import Guide

## 🎯 **Overview**

This guide provides comprehensive instructions for importing and using the **NIS Protocol v3.2 Complete Multimodal API Collection** in Postman. The collection includes all endpoints for AI image generation, multimodal research, collaborative reasoning, and document processing.

---

## 📂 **Collection Structure**

### **🏠 System Endpoints**
- Root system information and health checks
- Provider status and capability discovery
- Version information and demo interfaces

### **💬 Chat Endpoints** 
- Standard chat interface
- Formatted human-readable responses
- Streaming chat for real-time interactions

### **🧠 Multi-LLM Orchestration**
- Multi-provider consensus building
- Complex reasoning with provider validation
- Error handling and fallback mechanisms

### **🚀 NVIDIA Models**
- Nemotron general AI processing
- Nemo physics-focused computations
- Modulus simulation capabilities

### **🎯 Agent Endpoints**
- Learning agent parameter management
- Agent status monitoring and control
- Specialized agent coordination

### **🎨 AI Image Generation (NEW v3.2)**
- **DALL-E Scientific**: OpenAI-powered technical illustrations
- **Imagen Artistic**: Google-powered creative visualizations
- **Auto Provider**: Intelligent provider selection
- **Image Editing**: AI-powered enhancement and modification
- **Batch Generation**: Multiple image variations

### **🔬 Multimodal Research (NEW v3.2)**
- **Vision Analysis**: Physics-focused image analysis
- **Deep Research**: Multi-source academic research
- **Document Analysis**: Academic paper processing
- **Collaborative Reasoning**: Multi-model problem solving
- **Claim Validation**: Evidence-based fact checking
- **Agent Status**: Comprehensive multimodal capabilities

---

## 🔧 **Import Instructions**

### **Step 1: Download Collection**
```bash
# Collection file location
NIS_Protocol_v3_COMPLETE_Postman_Collection.json
```

### **Step 2: Import to Postman**
1. **Open Postman**
2. **Click "Import"** (top left)
3. **Select "Upload Files"**
4. **Choose** `NIS_Protocol_v3_COMPLETE_Postman_Collection.json`
5. **Click "Import"**

### **Step 3: Configure Environment**
```json
{
  "base_url": "http://localhost"
}
```

---

## 🎨 **NEW v3.2: AI Image Generation Testing**

### **🖼️ Generate Scientific Illustration**
```json
POST {{base_url}}/image/generate
{
  "prompt": "A quantum computer in a futuristic laboratory with holographic displays showing quantum entanglement patterns",
  "style": "scientific",
  "size": "1024x1024",
  "provider": "openai",
  "quality": "hd",
  "num_images": 1
}
```

**Expected Response:**
- **Status**: 200 OK
- **Generation Time**: 5-15 seconds
- **Image Format**: Base64 PNG/JPEG
- **Enhanced Prompt**: AI-optimized version
- **Metadata**: Model info, safety filters, content policy

### **🎨 Generate Artistic Visualization**
```json
POST {{base_url}}/image/generate
{
  "prompt": "Neural network visualization with flowing data streams and glowing nodes in an abstract digital space",
  "style": "artistic",
  "provider": "google",
  "num_images": 2
}
```

### **✏️ AI Image Editing**
```json
POST {{base_url}}/image/edit
{
  "image_data": "base64_encoded_image_data",
  "prompt": "Add glowing holographic interfaces and quantum particle effects",
  "provider": "openai"
}
```

---

## 🔬 **NEW v3.2: Multimodal Research Testing**

### **👁️ Vision Analysis**
```json
POST {{base_url}}/vision/analyze
{
  "image_data": "base64_image_data",
  "analysis_type": "physics_focused",
  "provider": "auto",
  "context": "Analyze scientific equipment for measurement accuracy"
}
```

### **📚 Deep Research**
```json
POST {{base_url}}/research/deep
{
  "query": "Latest developments in quantum computing error correction 2024",
  "research_depth": "comprehensive",
  "source_types": ["arxiv", "semantic_scholar", "wikipedia"],
  "min_sources": 5
}
```

### **🧠 Collaborative Reasoning**
```json
POST {{base_url}}/reasoning/collaborative
{
  "problem": "What are the philosophical implications of consciousness in AI systems?",
  "reasoning_type": "philosophical",
  "depth": "comprehensive",
  "require_consensus": true
}
```

---

## ✅ **Testing Workflow**

### **1. System Health Check**
```bash
GET {{base_url}}/health
Expected: 200 OK with provider status
```

### **2. Multimodal Agent Status**
```bash
GET {{base_url}}/agents/multimodal/status  
Expected: All agents operational with capabilities list
```

### **3. Test Image Generation**
```bash
POST {{base_url}}/image/generate
Expected: Generated image with metadata
```

### **4. Test Vision Analysis**
```bash
POST {{base_url}}/vision/analyze
Expected: Detailed image analysis results
```

### **5. Test Research Capabilities**
```bash
POST {{base_url}}/research/deep
Expected: Multi-source research compilation
```

---

## 🔧 **Environment Variables**

### **Required Variables**
```json
{
  "base_url": "http://localhost",
  "api_version": "v3.2"
}
```

### **Optional Variables**
```json
{
  "openai_model": "dall-e-3",
  "google_model": "imagen-2", 
  "default_style": "scientific",
  "default_size": "1024x1024"
}
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **502 Bad Gateway**
```bash
# Check if NIS Protocol is running
curl -I http://localhost/health

# If not running, start the system
./start.sh
```

#### **Image Generation Timeout**
```json
{
  "error": "Image generation timeout",
  "solution": "Reduce image size or complexity of prompt"
}
```

#### **Provider Unavailable**
```json
{
  "error": "Provider not available",
  "solution": "Use 'auto' provider or switch to alternative"
}
```

### **Debug Steps**
1. **Verify System Status**: `GET /health`
2. **Check Multimodal Agents**: `GET /agents/multimodal/status`
3. **Test Simple Endpoint**: `GET /` (root)
4. **Review Provider Capabilities**: Check response metadata
5. **Validate Request Format**: Ensure JSON structure matches schema

---

## 📊 **Performance Expectations**

| **Endpoint Type** | **Response Time** | **Success Rate** |
|-------------------|-------------------|------------------|
| 🎨 **Image Generation** | 5-15 seconds | >95% |
| ✏️ **Image Editing** | 8-20 seconds | >90% |
| 👁️ **Vision Analysis** | 2-5 seconds | >98% |
| 🔬 **Deep Research** | 30-120 seconds | >95% |
| 🧠 **Reasoning** | 10-30 seconds | >95% |
| 💬 **Chat** | 1-3 seconds | >99% |

---

## 🎯 **Best Practices**

### **Image Generation**
- **Use specific prompts** for better results
- **Choose appropriate style** for use case
- **Start with single images** before batch generation
- **Use 'auto' provider** for optimal selection

### **Research Queries**
- **Be specific** with research questions
- **Use appropriate depth** (comprehensive for detailed analysis)
- **Include relevant source types** for your domain
- **Allow sufficient time** for deep research (60-300 seconds)

### **Document Analysis**
- **Use base64 encoding** for PDF uploads
- **Specify document type** when known
- **Enable citation analysis** for academic papers
- **Choose appropriate processing mode** for your needs

---

## 🚀 **Advanced Usage**

### **Workflow Automation**
```javascript
// Example: Automated image generation workflow
const workflow = async () => {
  // 1. Generate concept image
  const concept = await generateImage({
    prompt: "AI research laboratory concept",
    style: "scientific"
  });
  
  // 2. Analyze generated image
  const analysis = await analyzeImage({
    image_data: concept.images[0].url,
    analysis_type: "technical"
  });
  
  // 3. Research related topics
  const research = await deepResearch({
    query: `Research topics related to: ${analysis.main_topics.join(", ")}`
  });
  
  return { concept, analysis, research };
};
```

### **Batch Processing**
```json
{
  "prompt": "Variations of quantum computer designs: gate-based, annealing, photonic, topological",
  "style": "scientific",
  "num_images": 4,
  "provider": "auto"
}
```

---

## 📞 **Support**

### **Documentation**
- **Complete API Reference**: `MULTIMODAL_API_DOCUMENTATION.md`
- **Interactive Docs**: `http://localhost/docs`
- **Chat Console**: `http://localhost/console`

### **Testing Resources**
- **All endpoints** include example requests
- **Response schemas** documented for each endpoint  
- **Error codes** explained with solutions
- **Performance metrics** included for planning

---

*🏺 NIS Protocol v3.2 - Advancing the frontiers of multimodal AI with revolutionary image generation and research capabilities*
