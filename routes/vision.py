"""
NIS Protocol v4.0 - Vision Routes

This module contains all vision, image, and visualization endpoints:
- Image analysis (multimodal)
- Image generation (DALL-E, etc.)
- Image editing
- Visualization creation (charts, diagrams)
- Document analysis

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.vision import router as vision_router
    app.include_router(vision_router, tags=["Vision"])
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.vision")

# Create router
router = APIRouter(tags=["Vision"])


# ====== Request Models ======

class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis"""
    image_data: str = Field(..., description="Base64 encoded image data")
    analysis_type: str = Field(default="general", description="Type of analysis: general, technical, scientific, mathematical")
    provider: Optional[str] = Field(default=None, description="Preferred AI provider")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for analysis")


class ImageGenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., description="Text prompt for image generation")
    style: str = Field(default="photorealistic", description="Style: photorealistic, artistic, scientific, anime, sketch")
    size: str = Field(default="1024x1024", description="Image size")
    provider: Optional[str] = Field(default=None, description="Preferred provider: openai, google")
    quality: str = Field(default="standard", description="Quality: standard, hd")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate")


class ImageEditRequest(BaseModel):
    """Request model for image editing"""
    image_data: str = Field(..., description="Base64 encoded original image")
    prompt: str = Field(..., description="Edit instructions")
    mask_data: Optional[str] = Field(default=None, description="Base64 encoded mask for selective editing")
    provider: Optional[str] = Field(default=None, description="Preferred provider")


class VisualizationRequest(BaseModel):
    """Request model for visualization creation"""
    data: Dict[str, Any] = Field(..., description="Data to visualize")
    chart_type: Optional[str] = Field(default="auto", description="Chart type: bar, line, pie, scatter, etc.")
    style: str = Field(default="scientific", description="Visual style")
    title: Optional[str] = Field(default=None, description="Chart title")
    physics_context: Optional[Dict[str, Any]] = Field(default=None, description="Physics context for scientific visualizations")


class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis"""
    document_data: str = Field(..., description="Base64 encoded document or URL")
    document_type: str = Field(default="auto", description="Document type: pdf, latex, html, text, auto")
    processing_mode: str = Field(default="comprehensive", description="Processing mode: quick, comprehensive, deep")
    extract_images: bool = Field(default=True, description="Extract and analyze images")
    analyze_citations: bool = Field(default=True, description="Analyze citations and references")


# ====== Dependency Injection ======

def get_vision_agent():
    return getattr(router, '_vision_agent', None)

def get_document_agent():
    return getattr(router, '_document_agent', None)

def get_diagram_agent():
    return getattr(router, '_diagram_agent', None)


# ====== Image Analysis Endpoints ======

@router.post("/vision/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    """
    🎨 Analyze images with advanced multimodal vision capabilities
    
    Supports:
    - Technical diagram analysis
    - Mathematical content recognition
    - Physics principle identification  
    - Scientific visualization understanding
    """
    try:
        vision_agent = get_vision_agent()
        
        if not vision_agent:
            raise HTTPException(status_code=500, detail="Vision agent not initialized")
        
        result = await vision_agent.analyze_image(
            image_data=request.image_data,
            analysis_type=request.analysis_type,
            provider=request.provider,
            context=request.context
        )
        
        return {
            "status": "success",
            "analysis": result,
            "agent_id": vision_agent.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")


@router.post("/vision/analyze/simple")
async def analyze_image_simple(request: Dict[str, Any]):
    """
    🔍 Simple image analysis endpoint
    
    Accepts flexible input format for quick analysis.
    """
    try:
        vision_agent = get_vision_agent()
        
        if not vision_agent:
            raise HTTPException(status_code=500, detail="Vision agent not initialized")
        
        image_data = request.get("image_data", request.get("image", ""))
        analysis_type = request.get("analysis_type", "general")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="image_data is required")
        
        result = await vision_agent.analyze_image(
            image_data=image_data,
            analysis_type=analysis_type,
            provider=request.get("provider"),
            context=request.get("context")
        )
        
        return {
            "status": "success",
            "analysis": result,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple vision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision/generate")
async def generate_vision(request: Dict[str, Any]):
    """
    🎨 Generate visual content based on description
    
    Alternative endpoint for vision generation tasks.
    """
    try:
        vision_agent = get_vision_agent()
        
        if not vision_agent:
            raise HTTPException(status_code=500, detail="Vision agent not initialized")
        
        prompt = request.get("prompt", request.get("description", ""))
        
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        
        result = await vision_agent.generate_image(
            prompt=prompt,
            style=request.get("style", "photorealistic"),
            size=request.get("size", "1024x1024"),
            provider=request.get("provider"),
            quality=request.get("quality", "standard"),
            num_images=request.get("num_images", 1)
        )
        
        return {
            "status": "success",
            "generation": result,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Image Generation Endpoints ======

@router.post("/image/generate")
async def generate_image(request: ImageGenerationRequest):
    """
    🎨 Generate images using AI providers (DALL-E, Imagen)
    
    Capabilities:
    - Text-to-image generation with multiple AI providers
    - Style control (photorealistic, artistic, scientific, anime, sketch)
    - Multiple sizes and quality settings
    - Batch generation (1-4 images)
    - Provider auto-selection based on style
    """
    try:
        vision_agent = get_vision_agent()
        
        # Try direct OpenAI API call first (most reliable)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and len(openai_api_key) > 10:
            try:
                import aiohttp
                
                headers = {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Use appropriate DALL-E model based on size and quality
                model = "dall-e-3" if request.quality == "hd" and request.size in ["1024x1024", "1792x1024", "1024x1792"] else "dall-e-2"
                
                payload = {
                    "model": model,
                    "prompt": f"{request.prompt} ({request.style} style, high quality)",
                    "n": 1,  # DALL-E 3 only supports 1 image
                    "size": request.size if request.size in ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] else "1024x1024"
                }
                
                if model == "dall-e-3":
                    payload["quality"] = "hd" if request.quality == "hd" else "standard"
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                    async with session.post(
                        "https://api.openai.com/v1/images/generations",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            image_url = result["data"][0]["url"]
                            
                            # Download and convert to base64
                            async with session.get(image_url) as img_response:
                                img_data = await img_response.read()
                                img_b64 = base64.b64encode(img_data).decode('utf-8')
                                data_url = f"data:image/png;base64,{img_b64}"
                            
                            generation_result = {
                                "status": "success",
                                "prompt": request.prompt,
                                "images": [{
                                    "url": data_url,
                                    "revised_prompt": result["data"][0].get("revised_prompt", request.prompt),
                                    "size": request.size,
                                    "format": "png"
                                }],
                                "provider_used": f"openai_direct_{model}",
                                "generation_info": {
                                    "model": model,
                                    "real_api": True,
                                    "method": "direct_api_call"
                                }
                            }
                            
                            logger.info(f"✅ Real OpenAI {model} image generation successful!")
                            
                            return {
                                "status": "success",
                                "generation": generation_result,
                                "agent_id": "direct_openai",
                                "timestamp": time.time()
                            }
                        else:
                            error_text = await response.text()
                            logger.warning(f"OpenAI API error {response.status}: {error_text}")
                            
            except Exception as openai_error:
                logger.warning(f"Direct OpenAI call failed: {openai_error}")
        
        # Fallback to vision agent
        if vision_agent:
            result = await vision_agent.generate_image(
                prompt=request.prompt,
                style=request.style,
                size=request.size,
                provider=request.provider,
                quality=request.quality,
                num_images=request.num_images
            )
            
            return {
                "status": "success",
                "generation": result,
                "agent_id": vision_agent.agent_id,
                "timestamp": time.time()
            }
        
        # Final fallback - create placeholder
        return await _create_enhanced_visual_placeholder(request.prompt, request.style, request.size)
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.post("/image/edit")
async def edit_image(request: ImageEditRequest):
    """
    ✏️ Edit existing images with AI-powered modifications
    
    Features:
    - AI-powered image editing and inpainting
    - Selective area editing with masks
    - Style transfer and modifications
    - Object addition/removal
    """
    try:
        vision_agent = get_vision_agent()
        
        if not vision_agent:
            raise HTTPException(status_code=500, detail="Vision agent not initialized")
        
        result = await vision_agent.edit_image(
            image_data=request.image_data,
            prompt=request.prompt,
            mask_data=request.mask_data,
            provider=request.provider
        )
        
        return {
            "status": "success",
            "editing": result,
            "agent_id": vision_agent.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image editing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")


# ====== Visualization Endpoints ======

@router.post("/visualization/create")
async def create_visualization(request: VisualizationRequest):
    """
    📊 Generate scientific visualizations and plots
    
    Capabilities:
    - Automatic chart type detection
    - Physics simulation visualizations
    - Scientific styling and formatting
    - AI-generated insights and interpretations
    """
    try:
        vision_agent = get_vision_agent()
        
        if not vision_agent:
            raise HTTPException(status_code=500, detail="Vision agent not initialized")
        
        result = await vision_agent.generate_visualization(
            data=request.data,
            chart_type=request.chart_type,
            style=request.style,
            title=request.title,
            physics_context=request.physics_context
        )
        
        return {
            "status": "success",
            "visualization": result,
            "agent_id": vision_agent.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization creation failed: {str(e)}")


@router.post("/visualization/chart")
async def generate_chart(request: dict):
    """
    📊 Generate precise charts using matplotlib (NOT AI image generation)
    
    Request format:
    {
        "chart_type": "bar|line|pie|scatter|histogram|heatmap",
        "data": {
            "categories": ["A", "B", "C"],
            "values": [10, 20, 15],
            "title": "My Chart",
            "xlabel": "Categories",
            "ylabel": "Values"
        },
        "style": "scientific|professional|default"
    }
    """
    try:
        from src.agents.visualization.diagram_agent import DiagramAgent
        local_diagram_agent = DiagramAgent()
        
        chart_type = request.get("chart_type", "bar")
        data = request.get("data", {})
        style = request.get("style", "scientific")
        
        logger.info(f"🎨 Generating precise {chart_type} chart")
        
        result = local_diagram_agent.generate_chart(chart_type, data, style)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "chart": result,
            "agent_id": "diagram_agent",
            "timestamp": time.time(),
            "note": "Generated with mathematical precision - NOT AI image generation"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@router.post("/visualization/diagram")
async def generate_diagram(request: dict):
    """
    🔧 Generate precise diagrams using code (NOT AI image generation)
    
    Request format:
    {
        "diagram_type": "flowchart|network|architecture|physics|pipeline",
        "data": {
            "nodes": [...],
            "edges": [...],
            "title": "My Diagram"
        },
        "style": "scientific|professional|default"
    }
    """
    try:
        from src.agents.visualization.diagram_agent import DiagramAgent
        local_diagram_agent = DiagramAgent()
        
        diagram_type = request.get("diagram_type", "flowchart")
        data = request.get("data", {})
        style = request.get("style", "scientific")
        
        logger.info(f"🔧 Generating precise {diagram_type} diagram")
        
        result = local_diagram_agent.generate_diagram(diagram_type, data, style)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "diagram": result,
            "agent_id": "diagram_agent", 
            "timestamp": time.time(),
            "note": "Generated with code precision - NOT AI image generation"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagram generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {str(e)}")


@router.post("/visualization/auto")
async def generate_visualization_auto(request: dict):
    """
    🎯 Auto-detect and generate the best visualization for your data
    
    Request format:
    {
        "prompt": "Show me a bar chart of sales data",
        "data": {...},
        "style": "scientific"
    }
    """
    try:
        prompt = request.get("prompt", "").lower()
        data = request.get("data", {})
        style = request.get("style", "scientific")
        
        # Auto-detect visualization type from prompt
        if any(word in prompt for word in ["bar", "column"]):
            viz_type, sub_type = "chart", "bar"
        elif any(word in prompt for word in ["line", "trend", "time"]):
            viz_type, sub_type = "chart", "line"
        elif any(word in prompt for word in ["pie", "proportion", "percentage"]):
            viz_type, sub_type = "chart", "pie"
        elif any(word in prompt for word in ["flow", "process", "workflow"]):
            viz_type, sub_type = "diagram", "flowchart"
        elif any(word in prompt for word in ["network", "graph", "connection"]):
            viz_type, sub_type = "diagram", "network"
        elif any(word in prompt for word in ["architecture", "system", "component"]):
            viz_type, sub_type = "diagram", "architecture"
        elif any(word in prompt for word in ["physics", "wave", "science"]):
            viz_type, sub_type = "diagram", "physics"
        elif any(word in prompt for word in ["pipeline", "nis", "transform"]):
            viz_type, sub_type = "diagram", "pipeline"
        else:
            viz_type, sub_type = "chart", "bar"
        
        logger.info(f"🎯 Auto-detected: {viz_type} -> {sub_type}")
        
        from src.agents.visualization.diagram_agent import DiagramAgent
        local_diagram_agent = DiagramAgent()
        
        if viz_type == "chart":
            result = local_diagram_agent.generate_chart(sub_type, data, style)
        else:
            result = local_diagram_agent.generate_diagram(sub_type, data, style)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "visualization": result,
            "detected_type": f"{viz_type}:{sub_type}",
            "agent_id": "diagram_agent_auto",
            "timestamp": time.time(),
            "note": f"Auto-detected {sub_type} from prompt, generated with precision"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto visualization failed: {str(e)}")


@router.post("/visualization/interactive")
async def generate_interactive_chart(request: dict):
    """
    🎯 Generate interactive Plotly charts with zoom, hover, and real-time capabilities
    
    Request format:
    {
        "chart_type": "line|bar|scatter|pie|real_time",
        "data": {...},
        "style": "scientific|professional|default"
    }
    """
    try:
        from src.agents.visualization.diagram_agent import DiagramAgent
        local_diagram_agent = DiagramAgent()
        
        chart_type = request.get("chart_type", "line")
        data = request.get("data", {})
        style = request.get("style", "scientific")
        
        logger.info(f"🎯 Generating INTERACTIVE {chart_type} chart")
        
        result = local_diagram_agent.generate_interactive_chart(chart_type, data, style)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "interactive_chart": result,
            "agent_id": "diagram_agent_interactive",
            "timestamp": time.time(),
            "note": "Interactive chart with zoom, hover, and real-time capabilities"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interactive chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interactive chart generation failed: {str(e)}")


@router.post("/visualization/dynamic")
async def generate_dynamic_chart(request: dict):
    """
    🎨 GPT-Style Dynamic Chart Generation
    
    Generates Python code on-the-fly and executes it to create precise charts.
    Similar to how GPT/Claude generate visualizations.
    
    Request format:
    {
        "content": "physics explanation text...",
        "topic": "bouncing ball physics",
        "chart_type": "physics" | "performance" | "comparison" | "auto"
    }
    """
    try:
        from src.agents.visualization.code_chart_agent import CodeChartAgent
        code_chart_agent = CodeChartAgent()
        
        content = request.get("content", "")
        topic = request.get("topic", "Data Visualization")
        chart_type = request.get("chart_type", "auto")
        original_question = request.get("original_question", "")
        response_content = request.get("response_content", "")
        
        logger.info(f"🎨 Dynamic chart generation: {topic} ({chart_type})")
        
        result = await code_chart_agent.generate_chart_from_content(
            content, topic, chart_type, original_question, response_content
        )
        
        if result.get("status") == "success":
            return {
                "status": "success",
                "dynamic_chart": result,
                "agent_id": "code_chart_agent",
                "timestamp": time.time(),
                "note": "Generated via Python code execution (GPT-style approach)",
                "method": "content_analysis_code_generation_execution"
            }
        else:
            return {
                "status": "fallback",
                "dynamic_chart": result,
                "agent_id": "code_chart_agent",
                "timestamp": time.time(),
                "note": "Using SVG fallback - code execution not available"
            }
            
    except Exception as e:
        logger.error(f"Dynamic chart generation failed: {e}")
        return {
            "status": "error",
            "message": f"Dynamic chart generation failed: {str(e)}",
            "fallback_chart": {
                "chart_image": _get_fallback_svg(),
                "title": request.get("topic", "Chart"),
                "method": "error_fallback"
            },
            "timestamp": time.time()
        }


# ====== Document Analysis Endpoints ======

@router.post("/document/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """
    📄 Analyze documents with advanced processing capabilities
    
    Supports:
    - PDF text extraction and analysis
    - Academic paper structure recognition
    - Table and figure extraction
    - Citation and reference analysis
    - Multi-language document support
    """
    try:
        document_agent = get_document_agent()
        
        if not document_agent:
            raise HTTPException(status_code=500, detail="Document agent not initialized")
        
        result = await document_agent.analyze_document(
            document_data=request.document_data,
            document_type=request.document_type,
            processing_mode=request.processing_mode,
            extract_images=request.extract_images,
            analyze_citations=request.analyze_citations
        )
        
        return {
            "status": "success",
            "analysis": result,
            "agent_id": document_agent.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


# ====== Helper Functions ======

async def _create_enhanced_visual_placeholder(prompt: str, style: str, size: str) -> Dict[str, Any]:
    """Create an enhanced visual placeholder when AI generation fails"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
        
        width, height = map(int, size.split('x'))
        
        # Create a visually appealing placeholder
        if style == "scientific":
            bg_color = (240, 248, 255)  # Alice blue
            border_color = (70, 130, 180)  # Steel blue
            text_color = (25, 25, 112)  # Midnight blue
        else:
            bg_color = (248, 250, 252)  # Gray-50
            border_color = (8, 145, 178)  # Cyan-600
            text_color = (31, 41, 55)  # Gray-800
        
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw border
        border_width = max(4, width // 200)
        draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
        
        # Draw inner decorative border
        inner_margin = border_width * 3
        draw.rectangle([inner_margin, inner_margin, width-inner_margin-1, height-inner_margin-1], 
                      outline=border_color, width=2)
        
        # Add title
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        title = "🎨 Visual Concept"
        title_y = height // 6
        
        if font:
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, title_y), title, fill=text_color, font=font)
        
        # Add prompt text (wrapped)
        prompt_text = f"Concept: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
        prompt_y = height // 3
        
        if font:
            max_chars = width // 8
            wrapped_lines = textwrap.wrap(prompt_text, width=max_chars)
            font_size = 16
            line_height = font_size + 4
            
            for i, line in enumerate(wrapped_lines[:4]):
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (width - line_width) // 2
                draw.text((line_x, prompt_y + i * line_height), line, fill=text_color, font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_data}"
        
        return {
            "status": "success",
            "generation": {
                "status": "success",
                "prompt": prompt,
                "images": [{
                    "url": data_url,
                    "revised_prompt": f"Enhanced placeholder: {prompt}",
                    "size": size,
                    "format": "png"
                }],
                "provider_used": "enhanced_placeholder",
                "generation_info": {
                    "model": "PIL_enhanced_placeholder",
                    "real_api": False,
                    "method": "local_generation",
                    "note": "AI generation temporarily unavailable"
                }
            },
            "agent_id": "enhanced_placeholder",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Enhanced placeholder creation failed: {e}")
        return {
            "status": "success", 
            "generation": {
                "status": "success",
                "prompt": prompt,
                "images": [{
                    "url": _get_fallback_svg(),
                    "revised_prompt": f"Simple placeholder: {prompt}",
                    "size": size,
                    "format": "svg"
                }],
                "provider_used": "simple_placeholder"
            },
            "agent_id": "simple_placeholder",
            "timestamp": time.time()
        }


def _get_fallback_svg() -> str:
    """Return a simple fallback SVG placeholder"""
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y4ZmFmYyIgc3Ryb2tlPSIjZTJlOGYwIiBzdHJva2Utd2lkdGg9IjIiLz48dGV4dCB4PSIyMDAiIHk9IjE1MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzY0NzQ4YiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTYiPvCfjoggVmlzdWFsIFBsYWNlaG9sZGVyPC90ZXh0Pjwvc3ZnPg=="


# ====== Dependency Injection Helper ======

def set_dependencies(
    vision_agent=None,
    document_agent=None,
    diagram_agent=None
):
    """Set dependencies for the vision router"""
    router._vision_agent = vision_agent
    router._document_agent = document_agent
    router._diagram_agent = diagram_agent
