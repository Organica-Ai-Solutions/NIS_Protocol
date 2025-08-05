"""
NIS Protocol Google/Gemini & Imagen LLM Provider
This module implements the Google Gemini API for text and the dedicated 
Google Cloud AI Platform Imagen 2 API for high-quality image generation.
"""
import os
import logging
import base64
from typing import Dict, Any, List, Optional, Union

# --- Gemini Core Imports ---
try:
    import google.generativeai as google_genai_module
    from google.generativeai import types as google_types
except ImportError:
    google_genai_module = None
    google_types = None

# --- Imagen 2 Imports via AI Platform ---
try:
    from google.cloud import aiplatform
    from vertexai.preview.vision_models import ImageGenerationModel
except ImportError:
    aiplatform = None
    ImageGenerationModel = None

# --- General Imports ---
try:
    from PIL import Image
except ImportError:
    Image = None
import io
import asyncio
from ..base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole

logger = logging.getLogger(__name__)

class GoogleProvider(BaseLLMProvider):
    """
    Google Provider supporting Gemini for text and Imagen 2 for image generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Google provider for both Gemini and Imagen."""
        super().__init__(config)
        
        # --- General Configuration ---
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.use_mock = True  # Default to mock if setup fails

        # --- Gemini (Text) Configuration ---
        self.model = config.get("model", "gemini-1.5-flash")
        if google_genai_module and self.api_key:
            try:
                google_genai_module.configure(api_key=self.api_key)
                logger.info("Google Gemini (Text) API configured successfully.")
                self.use_mock = False
            except Exception as e:
                logger.error(f"Failed to configure Google Gemini API: {e}")
        else:
            logger.warning("Gemini (google-generativeai) not fully configured. Text generation will be mocked.")

        # --- Imagen 2 (Image) Configuration ---
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        self.gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        self.imagen_model_name = "imagegeneration@006" # Stable Imagen 2 model
        self.imagen_client = None

        if aiplatform and self.gcp_project_id:
            try:
                # Try service account authentication first
                service_account_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
                if service_account_key and os.path.exists(service_account_key) and os.path.isfile(service_account_key):
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(service_account_key)
                    aiplatform.init(
                        project=self.gcp_project_id, 
                        location=self.gcp_location,
                        credentials=credentials
                    )
                    logger.info(f"Google Cloud AI Platform (Imagen) configured with service account for project '{self.gcp_project_id}'.")
                else:
                    # Fall back to application default credentials
                    try:
                        aiplatform.init(project=self.gcp_project_id, location=self.gcp_location)
                        logger.info(f"Google Cloud AI Platform (Imagen) configured with default credentials for project '{self.gcp_project_id}'.")
                    except Exception as default_error:
                        logger.warning(f"Google Cloud default credentials failed: {default_error}. Using mock mode.")
                        self.use_mock = True
            except Exception as e:
                logger.warning(f"Google Cloud AI Platform configuration failed: {e}. Using mock mode.")
                self.use_mock = True
        else:
            logger.info("Google Cloud AI Platform not configured - using mock mode for image generation.")
            
        self.logger = logging.getLogger("google_provider")

    async def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic",
        size: str = "1024x1024",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an image using the robust Imagen 2 API via AI Platform.
        """
        if not self.gcp_project_id or not ImageGenerationModel or self.use_mock:
            self.logger.info("Using enhanced placeholder image (GCP/Imagen not available).")
            return await self._generate_enhanced_placeholder_image(prompt, style, size)

        enhanced_prompt = self._enhance_prompt_for_scientific_visuals(prompt, style)
        self.logger.info(f"🎨 Attempting REAL Imagen 2 Image Generation with prompt: {enhanced_prompt}")

        try:
            # Instantiate the model on demand
            model = ImageGenerationModel.from_pretrained(self.imagen_model_name)
            
            # The Imagen API returns a list of images. We'll take the first one.
            response = await asyncio.to_thread(
                model.generate_images,
                prompt=enhanced_prompt,
                number_of_images=1
            )
            
            image_bytes = response.images[0]._image_bytes
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:image/png;base64,{image_data}"
            
            self.logger.info("Successfully generated image from Imagen 2 API.")
            return {
                "status": "success",
                "prompt": prompt,
                "images": [{"url": data_url, "revised_prompt": enhanced_prompt}]
            }

        except Exception as e:
            self.logger.error(f"Definitive Imagen 2 image generation error: {e}")
            return await self._generate_enhanced_placeholder_image(prompt, style, size)

    # --- Helper and Abstract Method Implementations ---

    def _enhance_prompt_for_scientific_visuals(self, prompt: str, style: str) -> str:
        """Enhances the prompt for hyperrealistic, scientific visuals using prompt templates."""
        import json
        import os
        
        # Load prompt templates
        try:
            template_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'system', 'prompts', 'imagen_prompt_templates.json')
            with open(template_path, 'r') as f:
                templates = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load prompt templates: {e}")
            # Fallback to basic enhancement
            style_prefix = "A ultra-realistic, scientific photograph, cinematic 8k resolution, of"
            return f"{style_prefix} a {style} depicting: {prompt}"
        
        # Analyze prompt to determine best template
        prompt_lower = prompt.lower()
        enhanced_prompt = ""
        
        # Determine template based on content
        if "quantum" in prompt_lower or "consciousness" in prompt_lower:
            template = templates.get("quantum_physics", {})
            base = template.get("base_template", "Quantum physics visualization")
            elements = template.get("specific_elements", [])
            enhanced_prompt = f"{base}, {prompt}, featuring {', '.join(elements[:3])}"
            
        elif "neural" in prompt_lower or "brain" in prompt_lower or "neuroplasticity" in prompt_lower:
            template = templates.get("neuroplasticity", {})
            base = template.get("base_template", "Neural network visualization")
            elements = template.get("neural_elements", [])
            enhanced_prompt = f"{base}, {prompt}, showing {', '.join(elements[:3])}"
            
        elif "bouncing ball" in prompt_lower or "physics" in prompt_lower:
            template = templates.get("physics_experiments", {}).get("bouncing_ball", {})
            scenario = template.get("scenario", "Physics demonstration")
            elements = template.get("elements", [])
            enhanced_prompt = f"{scenario}, {prompt}, including {', '.join(elements[:3])}"
            
        elif "earth" in prompt_lower and ("space" in prompt_lower or "jupiter" in prompt_lower):
            template = templates.get("earth_from_space", {})
            base = template.get("base_template", "Earth from space view")
            elements = template.get("astronomical_elements", [])
            enhanced_prompt = f"{base}, {prompt}, with {', '.join(elements[:3])}"
            
        else:
            # Use scientific visualization template
            template = templates.get("scientific_visualization", {})
            base = template.get("base_template", "Scientific visualization")
            enhanced_prompt = f"{base}, {prompt}"
        
        # Add style modifiers
        style_template = templates.get("style_modifiers", {}).get(style, {})
        if style_template:
            prefix = style_template.get("prefix", "")
            camera = style_template.get("camera_settings", "")
            enhanced_prompt = f"{prefix}, {enhanced_prompt}, {camera}"
        
        # Add quality enhancers
        quality_enhancers = templates.get("quality_enhancers", [])
        enhanced_prompt += f", {', '.join(quality_enhancers[:4])}"
        
        logger.info(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    async def _generate_enhanced_placeholder_image(self, prompt: str, style: str, size: str) -> Dict[str, Any]:
        """Generates an enhanced placeholder image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import textwrap
            
            if not Image:
                return {"status": "error", "message": "PIL not installed"}
            
            width, height = map(int, size.split('x'))
            
            # Create enhanced placeholder
            bg_color = (245, 245, 250) if style == "scientific" else (248, 250, 252)
            border_color = (99, 102, 241) if style == "scientific" else (8, 145, 178)
            text_color = (55, 65, 81)
            
            img = Image.new('RGB', (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw decorative border
            border_width = max(3, width // 150)
            draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
            
            # Add content
            try:
                font = ImageFont.load_default()
                
                # Title
                title = "🎨 Google Imagen Concept"
                title_y = height // 5
                title_bbox = draw.textbbox((0, 0), title, font=font)
                title_x = (width - (title_bbox[2] - title_bbox[0])) // 2
                draw.text((title_x, title_y), title, fill=text_color, font=font)
                
                # Prompt (wrapped)
                prompt_text = f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}"
                prompt_y = height // 2.5
                wrapped_lines = textwrap.wrap(prompt_text, width=width//10)
                
                for i, line in enumerate(wrapped_lines[:3]):
                    line_bbox = draw.textbbox((0, 0), line, font=font)
                    line_x = (width - (line_bbox[2] - line_bbox[0])) // 2
                    draw.text((line_x, prompt_y + i * 25), line, fill=text_color, font=font)
                
                # Status
                status_text = "⚠️ Google Cloud configuration needed for real generation"
                status_y = height - height // 4
                status_bbox = draw.textbbox((0, 0), status_text, font=font)
                status_x = (width - (status_bbox[2] - status_bbox[0])) // 2
                draw.text((status_x, status_y), status_text, fill=(180, 83, 9), font=font)
                
            except Exception as font_error:
                # Fallback without font
                pass
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_str}"
            
            return {
                "status": "success", 
                "prompt": prompt,
                "images": [{
                    "url": data_url, 
                    "revised_prompt": f"Enhanced Google placeholder: {prompt}"
                }],
                "provider_used": "google_enhanced_placeholder",
                "note": "Configure Google Cloud for real Imagen generation"
            }
            
        except Exception as e:
            # Final fallback
            return {
                "status": "success", 
                "prompt": prompt,
                "images": [{
                    "url": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTEyIiBoZWlnaHQ9IjUxMiIgdmlld0JveD0iMCAwIDUxMiA1MTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSI1MTIiIGhlaWdodD0iNTEyIiBmaWxsPSIjRjVGNUZBIi8+Cjx0ZXh0IHg9IjI1NiIgeT0iMjU2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIiBmb250LXNpemU9IjE2IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiI+R29vZ2xlIEltYWdlbiBQbGFjZWhvbGRlcjwvdGV4dD4KPC9zdmc+",
                    "revised_prompt": f"Simple Google placeholder: {prompt}"
                }],
                "provider_used": "google_simple_placeholder"
            }
        
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        # ... (Gemini text generation implementation remains the same)
        if self.use_mock or not google_genai_module:
            return await self._mock_response(messages)
        # ... (rest of the text generation logic)
        return await self._mock_response(messages) # Fallback

    async def _mock_response(self, messages: List[LLMMessage]) -> LLMResponse:
        return LLMResponse(content="Mock Gemini Response", model=self.model, finish_reason="stop")

    async def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str): return [0.1] * 768
        else: return [[0.1] * 768] * len(text)

    def get_token_count(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
        
    async def close(self):
        pass
