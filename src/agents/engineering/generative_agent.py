"""
NIS Protocol Generative Agent

This agent is responsible for creating generative content, such as 3D models or images,
based on structured design parameters.
"""

import logging
from typing import Dict, Any
import json

from src.llm.llm_manager import GeneralLLMProvider

class GenerativeAgent:
    """
    The GenerativeAgent translates design specifications into a tangible,
    generative output.
    """
    def __init__(self, llm_provider: GeneralLLMProvider):
        """
        Initializes the GenerativeAgent.
        """
        self.logger = logging.getLogger("generative_agent")
        self.llm_provider = llm_provider
        self.logger.info("Generative Agent initialized.")

    async def generate_model_from_parameters(self, design_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a 3D model description from a dictionary of design parameters.

        Args:
            design_parameters: A dictionary of structured design parameters.

        Returns:
            A dictionary representing the generated 3D model.
        """
        self.logger.info("Generating 3D model from parameters...")

        # 1. Create a detailed prompt for the LLM
        prompt = self._create_model_generation_prompt(design_parameters)
        
        messages = [{"role": "user", "content": prompt}]
        
        # 2. Call the LLM to get a structured description of the model
        llm_response = await self.llm_provider.generate_response(messages, agent_type='generative_agent')
        
        # 3. Parse and structure the output
        generated_model = self._parse_llm_response(llm_response.get('content', '{}'))
        generated_model['source_parameters'] = design_parameters
        
        self.logger.info(f"Generated model description: {json.dumps(generated_model, indent=2)}")
        return generated_model

    def _create_model_generation_prompt(self, design_parameters: Dict[str, Any]) -> str:
        """Creates a prompt for the LLM to generate a 3D model description."""
        params_str = json.dumps(design_parameters, indent=2)
        return f"""
        As a specialist in procedural 3D model generation, your task is to create a structured description of a 3D asset based on a set of engineering design parameters.

        **Design Parameters:**
        {params_str}

        **Instructions:**
        Generate a JSON object that describes the 3D model. The JSON object must represent a scene graph and include the following:
        - "model_name": A string name for the model (e.g., "Falcon_Inspired_Drone_Wing_v1").
        - "model_type": A string specifying the format, which should be "3D_vector_mesh".
        - "components": A list of objects, where each object is a part of the model (e.g., "main_wing_body", "control_surfaces", "winglets").
        - Each component object should have:
            - "name": A string name for the component.
            - "shape": A string describing the geometric shape (e.g., "airfoil_NACA2412_profile", "tapered_cuboid").
            - "material": A string referencing a material from the design parameters.
            - "transform": An object with "position", "rotation", and "scale" values.
            - "sub_components": (Optional) A nested list of components if the part is complex.
        
        Provide only the raw JSON object in your response, without any markdown formatting or explanations.
        """

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the JSON response from the LLM."""
        try:
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-4]
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            return {"error": "Failed to parse 3D model description from LLM response."} 