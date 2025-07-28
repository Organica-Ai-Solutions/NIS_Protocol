"""
NIS Protocol Design Agent

This agent is responsible for translating high-level, natural language design concepts
into structured, quantitative parameters that can be used for simulation and generation.
"""

import logging
from typing import Dict, Any
import json

# Assuming these are the correct import paths.
# I will verify and correct them if necessary in a later step.
from src.agents.research.web_search_agent import WebSearchAgent, ResearchDomain
from src.llm.llm_manager import GeneralLLMProvider

class DesignAgent:
    """
    The DesignAgent takes a user's creative concept and grounds it in a structured format
    by using deep research and LLM-based parameter generation.
    """
    def __init__(self, llm_provider: GeneralLLMProvider, web_search_agent: WebSearchAgent):
        """
        Initializes the DesignAgent.
        """
        self.logger = logging.getLogger("design_agent")
        self.llm_provider = llm_provider
        self.web_search_agent = web_search_agent
        self.logger.info("Design Agent initialized with LLM and Web Search capabilities.")

    async def translate_concept_to_parameters(self, concept: str) -> Dict[str, Any]:
        """
        Translates a natural language concept into a dictionary of design parameters,
        enriched with web research.

        Args:
            concept: A string describing the design concept.

        Returns:
            A dictionary of structured design parameters.
        """
        self.logger.info(f"Translating concept: {concept}")

        # 1. Deep Research using WebSearchAgent
        self.logger.info(f"Performing deep research for: '{concept}'")
        research_results = await self.web_search_agent.research(
            query=concept,
            domain=ResearchDomain.TECHNICAL
        )
        
        synthesis = research_results.get("synthesis", "No synthesis available.")
        top_sources = [result.get('url', 'No URL') for result in research_results.get("top_results", [])[:3]]

        # 2. Parameter Generation using LLM
        self.logger.info("Generating structured parameters using LLM, informed by research...")
        prompt = self._create_parameter_generation_prompt(concept, synthesis, top_sources)
        
        messages = [{"role": "user", "content": prompt}]
        
        llm_response = await self.llm_provider.generate_response(messages, agent_type='design_engineer') 

        # 3. Parse and structure the output
        parameters = self._parse_llm_response(llm_response.get('content', '{}'))
        parameters['research_summary'] = synthesis
        parameters['research_sources'] = top_sources
        
        self.logger.info(f"Generated parameters: {json.dumps(parameters, indent=2)}")
        return parameters

    def _create_parameter_generation_prompt(self, concept: str, research_synthesis: str, sources: list) -> str:
        """Creates a detailed prompt for the LLM to generate design parameters."""
        return f"""
        As a world-class aerospace design engineer, your task is to translate a high-level creative concept into a structured set of initial design parameters for a simulation.

        **Concept:** "{concept}"

        **Initial Research Synthesis:**
        {research_synthesis}

        **Key Research Sources:**
        - {"'"+', '.join(sources)+"'" if sources else "N/A"}

        **Instructions:**
        Based on the concept and the provided research, generate a JSON object containing the key design parameters. The JSON object must have the following structure:
        - "base_inspiration": A string identifying the core inspiration (e.g., "falcon_wing", "nautilus_shell").
        - "primary_function": A string describing the main purpose of the design.
        - "constraints": A list of strings outlining key limitations and requirements (e.g., "must operate in high-altitude", "minimize radar cross-section").
        - "materials": A list of strings suggesting suitable materials (e.g., "titanium_alloy", "carbon_fiber_composite").
        - "performance_targets": An object with key-value pairs for target metrics (e.g., "lift_to_drag_ratio": 20, "max_speed_kph": 300, "weight_kg": 2.5).
        
        Provide only the raw JSON object in your response, without any markdown formatting or explanations.
        """

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the JSON response from the LLM."""
        try:
            # Clean up potential markdown code fences
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-4]
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            return {"error": "Failed to parse design parameters from LLM response."} 