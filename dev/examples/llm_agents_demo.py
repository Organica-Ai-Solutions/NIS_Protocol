"""
NIS Protocol LLM-Enabled Agents Demo

This script demonstrates the use of LLM-enabled cognitive agents in the NIS Protocol.
"""

import asyncio
import json
import logging
from typing import Dict, Any

from src.llm import LLMManager
from src.llm.base_llm_provider import LLMMessage, LLMRole
from src.neural_hierarchy.perception import PatternRecognitionAgent
from src.neural_hierarchy.memory import WorkingMemoryAgent
from src.neural_hierarchy.emotional import EmotionalProcessingAgent
from src.neural_hierarchy.executive import ExecutiveControlAgent
from src.neural_hierarchy.motor import MotorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_cognitive_pipeline(input_data: Dict[str, Any]):
    """Run input through the cognitive pipeline with LLM-enabled agents.
    
    Args:
        input_data: Input data to process
    """
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    try:
        # Initialize agents
        perception_agent = PatternRecognitionAgent(
            llm_provider=llm_manager.get_agent_llm("perception_agent")
        )
        
        memory_agent = WorkingMemoryAgent(
            llm_provider=llm_manager.get_agent_llm("memory_agent")
        )
        
        emotional_agent = EmotionalProcessingAgent(
            llm_provider=llm_manager.get_agent_llm("emotional_agent")
        )
        
        executive_agent = ExecutiveControlAgent(
            llm_provider=llm_manager.get_agent_llm("executive_agent")
        )
        
        motor_agent = MotorAgent(
            llm_provider=llm_manager.get_agent_llm("motor_agent")
        )
        
        # Process through perception layer
        logger.info("Processing through perception layer...")
        perception_result = await perception_agent.process_with_llm(
            input_data,
            system_prompt="Analyze the input for patterns and key features."
        )
        
        # Update working memory
        logger.info("Updating working memory...")
        memory_result = await memory_agent.process_with_llm(
            perception_result,
            system_prompt="Store and integrate new information with existing knowledge."
        )
        
        # Process emotional context
        logger.info("Processing emotional context...")
        emotional_result = await emotional_agent.process_with_llm(
            {
                "perception": perception_result,
                "memory": memory_result
            },
            system_prompt="Analyze emotional content and update emotional state."
        )
        
        # Executive decision making
        logger.info("Making executive decisions...")
        executive_result = await executive_agent.process_with_llm(
            {
                "perception": perception_result,
                "memory": memory_result,
                "emotional": emotional_result
            },
            system_prompt="Make decisions based on all available information."
        )
        
        # Generate motor actions
        logger.info("Generating motor actions...")
        motor_result = await motor_agent.process_with_llm(
            executive_result,
            system_prompt="Convert decisions into concrete actions."
        )
        
        return {
            "perception": perception_result,
            "memory": memory_result,
            "emotional": emotional_result,
            "executive": executive_result,
            "motor": motor_result
        }
        
    finally:
        # Clean up
        await llm_manager.close()

async def main():
    """Run the demo."""
    # Example input
    input_data = {
        "type": "text",
        "content": "The system needs to analyze market trends and make trading decisions.",
        "context": {
            "market": "crypto",
            "timeframe": "4h",
            "indicators": ["RSI", "MACD", "MA"]
        }
    }
    
    try:
        # Process through cognitive pipeline
        results = await run_cognitive_pipeline(input_data)
        
        # Print results
        print("\nCognitive Processing Results:")
        print("============================")
        for layer, result in results.items():
            print(f"\n{layer.upper()} Layer Output:")
            print("-" * 20)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Error in cognitive pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 