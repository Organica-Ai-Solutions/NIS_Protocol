#!/usr/bin/env python3
"""
NIS Protocol Cognitive Orchestra Demo

Demonstrates the "cognitive orchestra" approach where different LLMs
are specialized for different cognitive functions, creating a symphony
of intelligence rather than relying on a single massive model.

This shows how:
- Consciousness tasks go to high-quality reasoning models (Anthropic/OpenAI)
- Creative tasks use higher temperature settings
- Execution tasks use fast, precise models (BitNet for speed)
- Cultural intelligence uses models with strong ethical reasoning
- Archaeological tasks use domain-specialized configurations

Run: python examples/cognitive_orchestra_demo.py
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from llm.llm_manager import LLMManager
    from llm.cognitive_orchestra import CognitiveOrchestra, CognitiveFunction
    from llm.base_llm_provider import LLMMessage
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🎼 This demo requires the cognitive orchestra implementation")
    print("   The demo shows the concept even without full implementation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cognitive_orchestra_demo")


class CognitiveOrchestraDemo:
    """Demonstrates the cognitive orchestra in action."""
    
    def __init__(self):
        """Initialize the demo."""
        self.llm_manager = None
        self.orchestra = None
    
    async def setup(self):
        """Set up the cognitive orchestra."""
        logger.info("🎼 Setting up Cognitive Orchestra...")
        
        # Initialize LLM manager
        self.llm_manager = LLMManager()
        
        # Initialize cognitive orchestra
        self.orchestra = CognitiveOrchestra(self.llm_manager)
        
        # Show available providers
        available_providers = self.llm_manager.get_configured_providers()
        logger.info(f"🎹 Available LLM providers: {available_providers}")
        
        if not available_providers:
            logger.info("🎭 No external providers configured - using mock provider for demo")
    
    async def demonstrate_specialized_processing(self):
        """Demonstrate how different cognitive functions use different LLMs."""
        logger.info("\n🎵 === Cognitive Orchestra Specialization Demo ===")
        
        # Define test scenarios for different cognitive functions
        test_scenarios = [
            {
                "function": CognitiveFunction.CONSCIOUSNESS,
                "task": "Analyze your own reasoning process when solving archaeological puzzles",
                "expected_specialization": "Meta-cognitive analysis, self-reflection"
            },
            {
                "function": CognitiveFunction.REASONING,
                "task": "Given these pottery fragments, determine the most likely cultural origin",
                "expected_specialization": "Logical analysis, structured thinking"
            },
            {
                "function": CognitiveFunction.CREATIVITY,
                "task": "Imagine systematic ways to preserve underwater archaeological sites",
                "expected_specialization": "systematic ideas, unconventional solutions"
            },
            {
                "function": CognitiveFunction.CULTURAL,
                "task": "Assess the cultural sensitivity of excavating a sacred burial ground",
                "expected_specialization": "Cultural intelligence, ethical considerations"
            },
            {
                "function": CognitiveFunction.ARCHAEOLOGICAL,
                "task": "Develop a preservation plan for newly discovered Mayan artifacts",
                "expected_specialization": "Domain expertise, methodological precision"
            },
            {
                "function": CognitiveFunction.EXECUTION,
                "task": "Generate precise coordinates for drone survey of archaeological site",
                "expected_specialization": "Fast, precise action selection"
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            function = scenario["function"]
            task = scenario["task"]
            
            logger.info(f"\n🎼 Processing {function.value.upper()} task...")
            logger.info(f"   Task: {task}")
            logger.info(f"   Expected: {scenario['expected_specialization']}")
            
            # Create messages for the task
            messages = [
                LLMMessage(role="user", content=task)
            ]
            
            try:
                # Process with cognitive orchestra
                response = await self.orchestra.process_cognitive_task(
                    function=function,
                    messages=messages,
                    context={"domain": "archaeology", "demo": True}
                )
                
                results[function] = response
                
                # Show specialization applied
                if hasattr(response, 'metadata') and response.metadata:
                    logger.info(f"   ✅ Specialization: {response.metadata.get('cognitive_function', 'none')}")
                
                # Show response preview (first 100 chars)
                preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
                logger.info(f"   📝 Response: {preview}")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing {function.value}: {e}")
                results[function] = None
        
        return results
    
    async def demonstrate_parallel_processing(self):
        """Demonstrate parallel processing of multiple cognitive functions."""
        logger.info("\n🎵 === Parallel Cognitive Processing Demo ===")
        
        # Define tasks that can be processed in parallel
        parallel_tasks = [
            (CognitiveFunction.REASONING, [LLMMessage(role="user", content="Analyze the structural integrity of this ancient building")], {"priority": "high"}),
            (CognitiveFunction.CREATIVITY, [LLMMessage(role="user", content="Brainstorm systematic archaeological documentation methods")], {"priority": "normal"}),
            (CognitiveFunction.CULTURAL, [LLMMessage(role="user", content="Evaluate cultural implications of this excavation plan")], {"priority": "high"}),
        ]
        
        logger.info(f"🎼 Processing {len(parallel_tasks)} cognitive tasks in parallel...")
        
        try:
            # Process tasks in parallel using orchestrate_parallel_processing
            # Note: This would be implemented in the full CognitiveOrchestra class
            results = {}
            
            # For demo, process sequentially but show the concept
            for function, messages, context in parallel_tasks:
                logger.info(f"   🎹 Starting {function.value} processing...")
                
                response = await self.orchestra.process_cognitive_task(
                    function=function,
                    messages=messages,
                    context=context
                )
                
                results[function] = response
                logger.info(f"   ✅ Completed {function.value}")
            
            logger.info(f"🎵 Parallel processing completed! Processed {len(results)} functions")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in parallel processing: {e}")
            return {}
    
    async def show_orchestra_status(self):
        """Show the current status of the cognitive orchestra."""
        logger.info("\n🎵 === Orchestra Status ===")
        
        try:
            status = self.orchestra.get_orchestra_status()
            
            logger.info(f"🎼 Active processes: {status['active_processes']}")
            logger.info(f"🎹 Active functions: {status['active_functions']}")
            logger.info(f"🎵 Harmony score: {status['harmony_score']:.3f}")
            logger.info(f"🎭 Available providers: {status['available_providers']}")
            
            # Show cognitive profiles
            logger.info("\n🎼 Cognitive Function Profiles:")
            for func_name, profile in status['cognitive_profiles'].items():
                logger.info(f"   {func_name}:")
                logger.info(f"     - Optimal providers: {profile['optimal_providers']}")
                logger.info(f"     - Latency priority: {profile['latency_priority']}")
                logger.info(f"     - Parallel capable: {profile['parallel_capable']}")
            
            # Show performance metrics if available
            if 'performance_metrics' in status and status['performance_metrics']:
                logger.info("\n📊 Performance Metrics:")
                for func, metrics in status['performance_metrics'].items():
                    logger.info(f"   {func.value}: {metrics}")
            
        except Exception as e:
            logger.error(f"❌ Error getting orchestra status: {e}")
    
    async def demonstrate_provider_optimization(self):
        """Show how different providers are efficient for different tasks."""
        logger.info("\n🎵 === Provider Optimization Demo ===")
        
        # Show the mapping strategy
        optimization_examples = [
            {
                "function": "Consciousness",
                "optimal_providers": ["anthropic", "openai"],
                "reason": "Deep reasoning capabilities, meta-cognitive analysis",
                "temperature": "0.3-0.7 (balanced creativity/precision)"
            },
            {
                "function": "Creativity", 
                "optimal_providers": ["openai", "anthropic"],
                "reason": "Strong creative capabilities, systematic idea generation",
                "temperature": "0.7-1.0 (high creativity)"
            },
            {
                "function": "Execution",
                "optimal_providers": ["bitnet", "deepseek"],
                "reason": "Fast inference, precise action selection",
                "temperature": "0.1-0.4 (high precision)"
            },
            {
                "function": "Cultural Intelligence",
                "optimal_providers": ["anthropic", "openai"],
                "reason": "Strong ethical reasoning, cultural sensitivity",
                "temperature": "0.4-0.7 (balanced approach)"
            },
            {
                "function": "Archaeological Domain",
                "optimal_providers": ["anthropic", "openai", "deepseek"],
                "reason": "Domain knowledge, methodological precision",
                "temperature": "0.3-0.6 (domain-focused)"
            }
        ]
        
        logger.info("🎼 Cognitive Orchestra Provider Optimization Strategy:")
        
        for example in optimization_examples:
            logger.info(f"\n🎹 {example['function']}:")
            logger.info(f"   Optimal Providers: {example['optimal_providers']}")
            logger.info(f"   Reason: {example['reason']}")
            logger.info(f"   Temperature: {example['temperature']}")
        
        logger.info("\n🎵 This creates a 'cognitive orchestra' where each LLM plays its optimal role!")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestra:
            await self.orchestra.close()
        logger.info("🎼 Cognitive Orchestra demo completed!")


async def demonstrate_cognitive_orchestra_concept():
    """Demonstrate the cognitive orchestra concept."""
    print("🎼 NIS Protocol Cognitive Orchestra Demo")
    print("=" * 50)
    print("Demonstrating multi-LLM specialization for different cognitive functions")
    print()
    
    # Show the cognitive orchestra concept
    print("🎵 === Cognitive Orchestra Architecture ===")
    print()
    
    cognitive_functions = {
        "🧠 Consciousness": {
            "description": "Deep self-reflection, meta-cognition, bias detection",
            "optimal_llms": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V2"],
            "temperature": "0.3-0.7 (balanced creativity/precision)",
            "specialization": "Meta-cognitive analysis, understanding reasoning processes",
            "parallel": "❌ (requires focused attention)",
            "use_case": "Analyzing own decision-making process in archaeological site evaluation"
        },
        
        "🔍 Reasoning": {
            "description": "Logical analysis, problem solving, structured thinking", 
            "optimal_llms": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V2"],
            "temperature": "0.1-0.4 (high precision)",
            "specialization": "Systematic problem breakdown, evidence-based conclusions",
            "parallel": "✅ (can run with other functions)",
            "use_case": "Determining cultural origin from pottery fragment analysis"
        },
        
        "🎨 Creativity": {
            "description": "systematic idea generation, artistic thinking, innovation",
            "optimal_llms": ["GPT-4o", "Claude-3.5-Sonnet", "DeepSeek-V2"],
            "temperature": "0.7-1.0 (high creativity)",
            "specialization": "Unconventional solutions, unexpected connections",
            "parallel": "✅ (independent creative process)",
            "use_case": "Brainstorming systematic underwater site preservation methods"
        },
        
        "🌍 Cultural Intelligence": {
            "description": "Cultural sensitivity, ethical considerations, indigenous rights",
            "optimal_llms": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V2"],
            "temperature": "0.4-0.7 (balanced approach)",
            "specialization": "Cultural context, ethical implications, respect protocols",
            "parallel": "✅ (can inform other functions)",
            "use_case": "Assessing cultural sensitivity of sacred burial ground excavation"
        },
        
        "🏛️ Archaeological Domain": {
            "description": "Archaeological expertise, cultural heritage, historical analysis",
            "optimal_llms": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V2"],
            "temperature": "0.3-0.6 (domain-focused)",
            "specialization": "Methodological precision, historical context, preservation",
            "parallel": "✅ (domain-specific processing)",
            "use_case": "Developing preservation plan for newly discovered Mayan artifacts"
        },
        
        "⚡ Execution": {
            "description": "Action selection, motor control, implementation",
            "optimal_llms": ["BitNet-1.58B", "DeepSeek-V2", "GPT-4o-mini"],
            "temperature": "0.1-0.4 (high precision)",
            "specialization": "Fast inference, precise action selection, real-time decisions",
            "parallel": "✅ (independent execution)",
            "use_case": "Generating precise drone survey coordinates for archaeological site"
        }
    }
    
    for function_name, details in cognitive_functions.items():
        print(f"{function_name}")
        print(f"   📝 Description: {details['description']}")
        print(f"   🤖 Optimal LLMs: {details['optimal_llms']}")
        print(f"   🌡️  Temperature: {details['temperature']}")
        print(f"   🎯 Specialization: {details['specialization']}")
        print(f"   🔄 Parallel: {details['parallel']}")
        print(f"   💡 Use Case: {details['use_case']}")
        print()
    
    print("🎵 === Orchestra Benefits ===")
    print()
    
    benefits = [
        "🎼 Specialized Excellence: Each LLM efficient for specific cognitive functions",
        "⚡ Computational Efficiency: Use expensive models only where needed",
        "🔄 Parallel Processing: Multiple functions can run simultaneously",
        "🎯 Quality Optimization: Right temperature and tokens for each task",
        "🛡️  Fallback Strategy: Graceful degradation when providers unavailable",
        "📊 Performance Monitoring: Track harmony and coordination between functions",
        "🌍 Cultural Intelligence: Specialized ethical and cultural reasoning",
        "🏛️ Domain Expertise: Archaeological knowledge integrated at architecture level"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print()
    print("🎵 === Example Orchestra Coordination ===")
    print()
    
    example_scenario = """
🎼 Scenario: Evaluating a newly discovered archaeological site

1. 🔍 REASONING (Claude-3.5-Sonnet, temp=0.2)
   → Analyzes site characteristics, dating evidence, structural patterns
   
2. 🌍 CULTURAL (Claude-3.5-Sonnet, temp=0.5) [PARALLEL]
   → Assesses cultural significance, indigenous rights, ethical considerations
   
3. 🏛️ ARCHAEOLOGICAL (GPT-4o, temp=0.4) [PARALLEL]  
   → Applies domain expertise, preservation protocols, methodology
   
4. 🎨 CREATIVITY (GPT-4o, temp=0.8) [PARALLEL]
   → Generates systematic documentation and preservation approaches
   
5. 🧠 CONSCIOUSNESS (Claude-3.5-Sonnet, temp=0.6)
   → Meta-analyzes the decision process, checks for biases
   
6. ⚡ EXECUTION (BitNet-1.58B, temp=0.2)
   → Generates precise action plan, coordinates, resource allocation

🎵 Result: Comprehensive, culturally-sensitive, methodologically-sound site evaluation
   with systematic preservation approaches and bias-checked decision making.
"""
    
    print(example_scenario)
    
    print("🎵 === Competitive Advantage ===")
    print()
    
    advantages = [
        "🆚 Traditional Scaling: 'Bigger model' vs 'Smarter orchestration'",
        "💰 Cost Efficiency: Use expensive models only for complex reasoning",
        "⚡ Speed: Fast models for execution, slow models for deep thinking", 
        "🎯 Quality: Each function gets optimal configuration",
        "🔄 Scalability: Add new cognitive functions without rebuilding",
        "🛡️  Reliability: Fallback strategies and graceful degradation",
        "🌍 Ethics: Cultural intelligence built into architecture",
        "🏛️ Domain: Archaeological expertise as first-class cognitive function"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print()
    print("🎼 === Implementation Status ===")
    print()
    print("✅ LLM Manager: Multi-provider support (OpenAI, Anthropic, DeepSeek, BitNet)")
    print("✅ Agent Architecture: Modular cognitive functions")
    print("🔄 Cognitive Orchestra: Enhanced specialization system")
    print("🔄 Configuration: Cognitive function profiles and optimization")
    print("🔄 Parallel Processing: Coordinated multi-function execution")
    print("🔄 Performance Monitoring: Harmony scoring and metrics")
    print()
    print("🎵 Ready for: Enhanced multi-LLM cognitive specialization!")


if __name__ == "__main__":
    asyncio.run(demonstrate_cognitive_orchestra_concept()) 