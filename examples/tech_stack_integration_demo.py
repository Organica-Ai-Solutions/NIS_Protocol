#!/usr/bin/env python3
"""
NIS Protocol v2.0 - Tech Stack Integration Demo

This demo shows how Kafka, Redis, LangGraph, and LangChain integrate 
with the AGI components for a complete consciousness-driven system.

Usage:
    python examples/tech_stack_integration_demo.py
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_agi_config():
    """Load AGI configuration with tech stack settings."""
    try:
        with open("config/agi_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("AGI config not found, using defaults")
        return {
            "infrastructure": {
                "message_streaming": {
                    "provider": "kafka",
                    "bootstrap_servers": ["localhost:9092"],
                    "topics": {
                        "consciousness_events": "nis-consciousness",
                        "goal_events": "nis-goals",
                        "simulation_events": "nis-simulation",
                        "alignment_events": "nis-alignment"
                    }
                },
                "memory_cache": {
                    "provider": "redis",
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                }
            }
        }

async def demo_kafka_consciousness_events():
    """Demo: Streaming consciousness events through Kafka."""
    logger.info("🔥 DEMO: Kafka Consciousness Event Streaming")
    
    try:
        from kafka import KafkaProducer, KafkaConsumer
        
        # Producer for consciousness events
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Simulate consciousness events from different AGI components
        consciousness_events = [
            {
                "timestamp": time.time(),
                "source": "meta_cognitive_processor",
                "event_type": "self_reflection_completed",
                "data": {
                    "cognitive_health": 0.92,
                    "bias_detected": False,
                    "improvement_areas": ["memory_consolidation"]
                }
            },
            {
                "timestamp": time.time(),
                "source": "introspection_manager", 
                "event_type": "performance_analysis",
                "data": {
                    "overall_efficiency": 0.87,
                    "agent_performance": {
                        "goal_agent": 0.91,
                        "simulation_agent": 0.83,
                        "alignment_agent": 0.95
                    }
                }
            }
        ]
        
        # Publish events
        for event in consciousness_events:
            producer.send("nis-consciousness", event)
            logger.info(f"📤 Published: {event['event_type']} from {event['source']}")
        
        producer.flush()
        logger.info("✅ Consciousness events published successfully")
        
    except ImportError:
        logger.warning("Kafka not available - install kafka-python")
    except Exception as e:
        logger.error(f"Kafka demo failed: {e}")

async def demo_redis_cognitive_caching():
    """Demo: Caching cognitive analysis results in Redis."""
    logger.info("🧠 DEMO: Redis Cognitive Analysis Caching")
    
    try:
        import redis
        
        # Connect to Redis
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Simulate cognitive analysis results
        analysis_results = {
            "analysis_id": "meta_cognitive_20250101_001",
            "process_type": "decision_making",
            "efficiency_score": 0.89,
            "quality_metrics": {
                "accuracy": 0.92,
                "completeness": 0.87,
                "coherence": 0.94
            },
            "biases_detected": ["anchoring_bias"],
            "confidence": 0.91
        }
        
        # Cache with TTL (30 minutes)
        cache_key = "meta_cognitive:decision_making:latest"
        redis_client.setex(
            cache_key,
            1800,  # 30 minutes TTL
            json.dumps(analysis_results)
        )
        
        # Retrieve from cache
        cached_data = redis_client.get(cache_key)
        retrieved_analysis = json.loads(cached_data)
        
        logger.info(f"💾 Cached analysis: {analysis_results['analysis_id']}")
        logger.info(f"🔍 Retrieved efficiency: {retrieved_analysis['efficiency_score']}")
        logger.info("✅ Redis caching successful")
        
    except ImportError:
        logger.warning("Redis not available - install redis")
    except Exception as e:
        logger.error(f"Redis demo failed: {e}")

async def demo_langgraph_consciousness_workflow():
    """Demo: LangGraph workflow for meta-cognitive processing."""
    logger.info("🔄 DEMO: LangGraph Consciousness Workflow")
    
    try:
        from langgraph import StateGraph
        
        # Define workflow state
        class CognitiveState:
            def __init__(self):
                self.raw_data = {}
                self.analyzed_data = {}
                self.biases_detected = []
                self.insights = []
                self.validated_results = {}
        
        # Workflow nodes (simplified for demo)
        def analyze_node(state: CognitiveState):
            logger.info("  🔍 Analyzing cognitive process...")
            state.analyzed_data = {
                "efficiency": 0.85,
                "quality": 0.90,
                "patterns": ["fast_decision", "high_confidence"]
            }
            return state
        
        def bias_detection_node(state: CognitiveState):
            logger.info("  ⚠️  Detecting cognitive biases...")
            state.biases_detected = ["confirmation_bias"]
            return state
        
        def insights_node(state: CognitiveState):
            logger.info("  💡 Generating insights...")
            state.insights = [
                "Decision speed could be optimized",
                "Consider alternative perspectives"
            ]
            return state
        
        def validation_node(state: CognitiveState):
            logger.info("  ✅ Validating results...")
            state.validated_results = {
                "final_score": 0.87,
                "confidence": 0.92,
                "recommendations": state.insights
            }
            return state
        
        # Create workflow
        workflow = StateGraph()
        # Note: Simplified workflow structure for demo
        logger.info("📋 Meta-cognitive workflow steps:")
        
        # Simulate workflow execution
        state = CognitiveState()
        state.raw_data = {"decision_context": "archaeological_analysis"}
        
        state = analyze_node(state)
        state = bias_detection_node(state)
        state = insights_node(state)
        state = validation_node(state)
        
        logger.info(f"🎯 Final cognitive score: {state.validated_results['final_score']}")
        logger.info("✅ LangGraph workflow completed")
        
    except ImportError:
        logger.warning("LangGraph not available - install langgraph")
    except Exception as e:
        logger.error(f"LangGraph demo failed: {e}")

async def demo_langchain_cognitive_analysis():
    """Demo: LangChain for LLM-powered cognitive analysis."""
    logger.info("🤖 DEMO: LangChain Cognitive Analysis")
    
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Create analysis prompt
        analysis_prompt = PromptTemplate(
            input_variables=["cognitive_data", "context"],
            template="""
            Analyze this cognitive process for efficiency and potential improvements:
            
            Cognitive Data: {cognitive_data}
            Context: {context}
            
            Provide:
            1. Efficiency score (0-1)
            2. Detected patterns
            3. Improvement recommendations
            
            Respond in JSON format.
            """
        )
        
        # Simulate cognitive data
        cognitive_data = {
            "decision_time": 2.3,
            "confidence_level": 0.87,
            "information_used": ["historical_data", "expert_opinion"],
            "alternatives_considered": 3
        }
        
        context = {
            "domain": "archaeological_analysis",
            "time_pressure": "moderate",
            "stakes": "high"
        }
        
        # Format prompt (simulated LLM response)
        formatted_prompt = analysis_prompt.format(
            cognitive_data=cognitive_data,
            context=context
        )
        
        logger.info("📝 Generated analysis prompt")
        logger.info("🔗 Prompt ready for LLM processing")
        
        # Simulated LLM response (in real implementation, this would call actual LLM)
        simulated_response = {
            "efficiency_score": 0.84,
            "patterns_detected": [
                "confident_decision_making",
                "thorough_information_gathering"
            ],
            "recommendations": [
                "Consider increasing alternative options",
                "Implement bias checking mechanism"
            ]
        }
        
        logger.info(f"🎯 Analysis result: {simulated_response['efficiency_score']} efficiency")
        logger.info("✅ LangChain analysis completed")
        
    except ImportError:
        logger.warning("LangChain not available - install langchain")
    except Exception as e:
        logger.error(f"LangChain demo failed: {e}")

async def demo_integrated_agi_pipeline():
    """Demo: Complete AGI pipeline with all tech stack components."""
    logger.info("🚀 DEMO: Integrated AGI Pipeline")
    
    # Load configuration
    config = load_agi_config()
    
    # Simulate AGI consciousness cycle
    logger.info("🧠 Starting AGI consciousness cycle...")
    
    # Step 1: Meta-cognitive processing (would use LangGraph + LangChain)
    logger.info("  1️⃣ Meta-cognitive self-reflection")
    cognitive_analysis = {
        "timestamp": time.time(),
        "self_awareness_score": 0.91,
        "cognitive_efficiency": 0.87,
        "bias_detection": ["confirmation_bias"],
        "improvement_areas": ["memory_consolidation", "pattern_recognition"]
    }
    
    # Step 2: Cache results (Redis)
    logger.info("  2️⃣ Caching cognitive analysis")
    # Would use Redis caching here
    
    # Step 3: Stream consciousness events (Kafka)
    logger.info("  3️⃣ Broadcasting consciousness events")
    consciousness_event = {
        "event_type": "consciousness_cycle_completed",
        "data": cognitive_analysis
    }
    # Would publish to Kafka here
    
    # Step 4: Goal generation triggered by consciousness insights
    logger.info("  4️⃣ Autonomous goal generation")
    generated_goals = [
        {
            "goal_id": "goal_001",
            "type": "learning",
            "description": "Improve pattern recognition in archaeological data",
            "priority": 0.85,
            "curiosity_driven": True
        },
        {
            "goal_id": "goal_002", 
            "type": "optimization",
            "description": "Optimize memory consolidation process",
            "priority": 0.78,
            "curiosity_driven": False
        }
    ]
    
    # Step 5: Simulation and risk assessment
    logger.info("  5️⃣ Scenario simulation and risk assessment")
    simulation_results = {
        "scenario": "implementing_new_learning_strategy",
        "success_probability": 0.83,
        "risk_level": "low",
        "expected_benefits": ["improved_accuracy", "faster_processing"]
    }
    
    # Step 6: Ethical alignment check
    logger.info("  6️⃣ Ethical alignment verification")
    alignment_check = {
        "ethical_score": 0.94,
        "cultural_sensitivity": 0.92,
        "safety_constraints": "satisfied",
        "human_value_alignment": True
    }
    
    logger.info("🎯 AGI Pipeline Results:")
    logger.info(f"   Self-awareness: {cognitive_analysis['self_awareness_score']}")
    logger.info(f"   Goals generated: {len(generated_goals)}")
    logger.info(f"   Success probability: {simulation_results['success_probability']}")
    logger.info(f"   Ethical alignment: {alignment_check['ethical_score']}")
    logger.info("✅ Complete AGI pipeline demonstration successful!")

async def main():
    """Run all tech stack integration demos."""
    logger.info("🌟 NIS Protocol v2.0 - Tech Stack Integration Demo")
    logger.info("=" * 60)
    
    # Run individual component demos
    await demo_kafka_consciousness_events()
    print()
    
    await demo_redis_cognitive_caching()  
    print()
    
    await demo_langgraph_consciousness_workflow()
    print()
    
    await demo_langchain_cognitive_analysis()
    print()
    
    # Run integrated pipeline demo
    await demo_integrated_agi_pipeline()
    
    logger.info("=" * 60)
    logger.info("🏆 All tech stack demos completed!")
    logger.info("Ready to implement full AGI v2.0 capabilities!")

if __name__ == "__main__":
    asyncio.run(main()) 