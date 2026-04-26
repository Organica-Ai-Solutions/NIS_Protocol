#!/usr/bin/env python3
"""
Integration Test - Verify All Systems Wired Up Correctly
Tests that all speed optimizations and AI/ML systems are properly integrated

Copyright 2025 Organica AI Solutions
"""

import asyncio
import sys


async def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Import Verification")
    print("="*60)
    
    try:
        # Core systems
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        from src.core.llm_planner import get_llm_planner
        from src.core.mcp_tool_executor import get_mcp_executor
        
        # Speed optimization systems
        from src.core.predict_prefetch import get_predict_prefetch_engine
        from src.core.backup_agents import get_backup_agent_executor
        from src.core.agent_competition import get_agent_competition_system
        from src.core.branching_strategies import get_branching_strategies_system
        
        # AI/ML systems
        from src.core.ml_prediction_engine import get_ml_prediction_engine
        from src.core.llm_judge import get_llm_judge
        from src.core.multi_critic_review import get_multi_critic_review_system
        from src.core.pipeline_processor import get_pipeline_processor
        from src.core.shared_workspace import get_shared_workspace
        
        # Bonus features
        from src.tools.database_query import get_database_query_tool
        from src.memory.rag_memory import get_rag_memory_system
        from src.agents.multi_agent_negotiation import get_multi_agent_negotiator
        from src.integration.consciousness_autonomous_bridge import get_consciousness_autonomous_bridge
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_orchestrator_initialization():
    """Test orchestrator initialization with all systems."""
    print("\n" + "="*60)
    print("TEST 2: Orchestrator Initialization")
    print("="*60)
    
    try:
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        
        # Test with all optimizations enabled
        print("Initializing with all systems enabled...")
        orchestrator = AutonomousOrchestrator(
            llm_provider=None,  # Will use mock for testing
            enable_speed_optimizations=True,
            enable_ai_enhancements=True
        )
        
        # Verify components exist
        assert orchestrator.mcp_executor is not None, "MCP executor missing"
        assert orchestrator.llm_planner is not None, "LLM planner missing"
        assert orchestrator.parallel_executor is not None, "Parallel executor missing"
        assert orchestrator.streaming_executor is not None, "Streaming executor missing"
        
        # Verify speed optimization systems
        if orchestrator.enable_speed_optimizations:
            assert orchestrator.backup_executor is not None, "Backup executor missing"
            assert orchestrator.branching_system is not None, "Branching system missing"
            print("✅ Speed optimization systems initialized")
        
        # Verify AI/ML systems (will be None without real LLM provider)
        print("✅ Orchestrator initialized successfully!")
        
        # Check available tools
        tools = orchestrator.mcp_executor.get_available_tools()
        print(f"✅ {len(tools)} MCP tools available: {tools}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ai_ml_systems():
    """Test AI/ML system initialization."""
    print("\n" + "="*60)
    print("TEST 3: AI/ML Systems")
    print("="*60)
    
    try:
        # Test ML prediction engine
        from src.core.ml_prediction_engine import MLPredictionEngine
        print("✅ ML Prediction Engine class available")
        
        # Test LLM judge
        from src.core.llm_judge import LLMJudge
        print("✅ LLM Judge class available")
        
        # Test multi-critic review
        from src.core.multi_critic_review import MultiCriticReviewSystem
        print("✅ Multi-Critic Review class available")
        
        # Test pipeline processor
        from src.core.pipeline_processor import PipelineProcessor
        print("✅ Pipeline Processor class available")
        
        # Test shared workspace
        from src.core.shared_workspace import SharedWorkspace
        print("✅ Shared Workspace class available")
        
        print("✅ All AI/ML systems available!")
        return True
        
    except Exception as e:
        print(f"❌ AI/ML systems test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bonus_features():
    """Test bonus features."""
    print("\n" + "="*60)
    print("TEST 4: Bonus Features")
    print("="*60)
    
    try:
        # Test database query tool
        from src.tools.database_query import DatabaseQueryTool
        print("✅ Database Query Tool available")
        
        # Test RAG memory
        from src.memory.rag_memory import RAGMemorySystem
        print("✅ RAG Memory System available")
        
        # Test multi-agent negotiation
        from src.agents.multi_agent_negotiation import MultiAgentNegotiator
        print("✅ Multi-Agent Negotiator available")
        
        # Test consciousness bridge
        from src.integration.consciousness_autonomous_bridge import ConsciousnessAutonomousBridge
        print("✅ Consciousness Bridge available")
        
        print("✅ All bonus features available!")
        return True
        
    except Exception as e:
        print(f"❌ Bonus features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_stats():
    """Test that stats methods work."""
    print("\n" + "="*60)
    print("TEST 5: System Statistics")
    print("="*60)
    
    try:
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        
        orchestrator = AutonomousOrchestrator(
            llm_provider=None,
            enable_speed_optimizations=True,
            enable_ai_enhancements=False  # Skip AI for this test
        )
        
        # Test backup executor stats
        if orchestrator.backup_executor:
            stats = orchestrator.backup_executor.get_stats()
            print(f"✅ Backup executor stats: {stats}")
        
        # Test parallel executor stats
        stats = orchestrator.parallel_executor.get_stats()
        print(f"✅ Parallel executor stats: {stats}")
        
        print("✅ All stats methods working!")
        return True
        
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "🚀"*30)
    print("INTEGRATION TEST SUITE")
    print("Verifying all systems are properly wired up")
    print("🚀"*30)
    
    results = {
        "imports": await test_imports(),
        "orchestrator": await test_orchestrator_initialization(),
        "ai_ml_systems": await test_ai_ml_systems(),
        "bonus_features": await test_bonus_features(),
        "stats": await test_system_stats()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests PASSED!")
        print("✅ System is properly wired up and ready to use")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) FAILED")
        print("❌ Some systems may not be properly integrated")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
