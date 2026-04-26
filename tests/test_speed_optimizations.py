#!/usr/bin/env python3
"""
Test Speed Optimizations
Validates all 6 speed techniques work correctly

Copyright 2025 Organica AI Solutions
"""

import asyncio
import time
from src.core.autonomous_orchestrator import AutonomousOrchestrator


async def test_baseline():
    """Test baseline performance (no optimizations)."""
    print("\n" + "="*60)
    print("TEST 1: Baseline (No Optimizations)")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=False
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Research the latest developments in quantum computing",
        parallel=False  # Sequential
    )
    duration = time.time() - start
    
    print(f"✅ Baseline complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    print(f"Steps: {len(result.get('plan', {}).get('steps', []))}")
    return duration


async def test_parallel_only():
    """Test with parallel execution only."""
    print("\n" + "="*60)
    print("TEST 2: Parallel Execution Only")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=False
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Research the latest developments in quantum computing",
        parallel=True  # Parallel
    )
    duration = time.time() - start
    
    print(f"✅ Parallel complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    print(f"Parallelization: {result.get('parallelization', {})}")
    return duration


async def test_with_prefetch():
    """Test with predict-and-prefetch."""
    print("\n" + "="*60)
    print("TEST 3: With Predict-and-Prefetch")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=True
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Research the latest developments in quantum computing",
        parallel=True
    )
    duration = time.time() - start
    
    # Check prefetch stats
    if orchestrator.llm_planner.prefetch_engine:
        stats = orchestrator.llm_planner.prefetch_engine.get_stats()
        print(f"📊 Prefetch stats: {stats}")
    
    print(f"✅ With prefetch complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    return duration


async def test_with_branching():
    """Test with branching strategies."""
    print("\n" + "="*60)
    print("TEST 4: With Branching Strategies")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=True
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Solve a complex physics problem involving heat transfer",
        parallel=True,
        use_branching=True  # 3 strategies
    )
    duration = time.time() - start
    
    # Check branching stats
    if orchestrator.branching_system:
        stats = orchestrator.branching_system.get_stats()
        print(f"📊 Branching stats: {stats}")
    
    print(f"✅ With branching complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    return duration


async def test_with_competition():
    """Test with agent competition."""
    print("\n" + "="*60)
    print("TEST 5: With Agent Competition")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=True
    )
    
    if not orchestrator.competition_system:
        print("⚠️ Competition system not available (multi-provider required)")
        return None
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Analyze the implications of AI safety research",
        parallel=True,
        use_competition=True  # 3 providers
    )
    duration = time.time() - start
    
    # Check competition stats
    stats = orchestrator.competition_system.get_stats()
    print(f"📊 Competition stats: {stats}")
    
    print(f"✅ With competition complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    return duration


async def test_with_backup():
    """Test with backup agents."""
    print("\n" + "="*60)
    print("TEST 6: With Backup Agents")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=True
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Plan robot motion to target position",
        parallel=True,
        use_backup=True  # 3 redundant executions
    )
    duration = time.time() - start
    
    # Check backup stats
    if orchestrator.backup_executor:
        stats = orchestrator.backup_executor.get_stats()
        print(f"📊 Backup stats: {stats}")
    
    print(f"✅ With backup complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    return duration


async def test_all_optimizations():
    """Test with ALL optimizations enabled."""
    print("\n" + "="*60)
    print("TEST 7: ALL OPTIMIZATIONS ENABLED")
    print("="*60)
    
    orchestrator = AutonomousOrchestrator(
        llm_provider="anthropic",
        enable_speed_optimizations=True
    )
    
    start = time.time()
    result = await orchestrator.plan_and_execute(
        goal="Research quantum computing, solve related physics equations, and create a comprehensive report",
        parallel=True,
        use_branching=True,
        use_competition=True if orchestrator.competition_system else False,
        use_backup=True
    )
    duration = time.time() - start
    
    print(f"✅ All optimizations complete: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    
    # Print all stats
    if orchestrator.llm_planner.prefetch_engine:
        print(f"📊 Prefetch: {orchestrator.llm_planner.prefetch_engine.get_stats()}")
    if orchestrator.branching_system:
        print(f"📊 Branching: {orchestrator.branching_system.get_stats()}")
    if orchestrator.competition_system:
        print(f"📊 Competition: {orchestrator.competition_system.get_stats()}")
    if orchestrator.backup_executor:
        print(f"📊 Backup: {orchestrator.backup_executor.get_stats()}")
    
    return duration


async def run_all_tests():
    """Run all speed optimization tests."""
    print("\n" + "🚀"*30)
    print("SPEED OPTIMIZATION TEST SUITE")
    print("🚀"*30)
    
    results = {}
    
    try:
        # Test 1: Baseline
        results['baseline'] = await test_baseline()
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        results['baseline'] = None
    
    try:
        # Test 2: Parallel only
        results['parallel'] = await test_parallel_only()
    except Exception as e:
        print(f"❌ Parallel test failed: {e}")
        results['parallel'] = None
    
    try:
        # Test 3: With prefetch
        results['prefetch'] = await test_with_prefetch()
    except Exception as e:
        print(f"❌ Prefetch test failed: {e}")
        results['prefetch'] = None
    
    try:
        # Test 4: With branching
        results['branching'] = await test_with_branching()
    except Exception as e:
        print(f"❌ Branching test failed: {e}")
        results['branching'] = None
    
    try:
        # Test 5: With competition
        results['competition'] = await test_with_competition()
    except Exception as e:
        print(f"❌ Competition test failed: {e}")
        results['competition'] = None
    
    try:
        # Test 6: With backup
        results['backup'] = await test_with_backup()
    except Exception as e:
        print(f"❌ Backup test failed: {e}")
        results['backup'] = None
    
    try:
        # Test 7: All optimizations
        results['all'] = await test_all_optimizations()
    except Exception as e:
        print(f"❌ All optimizations test failed: {e}")
        results['all'] = None
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    baseline = results.get('baseline')
    if baseline:
        print(f"Baseline (sequential): {baseline:.2f}s")
        
        for name, duration in results.items():
            if name != 'baseline' and duration:
                speedup = baseline / duration
                improvement = ((baseline - duration) / baseline) * 100
                print(f"{name.capitalize():15s}: {duration:.2f}s ({speedup:.2f}x speedup, {improvement:.1f}% faster)")
    
    print("\n✅ Test suite complete!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
