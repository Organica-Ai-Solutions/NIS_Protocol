#!/usr/bin/env python3
"""
Multi-Agent Scaling Benchmark for DGX Cloud

Tests coordination overhead with 1-100 concurrent agents.
Validates NIS-HUB can handle swarm-scale operations on H100.

Usage:
    python benchmark_multi_agent.py --agents 50
    python benchmark_multi_agent.py --agents 100 --parallel
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "multi_agent"


class SimulatedAgent:
    """Simulated agent for benchmarking coordination overhead"""
    
    def __init__(self, agent_id: int, state_dim: int = 64):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.state = np.random.randn(state_dim)
        self.model_weights = np.random.randn(state_dim, state_dim) * 0.1
        
    def perceive(self, environment: np.ndarray) -> np.ndarray:
        """Simulate perception/sensor fusion"""
        return np.tanh(environment @ self.model_weights[:len(environment), :])
    
    def decide(self, perception: np.ndarray, consensus: np.ndarray) -> np.ndarray:
        """Simulate decision making with consensus input"""
        combined = np.concatenate([perception[:32], consensus[:32]])
        decision = np.tanh(combined @ self.model_weights[:64, :32])
        return decision
    
    def act(self, decision: np.ndarray) -> Dict[str, Any]:
        """Simulate action execution"""
        action_vector = decision[:6]  # position + orientation
        return {
            "agent_id": self.agent_id,
            "action": action_vector.tolist(),
            "confidence": float(np.abs(decision).mean())
        }


class SwarmCoordinator:
    """Coordinates multiple agents with consensus mechanism"""
    
    def __init__(self, num_agents: int, state_dim: int = 64):
        self.agents = [SimulatedAgent(i, state_dim) for i in range(num_agents)]
        self.state_dim = state_dim
        self.consensus_state = np.zeros(state_dim)
        
    def compute_consensus(self, agent_states: List[np.ndarray]) -> np.ndarray:
        """Compute swarm consensus from all agent states"""
        if not agent_states:
            return np.zeros(self.state_dim)
        
        # Weighted average (could be more sophisticated)
        consensus = np.mean(agent_states, axis=0)
        return consensus
    
    def step(self, environment: np.ndarray) -> List[Dict[str, Any]]:
        """Execute one coordination step for all agents"""
        # 1. All agents perceive
        perceptions = [agent.perceive(environment) for agent in self.agents]
        
        # 2. Compute consensus
        self.consensus_state = self.compute_consensus(perceptions)
        
        # 3. All agents decide based on perception + consensus
        decisions = [
            agent.decide(perc, self.consensus_state) 
            for agent, perc in zip(self.agents, perceptions)
        ]
        
        # 4. All agents act
        actions = [agent.act(dec) for agent, dec in zip(self.agents, decisions)]
        
        return actions


async def benchmark_sequential(num_agents: int, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark sequential agent coordination"""
    logger.info(f"📊 Benchmarking {num_agents} agents (sequential)...")
    
    coordinator = SwarmCoordinator(num_agents)
    environment = np.random.randn(64)
    
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        actions = coordinator.step(environment)
        latencies.append((time.perf_counter() - start) * 1000)
        
        # Update environment based on actions
        environment = environment + np.random.randn(64) * 0.1
    
    return {
        "mode": "sequential",
        "num_agents": num_agents,
        "iterations": iterations,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_steps_per_sec": 1000 / np.mean(latencies),
        "agent_decisions_per_sec": num_agents * 1000 / np.mean(latencies)
    }


async def benchmark_parallel_threads(num_agents: int, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark parallel agent coordination using threads"""
    logger.info(f"📊 Benchmarking {num_agents} agents (parallel threads)...")
    
    agents = [SimulatedAgent(i) for i in range(num_agents)]
    environment = np.random.randn(64)
    
    def agent_step(agent: SimulatedAgent, env: np.ndarray, consensus: np.ndarray):
        perception = agent.perceive(env)
        decision = agent.decide(perception, consensus)
        return agent.act(decision)
    
    latencies = []
    with ThreadPoolExecutor(max_workers=min(num_agents, 32)) as executor:
        for i in range(iterations):
            consensus = np.zeros(64)
            
            start = time.perf_counter()
            
            # Parallel agent execution
            futures = [
                executor.submit(agent_step, agent, environment, consensus)
                for agent in agents
            ]
            actions = [f.result() for f in futures]
            
            latencies.append((time.perf_counter() - start) * 1000)
            
            # Update environment
            environment = environment + np.random.randn(64) * 0.1
    
    return {
        "mode": "parallel_threads",
        "num_agents": num_agents,
        "iterations": iterations,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_steps_per_sec": 1000 / np.mean(latencies),
        "agent_decisions_per_sec": num_agents * 1000 / np.mean(latencies)
    }


async def benchmark_gpu_batch(num_agents: int, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark GPU-batched agent coordination"""
    logger.info(f"📊 Benchmarking {num_agents} agents (GPU batch)...")
    
    try:
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"   Using device: {device}")
        
        # Batched agent model
        state_dim = 64
        agent_states = torch.randn(num_agents, state_dim, device=device)
        model_weights = torch.randn(state_dim, state_dim, device=device) * 0.1
        environment = torch.randn(state_dim, device=device)
        
        latencies = []
        for i in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            # Batched perception
            perceptions = torch.tanh(agent_states @ model_weights)
            
            # Batched consensus
            consensus = perceptions.mean(dim=0)
            
            # Batched decision
            combined = torch.cat([
                perceptions[:, :32],
                consensus[:32].unsqueeze(0).expand(num_agents, -1)
            ], dim=1)
            decisions = torch.tanh(combined @ model_weights[:64, :32])
            
            # Batched action
            actions = decisions[:, :6]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)
            
            # Update states
            agent_states = agent_states + torch.randn_like(agent_states) * 0.1
        
        return {
            "mode": "gpu_batch",
            "device": str(device),
            "num_agents": num_agents,
            "iterations": iterations,
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_steps_per_sec": 1000 / np.mean(latencies),
            "agent_decisions_per_sec": num_agents * 1000 / np.mean(latencies)
        }
        
    except ImportError:
        logger.warning("⚠️ PyTorch not available for GPU benchmark")
        return {"mode": "gpu_batch", "error": "PyTorch not available"}


async def run_scaling_benchmark(max_agents: int = 100) -> List[Dict[str, Any]]:
    """Run scaling benchmark across different agent counts"""
    agent_counts = [1, 5, 10, 25, 50, 75, 100]
    agent_counts = [n for n in agent_counts if n <= max_agents]
    
    results = []
    
    for num_agents in agent_counts:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {num_agents} agents")
        logger.info(f"{'='*40}")
        
        # Sequential
        seq_result = await benchmark_sequential(num_agents, iterations=50)
        results.append(seq_result)
        
        # Parallel threads
        par_result = await benchmark_parallel_threads(num_agents, iterations=50)
        results.append(par_result)
        
        # GPU batch
        gpu_result = await benchmark_gpu_batch(num_agents, iterations=50)
        results.append(gpu_result)
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print benchmark summary"""
    print("\n" + "=" * 80)
    print("📊 MULTI-AGENT SCALING BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by agent count
    by_agents = {}
    for r in results:
        if "error" in r:
            continue
        n = r["num_agents"]
        if n not in by_agents:
            by_agents[n] = {}
        by_agents[n][r["mode"]] = r
    
    print(f"\n{'Agents':<10} {'Sequential':<20} {'Parallel':<20} {'GPU Batch':<20}")
    print(f"{'':10} {'(ms)':<20} {'(ms)':<20} {'(ms)':<20}")
    print("-" * 70)
    
    for n in sorted(by_agents.keys()):
        modes = by_agents[n]
        seq = modes.get("sequential", {}).get("avg_latency_ms", "N/A")
        par = modes.get("parallel_threads", {}).get("avg_latency_ms", "N/A")
        gpu = modes.get("gpu_batch", {}).get("avg_latency_ms", "N/A")
        
        seq_str = f"{seq:.2f}" if isinstance(seq, float) else seq
        par_str = f"{par:.2f}" if isinstance(par, float) else par
        gpu_str = f"{gpu:.2f}" if isinstance(gpu, float) else gpu
        
        print(f"{n:<10} {seq_str:<20} {par_str:<20} {gpu_str:<20}")
    
    print("\n🎯 Target for H100: <1ms for 50 agents with GPU batching")


async def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Scaling Benchmark")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per test")
    parser.add_argument("--parallel", action="store_true", help="Include parallel benchmarks")
    parser.add_argument("--scaling", action="store_true", help="Run full scaling test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 NIS Protocol - Multi-Agent Scaling Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.scaling:
        results = await run_scaling_benchmark(args.agents)
    else:
        results = []
        results.append(await benchmark_sequential(args.agents, args.iterations))
        
        if args.parallel:
            results.append(await benchmark_parallel_threads(args.agents, args.iterations))
        
        results.append(await benchmark_gpu_batch(args.agents, args.iterations))
    
    # Save results
    results_file = OUTPUT_DIR / f"benchmark_{args.agents}_agents.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "results": results
        }, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_file}")
    
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
