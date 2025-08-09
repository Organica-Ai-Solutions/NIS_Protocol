#!/bin/bash

echo "=== Testing Fixed NIS Protocol Endpoints ==="

echo ""
echo "1. Testing FIXED Metrics Endpoint:"
curl -s http://localhost/metrics

echo ""
echo ""
echo "2. Testing Learning Agent with VALID operation:"
curl -s -X POST http://localhost/agents/learning/process \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_params"}'

echo ""
echo ""
echo "3. Testing Agent Simulation with proper structure:"
curl -s -X POST http://localhost/agents/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "test_physics", 
    "scenario_type": "physics",
    "parameters": {
      "mass": 1.0,
      "height": 10.0,
      "gravity": 9.8
    }
  }'

echo ""
echo ""
echo "4. Testing Planning Agent:"
curl -s -X POST http://localhost/agents/planning/create_plan \
  -H "Content-Type: application/json" \
  -d '{"goal": "test physics validation"}'

echo ""
echo ""
echo "5. Testing Curiosity Agent:"
curl -s -X POST http://localhost/agents/curiosity/process_stimulus \
  -H "Content-Type: application/json" \
  -d '{"stimulus": "physics experiment"}'

echo ""
echo ""
echo "6. Testing Ethics Agent:"
curl -s -X POST http://localhost/agents/alignment/evaluate_ethics \
  -H "Content-Type: application/json" \
  -d '{"scenario": "AI decision making", "context": "scientific research"}'