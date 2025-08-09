#!/bin/bash

echo "=== Testing Additional NIS Protocol Endpoints ==="

echo ""
echo "7. Testing Metrics Endpoint:"
curl -s http://localhost/metrics

echo ""
echo ""
echo "8. Testing Docs Endpoint:"
curl -s http://localhost/docs | head -5

echo ""
echo ""
echo "9. Testing Agent Learning:"
curl -s -X POST http://localhost/agents/learning/process \
  -H "Content-Type: application/json" \
  -d '{"operation": "status"}'

echo ""
echo ""
echo "10. Testing Agent Reasoning:"
curl -s -X POST http://localhost/agents/reasoning/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "test physics", "domain": "physics"}'

echo ""
echo ""
echo "11. Testing Agent Physics Validation:"
curl -s -X POST http://localhost/agents/physics/validate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "falling ball"}'

echo ""
echo ""
echo "12. Testing Agent Memory:"
curl -s -X POST http://localhost/agents/memory/store \
  -H "Content-Type: application/json" \
  -d '{"key": "test", "data": {"value": "test"}}'