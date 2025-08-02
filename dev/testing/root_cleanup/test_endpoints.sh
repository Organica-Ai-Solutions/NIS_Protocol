#!/bin/bash

echo "=== Testing NIS Protocol Endpoints ==="

echo ""
echo "1. Testing Health Endpoint:"
curl -s http://localhost/health

echo ""
echo ""
echo "2. Testing Root Endpoint:" 
curl -s http://localhost/

echo ""
echo ""
echo "3. Testing Chat Endpoint:"
curl -s -X POST http://localhost/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you test the physics validation?"}'

echo ""
echo ""
echo "4. Testing Simulation Endpoint:"
curl -s -X POST http://localhost/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"concept": "test energy conservation"}'

echo ""
echo ""
echo "5. Testing Consciousness Status:"
curl -s http://localhost/consciousness/status

echo ""
echo ""
echo "6. Testing Infrastructure Status:"
curl -s http://localhost/infrastructure/status