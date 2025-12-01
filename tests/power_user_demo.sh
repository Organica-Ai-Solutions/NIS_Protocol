#!/bin/bash

BASE_URL="http://localhost:8007"
echo "🚀 STARTING NIS PROTOCOL POWER USER DEMO"
echo "========================================"

# Helper function for formatting
format_json() {
    if command -v jq &> /dev/null; then
        jq .
    else
        python3 -m json.tool
    fi
}

# 1. Genesis
echo ""
echo "🧬 1. GENESIS: Creating Planetary Exploration Agent..."
curl -s -X POST "$BASE_URL/v4/consciousness/genesis" \
  -H "Content-Type: application/json" \
  -d '{"capability": "planetary_exploration", "model_recommendation": "specialized"}' | format_json

# 2. Plan
echo ""
echo "🎯 2. PLAN: Designing Mission 'Survey Mars Sector 4'..."
curl -s -X POST "$BASE_URL/v4/consciousness/plan" \
  -H "Content-Type: application/json" \
  -d '{"goal_id": "mars_mission_01", "high_level_goal": "Survey Mars Sector 4 for water ice deposits"}' | format_json

# 3. Collective Decision
echo ""
echo "⚖️ 3. COLLECTIVE: Allocating energy resources..."
curl -s -X POST "$BASE_URL/v4/consciousness/collective/decide" \
  -H "Content-Type: application/json" \
  -d '{"problem": "Energy allocation for deep scan vs wide scan", "local_decision": {"priority": "deep_scan", "confidence": 0.85}}' | format_json

# 4. Multipath Reasoning
echo ""
echo "🌳 4. MULTIPATH: Resolving Sensor Ambiguity..."
curl -s -X POST "$BASE_URL/v4/consciousness/multipath/start" \
  -H "Content-Type: application/json" \
  -d '{"problem": "Spectrometer reading 14.2: Ice vs Dry Ice", "num_paths": 3}' | format_json

# 5. Ethics
echo ""
echo "🛡️ 5. ETHICS: Evaluating Drilling Risk..."
curl -s -X POST "$BASE_URL/v4/consciousness/ethics/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"decision_context": {"action": "drill_deep_core", "risk": "biological_contamination", "benefit": "scientific_discovery"}}' | format_json

# 6. Physics (PINN)
echo ""
echo "🔬 6. PHYSICS: Validating Drill Heat Dissipation (Heat Equation)..."
curl -s -X POST "$BASE_URL/physics/solve/heat-equation" \
  -H "Content-Type: application/json" \
  -d '{"thermal_diffusivity": 0.5, "domain_length": 10.0, "final_time": 5.0}' | format_json

# 7. Simulation
echo ""
echo "🌌 7. SIMULATION: Generative Simulation of Core Extraction..."
curl -s -X POST "$BASE_URL/simulation/run" \
  -H "Content-Type: application/json" \
  -d '{"concept": "Ice core extraction thermodynamics under martian gravity"}' | format_json

# 8. Vision
echo ""
echo "👁️ 8. VISION: Analyzing Terrain Image..."
# Tiny white pixel base64
IMG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
curl -s -X POST "$BASE_URL/vision/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$IMG\", \"analysis_type\": \"scientific\"}" | format_json

# 9. Embodiment
echo ""
echo "🤖 9. EMBODIMENT: Executing Drill Deployment..."
curl -s -X POST "$BASE_URL/v4/consciousness/embodiment/action/execute" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "deploy_drill", "parameters": {"depth": 5.0, "rpm": 1200}}' | format_json

# 10. Marketplace
echo ""
echo "💼 10. MARKETPLACE: Publishing Discovery..."
curl -s -X POST "$BASE_URL/v4/consciousness/marketplace/publish" \
  -H "Content-Type: application/json" \
  -d '{"insight_type": "scientific_discovery", "content": {"finding": "Water Ice Confirmed at 5m depth"}, "metadata": {"confidence": 0.99}}' | format_json

echo ""
echo "========================================"
echo "🏁 DEMO COMPLETE"
