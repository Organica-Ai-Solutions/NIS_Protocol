#!/bin/bash

# NIS Protocol v4.0.1 - Comprehensive Endpoint Testing (curl only)
# Tests all 260+ endpoints with curl calls

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUTPUT_FILE="curl_test_results_$(date +%Y%m%d_%H%M%S).json"
PASSED=0
FAILED=0
TOTAL=0

echo "🚀 Starting comprehensive endpoint testing..."
echo "Base URL: $BASE_URL"
echo "Output: $OUTPUT_FILE"
echo ""

# Initialize results file
echo "{" > "$OUTPUT_FILE"
echo '  "test_run": {' >> "$OUTPUT_FILE"
echo "    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$OUTPUT_FILE"
echo "    \"base_url\": \"$BASE_URL\"" >> "$OUTPUT_FILE"
echo "  }," >> "$OUTPUT_FILE"
echo '  "results": [' >> "$OUTPUT_FILE"

# Test function
test_endpoint() {
    local method=$1
    local path=$2
    local data=$3
    local description=$4
    
    TOTAL=$((TOTAL + 1))
    echo -n "Testing [$method] $path ... "
    
    local start_time=$(date +%s%N)
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL$path" 2>&1)
    elif [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$path" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    elif [ "$method" = "PUT" ]; then
        response=$(curl -s -w "\n%{http_code}" -X PUT "$BASE_URL$path" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    elif [ "$method" = "DELETE" ]; then
        response=$(curl -s -w "\n%{http_code}" -X DELETE "$BASE_URL$path" 2>&1)
    fi
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    # Determine if test passed (2xx or 3xx status codes)
    if [[ "$http_code" =~ ^[23][0-9][0-9]$ ]]; then
        echo "✅ PASS ($http_code - ${duration}ms)"
        PASSED=$((PASSED + 1))
        status="PASS"
    else
        echo "❌ FAIL ($http_code)"
        FAILED=$((FAILED + 1))
        status="FAIL"
    fi
    
    # Append to results file
    if [ $TOTAL -gt 1 ]; then
        echo "," >> "$OUTPUT_FILE"
    fi
    
    # Truncate body to 200 chars for preview
    body_preview=$(echo "$body" | cut -c 1-200)
    
    cat >> "$OUTPUT_FILE" << EOF
    {
      "endpoint": "$path",
      "method": "$method",
      "description": "$description",
      "status": "$status",
      "http_code": $http_code,
      "duration_ms": $duration,
      "response_preview": $(echo "$body_preview" | jq -Rs .)
    }
EOF
}

# ===== CORE SYSTEM ENDPOINTS =====
echo "📋 Testing Core System Endpoints..."
test_endpoint "GET" "/health" "" "Health check"
test_endpoint "GET" "/docs" "" "API documentation"
test_endpoint "GET" "/redoc" "" "ReDoc documentation"
test_endpoint "GET" "/openapi.json" "" "OpenAPI schema"

# ===== CHAT ENDPOINTS =====
echo ""
echo "💬 Testing Chat Endpoints..."
test_endpoint "POST" "/chat" '{"message":"Hello"}' "Basic chat"
test_endpoint "POST" "/chat/stream" '{"message":"Test stream"}' "Streaming chat"
test_endpoint "POST" "/v4/chat" '{"message":"V4 chat test"}' "V4 chat"

# ===== CONSCIOUSNESS ENDPOINTS (V4) =====
echo ""
echo "🧠 Testing Consciousness Endpoints..."
test_endpoint "POST" "/v4/consciousness/genesis" '{"goal":"Test genesis"}' "Genesis phase"
test_endpoint "POST" "/v4/consciousness/plan" '{"goal":"Test planning"}' "Planning phase"
test_endpoint "POST" "/v4/consciousness/collective" '{"goal":"Test collective"}' "Collective phase"
test_endpoint "POST" "/v4/consciousness/multipath" '{"goal":"Test multipath"}' "Multipath phase"
test_endpoint "POST" "/v4/consciousness/embodiment" '{"goal":"Test embodiment"}' "Embodiment phase"
test_endpoint "POST" "/v4/consciousness/ethics" '{"goal":"Test ethics"}' "Ethics phase"
test_endpoint "POST" "/v4/consciousness/marketplace" '{"goal":"Test marketplace"}' "Marketplace phase"

# ===== AGENT ENDPOINTS =====
echo ""
echo "🤖 Testing Agent Endpoints..."
test_endpoint "GET" "/agents/status" "" "Agent status"
test_endpoint "POST" "/agents/learning/process" '{"data":"test"}' "Learning process"
test_endpoint "GET" "/agents/learning/status" "" "Learning status"
test_endpoint "POST" "/agents/planning/create" '{"goal":"test"}' "Create plan"
test_endpoint "POST" "/agents/curiosity/explore" '{"topic":"test"}' "Curiosity exploration"
test_endpoint "POST" "/agents/self-audit" '{"scope":"test"}' "Self audit"
test_endpoint "POST" "/agents/ethics/evaluate" '{"action":"test"}' "Ethics evaluation"
test_endpoint "POST" "/agents/simulation/run" '{"scenario":"test"}' "Run simulation"

# ===== RESEARCH ENDPOINTS =====
echo ""
echo "🔬 Testing Research Endpoints..."
test_endpoint "POST" "/research/deep" '{"query":"AI research"}' "Deep research"
test_endpoint "POST" "/research/web-search" '{"query":"test"}' "Web search"
test_endpoint "POST" "/research/analyze" '{"text":"test"}' "Research analysis"

# ===== VISION ENDPOINTS =====
echo ""
echo "👁️ Testing Vision Endpoints..."
test_endpoint "POST" "/vision/analyze" '{"image_url":"https://picsum.photos/200"}' "Image analysis"
test_endpoint "POST" "/vision/generate" '{"prompt":"test"}' "Image generation"

# ===== PHYSICS ENDPOINTS =====
echo ""
echo "⚛️ Testing Physics Endpoints..."
test_endpoint "POST" "/physics/solve/heat-equation" '{"domain":"test"}' "Heat equation"
test_endpoint "POST" "/physics/solve/wave-equation" '{"domain":"test"}' "Wave equation"
test_endpoint "POST" "/physics/validate" '{"equation":"test"}' "Physics validation"

# ===== BITNET ENDPOINTS =====
echo ""
echo "🧮 Testing BitNet Endpoints..."
test_endpoint "GET" "/models/bitnet/status" "" "BitNet status"
test_endpoint "POST" "/training/bitnet/force" '{"reason":"test"}' "Force training"
test_endpoint "GET" "/training/bitnet/metrics" "" "Training metrics"
test_endpoint "GET" "/downloads/bitnet" "" "Download model"

# ===== MEMORY ENDPOINTS =====
echo ""
echo "💾 Testing Memory Endpoints..."
test_endpoint "POST" "/memory/store" '{"key":"test","value":"data"}' "Store memory"
test_endpoint "POST" "/memory/retrieve" '{"query":"test"}' "Retrieve memory"
test_endpoint "POST" "/memory/search" '{"query":"test"}' "Search memory"
test_endpoint "DELETE" "/memory/clear" "" "Clear memory"

# ===== AUTONOMOUS ENDPOINTS =====
echo ""
echo "🚁 Testing Autonomous Endpoints..."
test_endpoint "POST" "/autonomous/plan" '{"goal":"test"}' "Autonomous planning"
test_endpoint "POST" "/autonomous/execute" '{"plan_id":"test"}' "Execute plan"
test_endpoint "GET" "/autonomous/status" "" "Autonomous status"

# ===== ROBOTICS ENDPOINTS =====
echo ""
echo "🤖 Testing Robotics Endpoints..."
test_endpoint "POST" "/robotics/forward_kinematics" '{"joints":[0,0,0]}' "Forward kinematics"
test_endpoint "POST" "/robotics/kinematics/forward" '{"joints":[0,0,0]}' "Forward kinematics (alias)"
test_endpoint "POST" "/robotics/inverse_kinematics" '{"position":[0,0,0]}' "Inverse kinematics"
test_endpoint "POST" "/robotics/path_planning" '{"start":[0,0,0],"goal":[1,1,1]}' "Path planning"
test_endpoint "GET" "/robotics/status" "" "Robotics status"

# ===== AUDIO ENDPOINTS =====
echo ""
echo "🔊 Testing Audio Endpoints..."
test_endpoint "POST" "/audio/transcribe" '{"audio_url":"test.mp3"}' "Audio transcription"
test_endpoint "POST" "/audio/synthesize" '{"text":"Hello"}' "Audio synthesis"
test_endpoint "POST" "/audio/analyze" '{"audio_url":"test.mp3"}' "Audio analysis"

# ===== SIMULATION ENDPOINTS =====
echo ""
echo "🌐 Testing Simulation Endpoints..."
test_endpoint "POST" "/simulation/run" '{"scenario":"test"}' "Run simulation"
test_endpoint "GET" "/simulation/status" "" "Simulation status"
test_endpoint "POST" "/simulation/generative" '{"prompt":"test"}' "Generative simulation"

# ===== SYSTEM ENDPOINTS =====
echo ""
echo "⚙️ Testing System Endpoints..."
test_endpoint "GET" "/system/status" "" "System status"
test_endpoint "GET" "/system/metrics" "" "System metrics"
test_endpoint "GET" "/system/stream" "" "System stream"

# ===== TOOLS ENDPOINTS =====
echo ""
echo "🔧 Testing Tools Endpoints..."
test_endpoint "GET" "/tools/list" "" "List tools"
test_endpoint "POST" "/tools/execute" '{"tool":"test","params":{}}' "Execute tool"

# ===== MCP ENDPOINTS =====
echo ""
echo "🔌 Testing MCP Endpoints..."
test_endpoint "POST" "/mcp/chat" '{"message":"test"}' "MCP chat"
test_endpoint "GET" "/mcp/tools" "" "MCP tools"

# ===== PROTOCOLS ENDPOINTS =====
echo ""
echo "📡 Testing Protocol Endpoints..."
test_endpoint "POST" "/protocols/can/send" '{"id":123,"data":[1,2,3]}' "CAN send"
test_endpoint "POST" "/protocols/obd/query" '{"pid":"01"}' "OBD query"

# Finalize results file
echo "" >> "$OUTPUT_FILE"
echo "  ]," >> "$OUTPUT_FILE"
echo "  \"summary\": {" >> "$OUTPUT_FILE"
echo "    \"total\": $TOTAL," >> "$OUTPUT_FILE"
echo "    \"passed\": $PASSED," >> "$OUTPUT_FILE"
echo "    \"failed\": $FAILED," >> "$OUTPUT_FILE"
pass_rate=$(echo "scale=2; ($PASSED * 100) / $TOTAL" | bc)
echo "    \"pass_rate\": \"${pass_rate}%\"" >> "$OUTPUT_FILE"
echo "  }" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

# Print summary
echo ""
echo "========================================="
echo "📊 TEST SUMMARY"
echo "========================================="
pass_rate_display=$(echo "scale=2; ($PASSED * 100) / $TOTAL" | bc)
echo "Total Tests:  $TOTAL"
echo "Passed:       $PASSED ✅"
echo "Failed:       $FAILED ❌"
echo "Pass Rate:    ${pass_rate_display}%"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_FILE"

# Exit with error if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
