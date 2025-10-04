#!/bin/bash
# 🧪 NIS Protocol v3 - Comprehensive Endpoint Test Script
# Tests all API endpoints to verify system functionality
# Usage: ./comprehensive_endpoint_test.sh

set -e

echo "🚀 NIS Protocol v3 - Comprehensive Endpoint Test"
echo "================================================"
echo ""

BASE_URL="${BASE_URL:-http://localhost}"
PASSED=0
FAILED=0
TOTAL=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_key="$5"
    
    TOTAL=$((TOTAL + 1))
    echo -n "${BLUE}[TEST ${TOTAL}]${NC} ${name}... "
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -X POST "${BASE_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s "${BASE_URL}${endpoint}" 2>&1)
    fi
    
    # Check if response contains expected key or is valid JSON
    if echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); sys.exit(0 if '$expected_key' in d or '$expected_key' == '' else 1)" 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "   Response: $(echo $response | head -c 100)"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

echo "🏠 Testing System Endpoints"
echo "=============================="
test_endpoint "Root Endpoint" "GET" "/" "" "system"
test_endpoint "Health Check" "GET" "/health" "" "status"
test_endpoint "Metrics" "GET" "/metrics" "" ""

echo ""
echo "💬 Testing Chat Endpoints"
echo "=============================="
test_endpoint "Basic Chat" "POST" "/chat" '{"message":"Hello test","user_id":"test"}' "response"
test_endpoint "Streaming Chat" "POST" "/chat/stream" '{"message":"Hi"}' ""
test_endpoint "Optimized Chat" "POST" "/chat/optimized" '{"message":"Test"}' ""

echo ""
echo "🔍 Testing Research Endpoints"
echo "=============================="
test_endpoint "Deep Research" "POST" "/research/deep" '{"query":"quantum computing","research_depth":"basic","time_limit":10}' "success"
test_endpoint "Research Capabilities" "GET" "/research/capabilities" "" "capabilities"

echo ""
echo "🤖 Testing Agent Endpoints"
echo "=============================="
test_endpoint "Agent Status" "GET" "/agents/status" "" "status"
test_endpoint "Consciousness Analyze" "POST" "/agents/consciousness/analyze" '{"input":"test consciousness"}' ""

echo ""
echo "🎨 Testing Multimodal Endpoints"
echo "=============================="
test_endpoint "Vision Analysis" "POST" "/vision/analyze" '{"image_url":"test.jpg","query":"test"}' "status"
test_endpoint "Document Analysis" "POST" "/document/analyze" '{"document_data":"test","context":"analyze"}' "status"

echo ""
echo "🧠 Testing Reasoning Endpoints"
echo "=============================="
test_endpoint "Collaborative Reasoning" "POST" "/reasoning/collaborative" '{"query":"test","max_iterations":2}' ""
test_endpoint "Debate Reasoning" "POST" "/reasoning/debate" '{"topic":"AI","positions":["pro","con"]}' ""

echo ""
echo "🔬 Testing Physics Endpoints"
echo "=============================="
test_endpoint "Physics Capabilities" "GET" "/physics/capabilities" "" "capabilities"
test_endpoint "Physics Validation" "POST" "/physics/validate" '{"equation":"F=ma","context":"classical mechanics"}' ""
test_endpoint "Physics Constants" "GET" "/physics/constants" "" ""

echo ""
echo "🔌 Testing MCP/LangGraph Endpoints"
echo "=============================="
test_endpoint "MCP Demo" "GET" "/api/mcp/demo" "" ""
test_endpoint "MCP Invoke" "POST" "/api/mcp/invoke" '{"input":{"messages":[{"role":"user","content":"test"}]},"config":{"configurable":{"thread_id":"test"}}}' ""

echo ""
echo "🚀 Testing NVIDIA NeMo Endpoints"
echo "=============================="
test_endpoint "NeMo Status" "GET" "/nvidia/nemo/status" "" "status"
test_endpoint "NeMo Toolkit Status" "GET" "/nvidia/nemo/toolkit/status" "" ""

echo ""
echo "📊 Test Summary"
echo "=============================="
echo -e "Total Tests: ${TOTAL}"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi
