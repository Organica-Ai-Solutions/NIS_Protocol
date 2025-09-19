#!/bin/bash
# Comprehensive NIS Protocol v3.2.1 Endpoint Testing
# Tests all optimizations and enhancements

echo "🧪 === COMPREHENSIVE NIS PROTOCOL v3.2.1 ENDPOINT TESTING ==="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local url="$3"
    local data="$4"
    local expected_status="$5"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${BLUE}$TOTAL_TESTS. $name${NC}"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$url")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$url")
    fi
    
    # Extract status code
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} - Status: $status_code"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Show response preview
        if command -v jq >/dev/null 2>&1; then
            echo "$response_body" | jq . 2>/dev/null | head -5 || echo "$response_body" | head -5
        else
            echo "$response_body" | head -5
        fi
    else
        echo -e "${RED}❌ FAIL${NC} - Expected: $expected_status, Got: $status_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "Response: $response_body" | head -3
    fi
    echo ""
}

echo "🔍 === BASIC SYSTEM HEALTH ==="

# 1. Basic Health Check
test_endpoint "Basic Health Check" "GET" "http://localhost/health" "" "200"

# 2. API Documentation
test_endpoint "API Documentation" "GET" "http://localhost/docs" "" "200"

echo "🔧 === TOOL OPTIMIZATION ENDPOINTS (NEW v3.2.1) ==="

# 3. Enhanced Tool Definitions
test_endpoint "Enhanced Tool Definitions" "GET" "http://localhost/api/tools/enhanced" "" "200"

# 4. Tool Optimization Metrics
test_endpoint "Tool Optimization Metrics" "GET" "http://localhost/api/tools/optimization/metrics" "" "200"

# 5. Optimized Chat with Token Efficiency
test_endpoint "Optimized Chat (Concise)" "POST" "http://localhost/chat/optimized" \
'{"message": "Test optimization", "response_format": "concise", "token_limit": 500}' "200"

echo "🚀 === NVIDIA INCEPTION INTEGRATION (NEW) ==="

# 6. NVIDIA Inception Status
test_endpoint "NVIDIA Inception Status" "GET" "http://localhost/nvidia/inception/status" "" "200"

# 7. NVIDIA NeMo Status
test_endpoint "NVIDIA NeMo Status" "GET" "http://localhost/nvidia/nemo/status" "" "200"

echo "🧠 === BRAIN ORCHESTRATION & AGENTS ==="

# 8. Agent Status (Consolidated)
test_endpoint "Agent Status (Consolidated)" "GET" "http://localhost/api/agents/status" "" "200"

# 9. Consciousness Status (Algorithmic Monitoring)
test_endpoint "Consciousness Status" "GET" "http://localhost/consciousness/status" "" "200"

# 10. Standard Chat
test_endpoint "Standard Chat" "POST" "http://localhost/chat" \
'{"message": "Hello NIS Protocol"}' "200"

echo "🔬 === PHYSICS VALIDATION ==="

# 11. Physics Constants
test_endpoint "Physics Constants" "GET" "http://localhost/physics/constants" "" "200"

# 12. Physics Validation
test_endpoint "Physics Validation" "POST" "http://localhost/physics/validate" \
'{"scenario": "5kg ball dropped from 10m", "expected_outcome": "Accelerates at 9.81 m/s²"}' "200"

echo "🔍 === RESEARCH & ANALYSIS ==="

# 13. Research Capabilities
test_endpoint "Research Capabilities" "GET" "http://localhost/research/capabilities" "" "200"

# 14. Deep Research
test_endpoint "Deep Research" "POST" "http://localhost/research/deep" \
'{"query": "quantum computing", "research_depth": "basic"}' "200"

echo "🎨 === MULTIMODAL & CREATIVE ==="

# 15. Multimodal Agent Status
test_endpoint "Multimodal Agent Status" "GET" "http://localhost/agents/multimodal/status" "" "200"

echo "🔌 === PROTOCOL INTEGRATION ==="

# 16. MCP Demo
test_endpoint "MCP Protocol Demo" "GET" "http://localhost/api/mcp/demo" "" "200"

# 17. LangGraph Status
test_endpoint "LangGraph Status" "GET" "http://localhost/api/langgraph/status" "" "200"

echo ""
echo "📊 === TEST SUMMARY ==="
echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! System fully operational with optimizations!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. System partially operational.${NC}"
    exit 1
fi
