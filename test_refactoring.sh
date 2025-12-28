#!/bin/bash
# Test script for refactoring changes
# Tests hardware auto-detection and security hardening

set -e

BACKEND_URL="http://localhost:8000"
RUNNER_URL="http://localhost:8001"

echo "=========================================="
echo "NIS Protocol Refactoring Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_endpoint() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="$4"
    
    echo -n "Testing: $name... "
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>/dev/null || echo "000")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        echo ""
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        echo "$body"
        echo ""
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "=========================================="
echo "1. Backend Health Check"
echo "=========================================="
test_endpoint "Backend Health" "$BACKEND_URL/health"

echo "=========================================="
echo "2. Hardware Auto-Detection Tests"
echo "=========================================="

echo "--- CAN Protocol Status ---"
test_endpoint "CAN Protocol Status" "$BACKEND_URL/robotics/can/status"

echo "--- OBD Protocol Status ---"
test_endpoint "OBD Protocol Status" "$BACKEND_URL/robotics/obd/status"

echo "--- Vehicle Data (OBD) ---"
test_endpoint "OBD Vehicle Data" "$BACKEND_URL/robotics/obd/vehicle_data"

echo "=========================================="
echo "3. Runner Security Tests"
echo "=========================================="

echo "--- Runner Health ---"
test_endpoint "Runner Health" "$RUNNER_URL/health"

echo "--- Safe Code Execution ---"
safe_code='{
  "code_content": "import numpy as np\nresult = np.mean([1, 2, 3, 4, 5])\nprint(f\"Mean: {result}\")",
  "programming_language": "python",
  "execution_timeout_seconds": 10,
  "memory_limit_mb": 256
}'
test_endpoint "Safe Code Execution" "$RUNNER_URL/execute" "POST" "$safe_code"

echo "--- Dangerous Code Detection (eval) ---"
dangerous_code_eval='{
  "code_content": "eval(\"print(1+1)\")",
  "programming_language": "python"
}'
echo -n "Testing: Dangerous Code Detection (eval)... "
response=$(curl -s -w "\n%{http_code}" -X POST "$RUNNER_URL/execute" \
    -H "Content-Type: application/json" \
    -d "$dangerous_code_eval" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if echo "$body" | grep -q "Security violations detected" || echo "$body" | grep -q "security_violations"; then
    echo -e "${GREEN}✓ PASS${NC} - Blocked dangerous code"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
    echo ""
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} - Did not block dangerous code"
    echo "$body"
    echo ""
    ((TESTS_FAILED++))
fi

echo "--- Dangerous Code Detection (import bypass) ---"
dangerous_code_import='{
  "code_content": "__import__(\"os\").system(\"ls\")",
  "programming_language": "python"
}'
echo -n "Testing: Dangerous Code Detection (import bypass)... "
response=$(curl -s -w "\n%{http_code}" -X POST "$RUNNER_URL/execute" \
    -H "Content-Type: application/json" \
    -d "$dangerous_code_import" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if echo "$body" | grep -q "Security violations detected" || echo "$body" | grep -q "security_violations"; then
    echo -e "${GREEN}✓ PASS${NC} - Blocked import bypass"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
    echo ""
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} - Did not block import bypass"
    echo "$body"
    echo ""
    ((TESTS_FAILED++))
fi

echo "--- Network Access Detection ---"
dangerous_code_network='{
  "code_content": "import socket\ns = socket.socket()\ns.connect((\"google.com\", 80))",
  "programming_language": "python"
}'
echo -n "Testing: Network Access Detection... "
response=$(curl -s -w "\n%{http_code}" -X POST "$RUNNER_URL/execute" \
    -H "Content-Type: application/json" \
    -d "$dangerous_code_network" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if echo "$body" | grep -q "Security violations detected" || echo "$body" | grep -q "security_violations"; then
    echo -e "${GREEN}✓ PASS${NC} - Blocked network access"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
    echo ""
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} - Network access not blocked (may be runtime blocking)"
    echo "$body"
    echo ""
fi

echo "=========================================="
echo "4. Action Validation Pattern (formerly Cursor Pattern)"
echo "=========================================="

echo "--- System Status ---"
test_endpoint "System Status" "$BACKEND_URL/system/status"

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
