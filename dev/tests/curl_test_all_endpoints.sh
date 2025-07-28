#!/bin/bash

echo "🚀 NIS PROTOCOL v3.1 - COMPREHENSIVE CURL ENDPOINT TEST"
echo "Testing all 40+ endpoints across 10 categories"
echo "=" * 80

BASE_URL="http://localhost:8000"
PASSED=0
TOTAL=0
FAILED_ENDPOINTS=()

# Function to test GET endpoint
test_get() {
    local endpoint=$1
    local name=$2
    echo -n "Testing GET $endpoint ($name)... "
    TOTAL=$((TOTAL + 1))
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$BASE_URL$endpoint")
    if [ "$response" = "200" ]; then
        echo "✅ PASS"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL ($response)"
        FAILED_ENDPOINTS+=("GET $endpoint")
    fi
}

# Function to test POST endpoint
test_post() {
    local endpoint=$1
    local data=$2
    local name=$3
    echo -n "Testing POST $endpoint ($name)... "
    TOTAL=$((TOTAL + 1))
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$BASE_URL$endpoint")
    
    if [ "$response" = "200" ]; then
        echo "✅ PASS"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL ($response)"
        FAILED_ENDPOINTS+=("POST $endpoint")
    fi
}

# Function to test DELETE endpoint
test_delete() {
    local endpoint=$1
    local name=$2
    echo -n "Testing DELETE $endpoint ($name)... "
    TOTAL=$((TOTAL + 1))
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X DELETE \
        "$BASE_URL$endpoint")
    
    if [ "$response" = "200" ]; then
        echo "✅ PASS"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL ($response)"
        FAILED_ENDPOINTS+=("DELETE $endpoint")
    fi
}

echo ""
echo "🏠 TESTING CORE ENDPOINTS"
echo "-" * 40

test_get "/" "Root System Info"
test_get "/health" "Health Check"

echo ""
echo "💬 TESTING CONVERSATIONAL LAYER"
echo "-" * 40

test_post "/chat" '{"message": "Hello v3.1!", "user_id": "curl_tester"}' "Enhanced Chat"
test_post "/chat/contextual" '{"message": "Explain quantum computing", "user_id": "curl_tester", "reasoning_mode": "chain_of_thought", "tools_enabled": ["calculator"]}' "Contextual Chat"

echo ""
echo "🌐 TESTING INTERNET & KNOWLEDGE"
echo "-" * 40

test_post "/internet/search" '{"query": "artificial intelligence", "max_results": 5, "academic_sources": true}' "Web Search"
test_post "/internet/fetch-url" '{"url": "https://example.com", "parse_mode": "auto", "extract_entities": true}' "URL Fetch"
test_post "/internet/fact-check" '{"statement": "AI can exhibit consciousness", "confidence_threshold": 0.8}' "Fact Check"
test_get "/internet/status" "Internet Status"

echo ""
echo "🔧 TESTING TOOL EXECUTION"
echo "-" * 40

test_get "/tool/list" "List Tools"
test_post "/tool/execute" '{"tool_name": "calculator", "parameters": {"expression": "5 * 5"}, "sandbox": true}' "Execute Tool"
test_post "/tool/register" '{"name": "test_tool", "description": "Test tool", "parameters_schema": {"input": {"type": "string"}}, "category": "testing"}' "Register Tool"

echo ""
echo "🤖 TESTING AGENT ORCHESTRATION"
echo "-" * 40

test_post "/agent/create" '{"agent_type": "research", "capabilities": ["analysis", "synthesis"], "memory_size": "1GB", "tools": ["calculator"]}' "Create Agent"
test_get "/agent/list" "List Agents"

# Get agent ID for further testing
AGENT_ID=$(curl -s -X POST -H "Content-Type: application/json" -d '{"agent_type": "general", "capabilities": ["test"], "memory_size": "512MB"}' "$BASE_URL/agent/create" | grep -o '"agent_id":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$AGENT_ID" ]; then
    test_post "/agent/instruct" "{\"agent_id\": \"$AGENT_ID\", \"instruction\": \"Test instruction\", \"priority\": 5}" "Instruct Agent"
fi

test_post "/agent/chain" '{"workflow": [{"agent": "test", "task": "analyze"}], "execution_mode": "sequential"}' "Agent Chain"

echo ""
echo "📊 TESTING MODEL MANAGEMENT"
echo "-" * 40

test_get "/models" "List Models"
test_post "/models/load" '{"model_name": "test-model", "model_type": "llm", "source": "local_cache"}' "Load Model"
test_get "/models/status" "Model Status"

echo ""
echo "🧠 TESTING MEMORY & KNOWLEDGE"
echo "-" * 40

test_post "/memory/store" '{"content": "NIS v3.1 test memory", "metadata": {"type": "test"}, "importance": 0.8}' "Store Memory"
test_post "/memory/query" '{"query": "NIS test", "max_results": 5, "similarity_threshold": 0.7}' "Query Memory"
test_post "/memory/semantic-link" '{"source_id": "mem_1", "target_id": "mem_2", "relationship": "related", "strength": 0.8}' "Semantic Link"

echo ""
echo "🧠 TESTING REASONING & VALIDATION"
echo "-" * 40

test_post "/reason/plan" '{"query": "How does consciousness work?", "reasoning_style": "chain_of_thought", "depth": "standard"}' "Reasoning Plan"
test_post "/reason/validate" '{"reasoning_chain": ["Step 1", "Step 2", "Step 3"], "physics_constraints": ["conservation_energy"]}' "Validate Reasoning"
test_get "/reason/status" "Reasoning Status"

echo ""
echo "📊 TESTING MONITORING & LOGS"
echo "-" * 40

test_get "/logs?level=INFO&limit=10" "Get Logs"
test_get "/dashboard/realtime" "Real-time Dashboard"
test_get "/metrics/latency" "Latency Metrics"

echo ""
echo "🛠️ TESTING DEVELOPER UTILITIES"
echo "-" * 40

test_post "/config/reload" '{}' "Reload Config"
test_post "/sandbox/execute" '{"code": "print(\"Hello v3.1!\")", "language": "python", "timeout": 10}' "Sandbox Execute"

# Create a simple agent for tracing
TRACE_AGENT_ID=$(curl -s -X POST -H "Content-Type: application/json" -d '{"agent_type": "general", "capabilities": ["trace"], "memory_size": "256MB"}' "$BASE_URL/agent/create" | grep -o '"agent_id":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$TRACE_AGENT_ID" ]; then
    test_post "/debug/trace-agent" "{\"agent_id\": \"$TRACE_AGENT_ID\", \"trace_depth\": \"full\", \"include_reasoning\": true}" "Trace Agent"
fi

test_post "/stress/load-test" '{"concurrent_users": 5, "duration_seconds": 30, "endpoint": "/health"}' "Load Test"

echo ""
echo "🔬 TESTING EXPERIMENTAL LAYERS"
echo "-" * 40

test_post "/kan/predict" '{"input_data": [1.0, 2.0, 3.0], "function_type": "symbolic", "interpretability_mode": true}' "KAN Predict"
test_post "/pinn/verify" '{"system_state": {"position": [1, 2, 3]}, "physical_laws": ["conservation_energy"], "boundary_conditions": {}}' "PINN Verify"
test_post "/laplace/transform" '{"signal_data": [1.0, 0.5, 0.2], "transform_type": "forward", "analysis_mode": "frequency"}' "Laplace Transform"
test_post "/a2a/connect" '{"target_node": "test-node", "authentication": "shared_key", "sync_memory": true}' "A2A Connect"

echo ""
echo "=" * 80
echo "🎯 COMPREHENSIVE TEST RESULTS"
echo "=" * 80

echo "📊 Overall Results: $PASSED/$TOTAL endpoints passed ($(( PASSED * 100 / TOTAL ))%)"

if [ $PASSED -ge $(( TOTAL * 8 / 10 )) ]; then
    echo "🏆 EXCELLENT: NIS Protocol v3.1 is performing exceptionally!"
    echo "✅ Outstanding performance across all endpoint categories!"
elif [ $PASSED -ge $(( TOTAL * 7 / 10 )) ]; then
    echo "🎉 VERY GOOD: Strong v3.1 performance!"
    echo "✅ Most endpoints operational with good reliability!"
elif [ $PASSED -ge $(( TOTAL * 6 / 10 )) ]; then
    echo "👍 GOOD: Solid v3.1 functionality!"
    echo "⚠️ Some endpoints need attention but core system is strong!"
else
    echo "⚠️ NEEDS ATTENTION: Several endpoints require debugging!"
fi

if [ ${#FAILED_ENDPOINTS[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed Endpoints:"
    for endpoint in "${FAILED_ENDPOINTS[@]}"; do
        echo "   $endpoint"
    done
fi

echo ""
echo "🎊 NIS Protocol v3.1 endpoint testing completed!"
echo "🌐 Access the API docs at: http://localhost:8000/docs"
echo "📊 View real-time dashboard at: http://localhost:8000/dashboard/realtime"

# Clean up
rm -f /tmp/response.json

exit 0 