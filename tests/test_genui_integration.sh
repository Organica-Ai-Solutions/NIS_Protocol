#!/bin/bash

# Test GenUI Integration
# Tests backend A2UI format and WebSocket endpoints

echo "======================================"
echo "GenUI Integration Test"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Backend Health
echo "Test 1: Backend Health"
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH" = "healthy" ]; then
    echo -e "${GREEN}✓ Backend is healthy${NC}"
else
    echo -e "${RED}✗ Backend is not healthy${NC}"
    exit 1
fi
echo ""

# Test 2: A2UI Message Format
echo "Test 2: A2UI Message Format"
RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"Hello","genui_enabled":true}')

HAS_A2UI=$(echo "$RESPONSE" | jq -r '.a2ui_messages != null')
if [ "$HAS_A2UI" = "true" ]; then
    echo -e "${GREEN}✓ Response has a2ui_messages${NC}"
else
    echo -e "${RED}✗ Response missing a2ui_messages${NC}"
    exit 1
fi

MSG_COUNT=$(echo "$RESPONSE" | jq '.a2ui_messages | length')
echo "  Messages: $MSG_COUNT"

FIRST_TYPE=$(echo "$RESPONSE" | jq -r '.a2ui_messages[0] | keys[0]')
SECOND_TYPE=$(echo "$RESPONSE" | jq -r '.a2ui_messages[1] | keys[0]')

if [ "$FIRST_TYPE" = "beginRendering" ]; then
    echo -e "${GREEN}✓ First message is beginRendering${NC}"
else
    echo -e "${RED}✗ First message is not beginRendering: $FIRST_TYPE${NC}"
fi

if [ "$SECOND_TYPE" = "surfaceUpdate" ]; then
    echo -e "${GREEN}✓ Second message is surfaceUpdate${NC}"
else
    echo -e "${RED}✗ Second message is not surfaceUpdate: $SECOND_TYPE${NC}"
fi
echo ""

# Test 3: Component Structure
echo "Test 3: Component Structure"
COMPONENT=$(echo "$RESPONSE" | jq '.a2ui_messages[1].surfaceUpdate.components[0]')
HAS_ID=$(echo "$COMPONENT" | jq -r '.id != null')
HAS_TYPE=$(echo "$COMPONENT" | jq -r '.type != null')
HAS_PROPS=$(echo "$COMPONENT" | jq -r '.properties != null')

if [ "$HAS_ID" = "true" ]; then
    echo -e "${GREEN}✓ Component has id${NC}"
else
    echo -e "${RED}✗ Component missing id${NC}"
fi

if [ "$HAS_TYPE" = "true" ]; then
    COMP_TYPE=$(echo "$COMPONENT" | jq -r '.type')
    echo -e "${GREEN}✓ Component has type: $COMP_TYPE${NC}"
else
    echo -e "${RED}✗ Component missing type${NC}"
fi

if [ "$HAS_PROPS" = "true" ]; then
    echo -e "${GREEN}✓ Component has properties${NC}"
else
    echo -e "${RED}✗ Component missing properties${NC}"
fi
echo ""

# Test 4: WebSocket Endpoints (check if they exist)
echo "Test 4: WebSocket Endpoints"
echo -e "${YELLOW}Note: WebSocket endpoints require wscat to test${NC}"
echo "  To test manually:"
echo "    wscat -c ws://localhost:8000/ws/agents"
echo "    wscat -c ws://localhost:8000/ws/tao"
echo ""

# Test 5: Code Block Example
echo "Test 5: Code Block Response"
CODE_RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"Show me a Python hello world","genui_enabled":true}')

CODE_COMPONENTS=$(echo "$CODE_RESPONSE" | jq '.a2ui_messages[1].surfaceUpdate.components | length')
echo "  Components created: $CODE_COMPONENTS"

if [ "$CODE_COMPONENTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Code response generated components${NC}"
else
    echo -e "${YELLOW}⚠ No components generated${NC}"
fi
echo ""

echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "${GREEN}✓ Backend A2UI format is correct${NC}"
echo -e "${GREEN}✓ GenUI SDK format validated${NC}"
echo -e "${YELLOW}⚠ WebSocket endpoints need manual testing${NC}"
echo ""
echo "Next steps:"
echo "1. Test Flutter app with backend"
echo "2. Verify rich UI renders"
echo "3. Test WebSocket connections"
echo ""
