# GenUI A2UI Protocol Analysis

## Executive Summary

**Status**: ⚠️ **Partial Compatibility** - Our implementation works but doesn't follow the official GenUI A2UI protocol

**Critical Finding**: The official GenUI A2UI package uses **WebSocket streaming** with the A2A (Agent-to-Agent) protocol, not simple HTTP JSON responses.

---

## Official GenUI A2UI Protocol

### Architecture

```
Flutter App → WebSocket → A2A Server → AI Agent
     ↓                          ↓
GenUiConversation         A2UI Messages
     ↓                          ↓
A2uiMessageProcessor      SurfaceUpdate
     ↓                     DataModelUpdate
GenUiSurface              BeginRendering
```

### Key Components

1. **A2uiContentGenerator**: Manages WebSocket connection to A2A server
2. **A2uiAgentConnector**: Handles low-level WebSocket communication
3. **A2uiMessageProcessor**: Processes incoming A2UI messages
4. **GenUiConversation**: Maintains conversation context (taskId, contextId)

### Message Types

The official protocol uses streaming messages:
- `SurfaceUpdate` - Updates UI surfaces
- `DataModelUpdate` - Updates data models
- `BeginRendering` - Signals rendering start
- `AgentCard` - Agent metadata

---

## Our Current Implementation

### What We Built

**File**: `src/utils/a2ui_formatter.py`

**Approach**: HTTP REST endpoint that returns A2UI-like JSON

```python
# Our format
{
  "a2ui_message": {
    "role": "model",
    "widgets": [
      {"type": "Card", "data": {...}},
      {"type": "NISCodeBlock", "data": {...}}
    ]
  },
  "genui_formatted": true
}
```

### Differences from Official Protocol

| Aspect | Official GenUI A2UI | Our Implementation |
|--------|---------------------|-------------------|
| **Transport** | WebSocket streaming | HTTP REST |
| **Protocol** | A2A (Agent-to-Agent) | Custom JSON |
| **Messages** | SurfaceUpdate, DataModelUpdate | Single JSON response |
| **Connection** | Persistent WebSocket | Request/response |
| **Streaming** | Real-time updates | Single response |
| **Context** | taskId, contextId | conversation_id |

---

## Compatibility Assessment

### ✅ What Works

1. **Widget Structure**: Our widgets are valid GenUI widgets
2. **Custom Catalog**: NISCodeBlock and other custom widgets work
3. **Flutter Integration**: Your Flutter app can parse our responses
4. **Content**: The actual UI content is correct

### ⚠️ What's Missing

1. **WebSocket Support**: We use HTTP, not WebSocket
2. **A2A Protocol**: We don't implement the official A2A message format
3. **Streaming**: No real-time surface updates
4. **Agent Metadata**: No AgentCard implementation
5. **Task Context**: No taskId/contextId management

### ❌ What Doesn't Match

1. **Server URL**: Official expects `ws://` or `wss://`, we use `http://`
2. **Message Format**: Official uses A2A protocol, we use custom JSON
3. **Event Handling**: Official sends UI events back to server via WebSocket

---

## Two Integration Paths

### Path 1: Keep Current Implementation (Simpler)

**Pros**:
- ✅ Already working
- ✅ Simpler backend (HTTP REST)
- ✅ No WebSocket complexity
- ✅ Works with your custom Flutter ContentGenerator

**Cons**:
- ❌ Not using official GenUI A2UI package
- ❌ No streaming updates
- ❌ Custom integration code needed

**Recommendation**: Good for MVP and current use case

### Path 2: Implement Official A2A Protocol (Standard)

**Pros**:
- ✅ Uses official GenUI packages
- ✅ Standard protocol
- ✅ Streaming updates
- ✅ Better for long-term maintenance

**Cons**:
- ❌ Requires WebSocket implementation
- ❌ More complex backend
- ❌ Need to implement A2A protocol
- ❌ Significant refactoring

**Recommendation**: Consider for production if you need streaming

---

## Current Status: Why It Works

Your Flutter app uses a **custom ContentGenerator** (`NisContentGenerator`), not the official `A2uiContentGenerator`. This is why our HTTP-based approach works:

```dart
// Your custom implementation
class NisContentGenerator extends ContentGenerator {
  // Makes HTTP POST to /chat
  // Parses our custom JSON format
  // Converts to GenUI surfaces
}
```

vs.

```dart
// Official GenUI A2UI implementation
class A2uiContentGenerator extends ContentGenerator {
  // Opens WebSocket to A2A server
  // Parses A2A protocol messages
  // Processes SurfaceUpdate events
}
```

---

## Recommendations

### For Current MVP (Recommended)

**Keep our implementation** - it works and is simpler:

1. ✅ Backend returns A2UI-like JSON via HTTP
2. ✅ Flutter uses custom `NisContentGenerator`
3. ✅ All 11 custom widgets supported
4. ✅ No breaking changes needed

**Action**: Document that we use a custom A2UI-inspired format, not the official A2A protocol

### For Production (Future)

**Consider implementing official A2A protocol** if you need:

1. Real-time streaming updates
2. Standard protocol compliance
3. WebSocket-based communication
4. Official GenUI package support

**Action**: Create separate `/a2a` WebSocket endpoint that implements the official protocol

---

## Implementation Comparison

### Official A2A Server Example

```python
# What the official protocol expects
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/a2a")
async def a2a_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send AgentCard
    await websocket.send_json({
        "type": "agent_card",
        "data": {
            "name": "NIS Protocol",
            "description": "AI Agent"
        }
    })
    
    # Receive user message
    message = await websocket.receive_json()
    
    # Send SurfaceUpdate
    await websocket.send_json({
        "type": "surface_update",
        "surfaceId": "main_surface",
        "data": {
            "type": "NISCodeBlock",
            "code": "print('hello')",
            "language": "python"
        }
    })
```

### Our Current Implementation

```python
# What we actually do
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Generate response
    response_text = llm.generate(request.message)
    
    # Format as A2UI if requested
    if request.genui_enabled:
        a2ui = format_text_as_a2ui(response_text)
        return {
            "a2ui_message": a2ui,
            "genui_formatted": True
        }
    
    return {"response": response_text}
```

---

## Migration Path (If Needed)

### Phase 1: Add WebSocket Support
```python
@app.websocket("/a2a")
async def a2a_websocket(websocket: WebSocket):
    # Implement A2A protocol
    pass
```

### Phase 2: Implement A2A Messages
- AgentCard
- SurfaceUpdate
- DataModelUpdate
- BeginRendering

### Phase 3: Update Flutter App
```dart
// Switch to official package
final contentGenerator = A2uiContentGenerator(
  serverUrl: Uri.parse('ws://localhost:8000/a2a'),
);
```

---

## Conclusion

**Current State**: ✅ Working with custom implementation

**Official Protocol**: ⚠️ Not implemented (WebSocket + A2A required)

**Recommendation**: 
- **Short-term**: Keep current HTTP-based approach (it works!)
- **Long-term**: Consider implementing official A2A protocol for streaming

**Action Items**:
1. ✅ Document that we use A2UI-inspired format (not official A2A)
2. ⏳ Test all 11 custom widgets with current implementation
3. ⏳ Consider WebSocket/A2A implementation for v2.0

---

## References

- **Official GenUI A2UI**: https://github.com/flutter/genui/tree/main/packages/genui_a2ui
- **A2A Protocol**: Agent-to-Agent streaming protocol
- **Our Implementation**: `src/utils/a2ui_formatter.py`
- **Flutter Integration**: Custom `NisContentGenerator`
