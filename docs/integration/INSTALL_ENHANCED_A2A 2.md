# Install Enhanced A2A WebSocket Endpoint

## Quick Setup

Add this endpoint to your `main.py` file after the existing `/ws/agentic` endpoint (around line 408):

```python
# ====== ENHANCED A2A WEBSOCKET ENDPOINT ======
from enhanced_a2a_websocket import enhanced_a2a_websocket

@app.websocket("/ws/a2a")
async def a2a_endpoint(websocket: WebSocket):
    """
    🚀 Enhanced A2A WebSocket - Full GenUI Integration
    
    Combines AG-UI transparency events with A2UIFormatter for rich widget generation.
    This endpoint properly formats responses as GenUI surfaces.
    """
    await enhanced_a2a_websocket(websocket, llm_provider, a2ui_formatter_instance)
```

## Verify Dependencies

Make sure these are initialized in main.py (they should already be):

1. **llm_provider** - Line 127: `llm_provider: Optional[GeneralLLMProvider] = None`
2. **a2ui_formatter_instance** - Line 150: `a2ui_formatter_instance: Optional[A2UIFormatter] = None`

Both should be initialized during app startup in the initialization section.

## Test the Endpoint

1. Start the backend:
   ```bash
   cd /Users/diegofuego/Desktop/NIS_Protocol
   python main.py
   ```

2. Check logs for:
   ```
   🚀 Enhanced A2A WebSocket connected
   ```

3. The Flutter app is already configured to connect to `/ws/a2a`

## What This Fixes

✅ **No more repeated messages** - Uses proper GenUI surface rendering
✅ **Rich widgets** - Code blocks, lists, headers, action buttons
✅ **Transparency** - Still shows AG-UI events (thinking steps, agent activation)
✅ **Leverages A2UIFormatter** - Backend does the heavy lifting

## Endpoint Location

Add after line 408 in main.py (after the `/ws/agentic` endpoint closes)
