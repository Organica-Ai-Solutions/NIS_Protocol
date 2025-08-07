#!/usr/bin/env python3
"""
Simple test server for endpoint testing
"""
import main
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting NIS Protocol v3 Test Server...")
    uvicorn.run(main.app, host="127.0.0.1", port=8000, log_level="info")