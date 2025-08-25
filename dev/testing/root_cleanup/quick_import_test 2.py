#!/usr/bin/env python3
"""
Quick test to verify main.py imports work after dependency fixes
"""

print("Testing main.py imports...")

try:
    print("Testing basic Python imports...")
    import sys
    import time
    import asyncio
    print("✅ Basic imports OK")
    
    print("Testing FastAPI imports...")
    import fastapi
    print("✅ FastAPI OK")
    
    print("Testing transformers imports...")
    import transformers
    print("✅ Transformers OK")
    
    print("Testing sentence_transformers imports...")
    import sentence_transformers
    print("✅ Sentence Transformers OK")
    
    print("Testing main module import...")
    import main
    print("✅ Main module imports successfully!")
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
