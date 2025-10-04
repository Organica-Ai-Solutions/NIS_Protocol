#!/usr/bin/env python3
"""
NIS Protocol Configuration Verification
Checks which real integrations are available vs mocks
"""

import os
import sys

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded: {env_path}\n")
    else:
        print("⚠️  No .env file found - using system environment only\n")
except ImportError:
    print("Note: python-dotenv not installed, reading from system environment only\n")

def check_config():
    """Verify configuration and show what's enabled"""
    
    print("=" * 80)
    print("NIS PROTOCOL v3.2 - CONFIGURATION STATUS")
    print("=" * 80)
    print()
    
    # LLM Providers
    print("🤖 LLM PROVIDERS:")
    print("-" * 80)
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    openai_status = "✅ REAL API" if openai_key and openai_key != "your-openai-key-here" else "⚠️  MOCK (no key)"
    anthropic_status = "✅ REAL API" if anthropic_key and anthropic_key != "your-anthropic-key-here" else "⚠️  MOCK (no key)"
    google_status = "✅ REAL API" if google_key and google_key != "your-google-key-here" else "⚠️  MOCK (no key)"
    deepseek_status = "✅ REAL API" if deepseek_key and deepseek_key != "your-deepseek-key-here" else "⚠️  MOCK (no key)"
    
    print(f"  OpenAI:    {openai_status}")
    print(f"  Anthropic: {anthropic_status}")
    print(f"  Google:    {google_status}")
    print(f"  DeepSeek:  {deepseek_status}")
    print()
    
    # Protocol Adapters
    print("🌐 THIRD-PARTY PROTOCOLS:")
    print("-" * 80)
    
    mcp_url = os.getenv("MCP_SERVER_URL", "")
    a2a_key = os.getenv("A2A_API_KEY", "")
    acp_url = os.getenv("ACP_BASE_URL", "")
    
    mcp_status = "✅ CONFIGURED" if mcp_url else "⚠️  NOT SET"
    a2a_status = "✅ CONFIGURED" if a2a_key and a2a_key != "your-google-a2a-key-here" else "⚠️  NOT SET"
    acp_status = "✅ CONFIGURED" if acp_url else "⚠️  NOT SET"
    
    print(f"  MCP (Model Context):     {mcp_status}")
    if mcp_url:
        print(f"    → {mcp_url}")
    
    print(f"  A2A (Agent2Agent):       {a2a_status}")
    if a2a_key and a2a_key != "your-google-a2a-key-here":
        print(f"    → Configured with API key")
    
    print(f"  ACP (Agent Comm):        {acp_status}")
    if acp_url:
        print(f"    → {acp_url}")
    print()
    
    # Vector Store
    print("🗄️  VECTOR DATABASE:")
    print("-" * 80)
    
    vector_backend = os.getenv("VECTOR_STORE_BACKEND", "auto")
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    weaviate_url = os.getenv("WEAVIATE_URL", "")
    
    print(f"  Backend Mode: {vector_backend}")
    
    if pinecone_key and pinecone_key != "your-pinecone-key-here":
        print(f"  Pinecone:     ✅ CONFIGURED (production)")
        print(f"    → Environment: {os.getenv('PINECONE_ENVIRONMENT', 'not set')}")
    else:
        print(f"  Pinecone:     ⚠️  NOT SET")
    
    if weaviate_url and weaviate_url != "http://localhost:8080":
        print(f"  Weaviate:     ✅ CONFIGURED")
        print(f"    → {weaviate_url}")
    else:
        print(f"  Weaviate:     ⚠️  DEFAULT/NOT SET")
    
    # Check for hnswlib
    try:
        import hnswlib
        print(f"  HNSW:         ✅ AVAILABLE (local fallback)")
    except ImportError:
        print(f"  HNSW:         ⚠️  NOT AVAILABLE")
    
    print(f"  Simple Store: ✅ ALWAYS AVAILABLE (last resort)")
    print()
    
    # Voice
    print("🎙️  VOICE & AUDIO:")
    print("-" * 80)
    
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    
    elevenlabs_status = "✅ CONFIGURED" if elevenlabs_key and elevenlabs_key != "your-elevenlabs-key-here" else "⚠️  NOT SET"
    
    print(f"  ElevenLabs TTS: {elevenlabs_status}")
    print(f"  Whisper STT:    ✅ Model: {whisper_model}")
    print(f"  Bark TTS:       ✅ ALWAYS AVAILABLE (local)")
    print()
    
    # Infrastructure
    print("🗃️  INFRASTRUCTURE:")
    print("-" * 80)
    
    db_url = os.getenv("DATABASE_URL", "")
    kafka = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    redis = os.getenv("REDIS_HOST", "")
    
    print(f"  PostgreSQL: {'✅ CONFIGURED' if db_url else '⚠️  NOT SET'}")
    print(f"  Kafka:      {'✅ CONFIGURED' if kafka else '⚠️  NOT SET'}")
    print(f"  Redis:      {'✅ CONFIGURED' if redis else '⚠️  NOT SET'}")
    print()
    
    # Summary
    print("=" * 80)
    print("SYSTEM READINESS:")
    print("=" * 80)
    
    real_llm_count = sum([
        bool(openai_key and openai_key != "your-openai-key-here"),
        bool(anthropic_key and anthropic_key != "your-anthropic-key-here"),
        bool(google_key and google_key != "your-google-key-here")
    ])
    
    protocol_count = sum([
        bool(mcp_url),
        bool(a2a_key and a2a_key != "your-google-a2a-key-here"),
        bool(acp_url)
    ])
    
    print(f"  ✅ Real LLM Providers:    {real_llm_count}/3")
    print(f"  ✅ Protocol Adapters:     {protocol_count}/3")
    print(f"  ✅ Vector Database:       {'Production' if pinecone_key or weaviate_url else 'Local/Mock'}")
    print()
    
    if real_llm_count > 0:
        print("  🎉 READY FOR PRODUCTION with real AI!")
    else:
        print("  ⚠️  Using MOCK responses - add LLM API keys for real AI")
    
    if protocol_count > 0:
        print(f"  🌐 {protocol_count} external protocol(s) configured")
    else:
        print("  ℹ️  No external protocols configured (optional)")
    
    print()
    print("To start the system: ./start.sh")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        check_config()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)

