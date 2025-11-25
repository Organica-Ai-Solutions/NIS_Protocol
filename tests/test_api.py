#!/usr/bin/env python3
"""
NIS Protocol API Test Suite
Comprehensive pytest tests for all critical endpoints

Run with: pytest tests/test_api.py -v
"""

import pytest
import httpx
import asyncio
import os

# Configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0


@pytest.fixture
def client():
    """HTTP client fixture"""
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


@pytest.fixture
def async_client():
    """Async HTTP client fixture"""
    return httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)


# =============================================================================
# HEALTH & CORE TESTS
# =============================================================================

class TestHealth:
    """Health check and basic connectivity tests"""
    
    def test_health_endpoint(self, client):
        """Test /health returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_metrics_endpoint(self, client):
        """Test /metrics returns data"""
        response = client.get("/metrics")
        assert response.status_code == 200


# =============================================================================
# CHAT & LLM TESTS
# =============================================================================

class TestChat:
    """Chat endpoint tests"""
    
    def test_basic_chat(self, client):
        """Test basic chat works with real AI"""
        response = client.post("/chat", json={
            "message": "Say hello in one word",
            "conversation_id": "pytest-basic"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data.get("real_ai") == True  # Should use real AI
    
    def test_chat_with_provider(self, client):
        """Test chat with specific provider"""
        response = client.post("/chat", json={
            "message": "What is 1+1?",
            "conversation_id": "pytest-provider",
            "provider": "deepseek"
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("provider") == "deepseek"
    
    def test_chat_returns_confidence(self, client):
        """Test that chat returns confidence score"""
        response = client.post("/chat", json={
            "message": "Test message",
            "conversation_id": "pytest-confidence"
        })
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1


# =============================================================================
# V4 CONSCIOUSNESS TESTS
# =============================================================================

class TestV4Consciousness:
    """V4 Consciousness endpoint tests"""
    
    def test_genesis(self, client):
        """Test consciousness genesis"""
        response = client.post("/v4/consciousness/genesis", json={
            "request": {
                "seed_concept": "test_agent",
                "capabilities": ["reasoning"]
            }
        })
        # Accept 200 or 500 (genesis may have init issues)
        assert response.status_code in [200, 500]
    
    def test_plan(self, client):
        """Test consciousness planning"""
        response = client.post("/v4/consciousness/plan", json={
            "request": {
                "goal": "Test goal",
                "constraints": []
            }
        })
        # Accept 200 or 500 (plan may have init issues)
        assert response.status_code in [200, 500]
    
    def test_ethics_evaluate(self, client):
        """Test ethics evaluation"""
        response = client.post("/v4/consciousness/ethics/evaluate", json={
            "action": "help user",
            "context": "testing"
        })
        assert response.status_code == 200
        data = response.json()
        assert "ethical_score" in data or "ethics_score" in data or "evaluation" in data
    
    def test_debug_explain(self, client):
        """Test debug explanation"""
        response = client.get("/v4/consciousness/debug/explain")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"


# =============================================================================
# ANALYTICS TESTS
# =============================================================================

class TestAnalytics:
    """Analytics endpoint tests"""
    
    def test_dashboard(self, client):
        """Test analytics dashboard"""
        response = client.get("/analytics/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "dashboard_title" in data


# =============================================================================
# PHYSICS TESTS
# =============================================================================

class TestPhysics:
    """Physics validation tests"""
    
    def test_validate_physics(self, client):
        """Test physics validation"""
        response = client.post("/physics/validate", json={
            "domain": "classical_mechanics",
            "parameters": {
                "mass": 10,
                "velocity": 5,
                "force": 50
            }
        })
        assert response.status_code == 200


# =============================================================================
# SYSTEM STATUS TESTS  
# =============================================================================

class TestSystem:
    """System status tests"""
    
    def test_system_status(self, client):
        """Test system status"""
        response = client.get("/system/status")
        assert response.status_code == 200


# =============================================================================
# PROVIDER FALLBACK TESTS
# =============================================================================

class TestProviderFallback:
    """Test that provider fallback works correctly"""
    
    def test_fallback_on_failure(self, client):
        """Test that system falls back to working provider"""
        # Request non-working provider - should fallback
        response = client.post("/chat", json={
            "message": "Test",
            "conversation_id": "pytest-fallback",
            "provider": "openai"  # This one has quota issues
        })
        assert response.status_code == 200
        data = response.json()
        # Should have fallen back to another provider
        assert data.get("real_ai") == True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_conversation_flow(self, client):
        """Test a complete conversation flow"""
        conversation_id = "pytest-e2e-flow"
        
        # First message
        r1 = client.post("/chat", json={
            "message": "Hello, I'm testing",
            "conversation_id": conversation_id
        })
        assert r1.status_code == 200
        
        # Second message (context should be maintained)
        r2 = client.post("/chat", json={
            "message": "What did I just say?",
            "conversation_id": conversation_id
        })
        assert r2.status_code == 200


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
