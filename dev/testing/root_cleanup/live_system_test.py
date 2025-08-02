import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_status(test_name, success, status_code=None, error=None, response_data=None):
    """Helper function to print formatted test status."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"--- Testing: {test_name} ---")
    print(f"Status: {status}")
    if status_code:
        print(f"HTTP Status Code: {status_code}")
    if error:
        print(f"Error: {error}")
    if response_data:
        # Truncate long responses for readability
        display_data = json.dumps(response_data, indent=2)
        if len(display_data) > 500:
            display_data = display_data[:500] + "\n... (response truncated)"
        print(f"Response Body:\n{display_data}")
    print("-" * (len(test_name) + 16))
    print()

def test_health_endpoint():
    """Tests the /health endpoint."""
    test_name = "GET /health"
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        # The health endpoint returns a detailed object, just check for 200 OK
        if response.status_code == 200:
            print_status(test_name, True, response.status_code, response_data=response.json())
            return True
        else:
            print_status(test_name, False, response.status_code, response_data=response.json())
            return False
    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False

def test_agents_endpoint():
    """Tests the /agents endpoint."""
    test_name = "GET /agents"
    try:
        response = requests.get(f"{BASE_URL}/agents", timeout=10)
        # The agents endpoint returns a dictionary of agents under the "agents" key
        if response.status_code == 200 and "agents" in response.json():
            print_status(test_name, True, response.status_code, response_data=response.json())
            return True
        else:
            print_status(test_name, False, response.status_code, response_data=response.json())
            return False
    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False

def test_consciousness_status_endpoint():
    """Tests the /consciousness/status endpoint."""
    test_name = "GET /consciousness/status"
    try:
        response = requests.get(f"{BASE_URL}/consciousness/status", timeout=10)
        if response.status_code == 200 and "consciousness_level" in response.json():
            print_status(test_name, True, response.status_code, response_data=response.json())
            return True
        else:
            print_status(test_name, False, response.status_code, response_data=response.json())
            return False
    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False
        
def test_metrics_endpoint():
    """Tests the /metrics endpoint."""
    test_name = "GET /metrics"
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        if response.status_code == 200 and "uptime" in response.json():
            print_status(test_name, True, response.status_code, response_data=response.json())
            return True
        else:
            print_status(test_name, False, response.status_code, response_data=response.json())
            return False
    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False

def test_chat_endpoint():
    """Tests the /chat endpoint."""
    test_name = "POST /chat"
    payload = {"message": "Hello, NIS."}
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60) # Increased timeout
        if response.status_code == 200 and "response" in response.json():
            print_status(test_name, True, response.status_code, response_data=response.json())
            return True
        else:
            print_status(test_name, False, response.status_code, response_data=response.json())
            return False
    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False

def test_chat_stream_endpoint():
    """Tests the /chat/stream endpoint."""
    test_name = "POST /chat/stream"
    payload = {"message": "Hello, streaming NIS."}
    try:
        with requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True, timeout=60) as response:
            if response.status_code != 200:
                print_status(test_name, False, response.status_code, error="Failed to connect to stream")
                return False

            received_data = False
            full_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    received_data = True
                    chunk_str = chunk.decode('utf-8')
                    full_response += chunk_str
            
            if received_data:
                print_status(test_name, True, response.status_code, response_data={"stream_content": full_response})
                return True
            else:
                print_status(test_name, False, response.status_code, error="Stream opened but no data received.")
                return False

    except requests.exceptions.RequestException as e:
        print_status(test_name, False, error=str(e))
        return False

def main():
    """Runs all API endpoint tests."""
    print("--- Starting NIS Protocol v3.1 Live System Test ---")
    # Give the system a moment to ensure all services are ready
    print("Waiting for 30 seconds for all services to initialize...")
    time.sleep(30)
    
    results = {
        "health": test_health_endpoint(),
        "agents": test_agents_endpoint(),
        "consciousness": test_consciousness_status_endpoint(),
        "metrics": test_metrics_endpoint(),
        "chat": test_chat_endpoint(),
        "chat_stream": test_chat_stream_endpoint(),
    }
    
    print("\n--- Test Summary ---")
    all_success = True
    for test, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test.replace('_', ' ').title()}")
        if not success:
            all_success = False
            
    print("\n---" + (" All tests passed! " if all_success else " Some tests failed. ") + "---")

if __name__ == "__main__":
    main() 