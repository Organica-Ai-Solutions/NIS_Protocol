import requests
import json

def test_planning_endpoint():
    """
    Tests the /agents/planning/create_plan endpoint.
    """
    url = "http://127.0.0.1:8001/agents/planning/create_plan"
    headers = {"Content-Type": "application/json"}
    data = {"goal": "Achieve AGI"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes

        print("Response from /agents/planning/create_plan:")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error calling the planning endpoint: {e}")

if __name__ == "__main__":
    test_planning_endpoint() 