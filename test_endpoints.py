import requests

def test_endpoint(endpoint, file):
    try:
        response = requests.get(f"http://localhost:8000/{endpoint}")
        if response.status_code == 200:
            file.write(f"Endpoint /{endpoint}: SUCCESS\n")
            file.write(str(response.json()) + "\n")
        else:
            file.write(f"Endpoint /{endpoint}: FAILED with status {response.status_code}\n")
    except requests.exceptions.ConnectionError:
        file.write(f"Endpoint /{endpoint}: FAILED to connect\n")

def test_chat_endpoint(file):
    try:
        response = requests.post("http://localhost:8000/chat", json={"message": "hello"})
        if response.status_code == 200:
            file.write(f"Endpoint /chat: SUCCESS\n")
            file.write(str(response.json()) + "\n")
        else:
            file.write(f"Endpoint /chat: FAILED with status {response.status_code}\n")
    except requests.exceptions.ConnectionError:
        file.write(f"Endpoint /chat: FAILED to connect\n")

if __name__ == "__main__":
    with open("live_test_results.txt", "w") as f:
        test_endpoint("health", f)
        test_endpoint("agents", f)
        test_chat_endpoint(f) 