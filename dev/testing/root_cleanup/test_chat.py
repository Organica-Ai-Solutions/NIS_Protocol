import requests
import json

url = "http://localhost:8001/chat"
headers = {"Content-Type": "application/json"}
data = {"message": "Hello NIS Protocol!", "user_id": "test_user"}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.json()) 