#!/usr/bin/env python3
import requests
import json

# Test different learning operations
print("Testing Learning Process Operations...")

operations_to_test = [
    "train",
    "learn", 
    "process",
    "update",
    "adapt",
    "analyze",
    "extract",
    "infer"
]

base_data = {
    "learning_data": "Sample learning dataset",
    "learning_type": "supervised"
}

for operation in operations_to_test:
    print(f"\n--- Testing operation: '{operation}' ---")
    
    data = base_data.copy()
    data["operation"] = operation
    
    try:
        resp = requests.post("http://localhost:8000/agents/learning/process", json=data, timeout=5)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("✅ SUCCESS!")
            result = resp.json()
            print(json.dumps(result, indent=2)[:300] + "...")
            break  # Found working operation
        elif resp.status_code == 422:
            print("⚠️ Parameter error:")
            print(resp.json())
        elif resp.status_code == 500:
            error = resp.json()
            if "Unknown operation" in error.get("error", ""):
                print(f"❌ Unknown operation: {operation}")
            else:
                print(f"❌ Server error: {error}")
        else:
            print(f"❌ Error {resp.status_code}: {resp.text[:100]}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout")
    except Exception as e:
        print(f"❌ Exception: {e}")

print("\nOperation testing completed.")