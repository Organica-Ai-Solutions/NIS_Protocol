#!/usr/bin/env python3
"""
Cosmos Stack Integration Test
Tests Cosmos agent (8009), NIS Protocol (8000), and /cookoff/robot-plan
"""
import json
import sys
import urllib.request
import urllib.error

def test(url, method="GET", data=None):
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode(), r.status
    except urllib.error.URLError as e:
        return str(e), None

def main():
    print("=" * 60)
    print("Cosmos Stack Integration Test")
    print("=" * 60)

    # 1. Cosmos Agent health
    print("\n1. Cosmos Agent (8009) /health")
    body, status = test("http://localhost:8009/health")
    if status == 200:
        print(f"   OK: {body[:80]}...")
    else:
        print(f"   FAIL: {body}")
        return 1

    # 2. Cosmos Agent status (full stack)
    print("\n2. Cosmos Agent (8009) /status")
    body, status = test("http://localhost:8009/status")
    if status == 200:
        d = json.loads(body)
        print(f"   mode: {d.get('mode')}")
        for k, v in d.get("cosmos_stack", {}).items():
            print(f"   {k}: {v.get('url')} (available={v.get('available')})")
    else:
        print(f"   FAIL: {body}")
        return 1

    # 3. Cosmos Agent /reason
    print("\n3. Cosmos Agent (8009) POST /reason")
    payload = json.dumps({"query": "Pick up the red cube", "robot_state": {}}).encode()
    body, status = test("http://localhost:8009/reason", "POST", payload)
    if status == 200:
        d = json.loads(body)
        print(f"   OK: reasoning_chain={d.get('reasoning_chain', '')[:60]}...")
        print(f"   action_plan: {d.get('action_plan')}")
    else:
        print(f"   FAIL: {body}")
        return 1

    # 4. NIS Protocol health
    print("\n4. NIS Protocol (8000) /health")
    body, status = test("http://localhost:8000/health")
    if status == 200:
        print(f"   OK: {body[:80]}...")
    else:
        print(f"   FAIL: {body}")
        return 1

    # 5. NIS Protocol /cookoff/robot-plan
    print("\n5. NIS Protocol (8000) POST /cookoff/robot-plan")
    payload = json.dumps({"query": "Pick up the red cube", "robot_state": {}}).encode()
    body, status = test("http://localhost:8000/cookoff/robot-plan", "POST", payload)
    if status == 200:
        d = json.loads(body)
        chain = d.get("cosmos_reasoning", {}).get("reasoning_chain", "")[:60]
        actions = d.get("action_recommendations", [])
        print(f"   OK: reasoning_chain={chain}...")
        print(f"   action_recommendations: {actions}")
    else:
        print(f"   FAIL: {body}")
        return 1

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
