#!/usr/bin/env python3
"""Check actual /robot-plan response structure from H100."""
import urllib.request, json

req = urllib.request.Request(
    "http://localhost:8100/robot-plan",
    data=json.dumps({"command": "Pick up the red cube", "robot_type": "xarm"}).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(json.dumps(resp, indent=2))
