#!/usr/bin/env python3
"""Quick connectivity check for NIS Protocol Docker backend."""

import json
import os
import sys
from typing import Dict, Tuple

import requests

DEFAULT_BASE_URL = os.environ.get("NIS_BACKEND_URL", "http://localhost:8000")

CHECKS = {
    "/health": "Core health",
    "/agents/status": "Agent orchestration",
    "/chat/routing/info": "Multiprovider routing",
    "/models/bitnet/status": "BitNet bundle status"
}


def check_endpoint(base_url: str, endpoint: str) -> Tuple[bool, Dict]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, data
    except Exception as exc:  # noqa: BLE001 - surface any issue
        return False, {"error": str(exc), "url": url}


def main() -> None:
    base_url = DEFAULT_BASE_URL
    print(f"Checking NIS backend at {base_url}\n")
    results = {}

    for endpoint, label in CHECKS.items():
        ok, payload = check_endpoint(base_url, endpoint)
        results[endpoint] = {"label": label, "ok": ok, "payload": payload}

    print(json.dumps(results, indent=2))

    if not all(item["ok"] for item in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
