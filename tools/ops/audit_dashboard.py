"""Audit the Pi agent's cosmos/dashboard and the local index.html."""
import requests, re, json

PI = "http://192.168.1.163:8085"

# === Pi's cosmos/dashboard ===
print("=== Pi cosmos/dashboard ===")
r = requests.get(f"{PI}/cosmos/dashboard", timeout=8)
html = r.text
titles = re.findall(r'<title>(.*?)</title>', html)
print(f"Title: {titles}")
print(f"Size: {len(html)} chars")
fetches = re.findall(r"fetch\(['\"]([^'\"]+)['\"]", html)
print(f"fetch() calls ({len(fetches)}):")
for f in sorted(set(fetches)):
    print(f"  {f}")

print()

# === Check which agent endpoints from index.html actually exist ===
print("=== Endpoint check (local index.html xarm paths vs Pi) ===")

xarm_calls = [
    ("/xarm/move",              "POST", "/arm/move"),
    ("/xarm/preset/home",       "POST", "/arm/named/home"),
    ("/xarm/gripper",           "POST", None),
    ("/xarm/stop",              "POST", "/arm/stop"),
    ("/xarm/positions",         "GET",  "/arm/positions"),
    ("/xarm/status",            "GET",  "/arm/status"),
    ("/xarm/record/start",      "POST", None),
    ("/xarm/record/keyframe",   "POST", None),
    ("/xarm/record/stop",       "POST", None),
    ("/xarm/sequence/play",     "POST", None),
    ("/xarm/sequences",         "GET",  None),
    ("/xarm/home",              "POST", "/arm/home"),
    ("/camera/info",            "GET",  "/camera/status"),
    ("/camera/snapshot.json",   "GET",  "/camera/snapshot"),
    ("/system/nis/probe",       "GET",  None),
    ("/system/docker/install",  "POST", None),
    ("/system/service/start",   "POST", None),
    ("/system/logs",            "GET",  None),
    ("/system/exec",            "POST", None),
    ("/system/restart",         "POST", None),
    ("/offline/status",         "GET",  None),
    ("/offline/chat",           "POST", None),
    ("/offline/pull",           "POST", None),
    ("/skills",                 "GET",  None),
    ("/skills/reload",          "POST", None),
    ("/skill/invoke",           "POST", None),
    ("/sessions",               "GET",  None),
    ("/agent/chat",             "POST", None),
]

print(f"{'Old path':<30} {'Status':<12} {'Replacement'}")
print("-" * 70)
for old, method, replacement in xarm_calls:
    try:
        if method == "GET":
            resp = requests.get(f"{PI}{old}", timeout=3)
        else:
            resp = requests.post(f"{PI}{old}", json={}, timeout=3)
        exists = "EXISTS" if resp.status_code not in [404, 405] else f"404/405"
        rep_str = replacement if replacement else "REMOVED"
        print(f"{old:<30} {exists:<12} {rep_str}")
    except Exception as e:
        print(f"{old:<30} {'ERROR':<12} {str(e)[:30]}")
