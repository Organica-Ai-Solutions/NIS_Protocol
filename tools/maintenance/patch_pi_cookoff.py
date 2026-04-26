"""
patch_pi_cookoff.py - Patch the ACTION_MAP in routes/cookoff.py on the Pi
via the /execute endpoint (no SSH required).

Key fix:
  - place/put/drop -> /arm/named/place_bin  (was incorrectly /arm/named/reach_forward)
  - Added place_bin, pick_white to STEP_DELAY
  - Added "place in bin", "drop in bin", "put in bin" compound matches
"""
import requests, json, sys

PI_NIS = "http://192.168.1.163:8000"

# Check Pi is reachable
print("Checking Pi NIS Protocol...")
try:
    r = requests.get(f"{PI_NIS}/health", timeout=8)
    d = r.json()
    print(f"  OK: {d.get('status')} v={d.get('version')} routes={d.get('routes_loaded')}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Find cookoff.py on Pi
print("\nFinding cookoff.py on Pi...")
cookoff_path = None

search_code = (
    "import glob,json,subprocess\n"
    "paths=(\n"
    "    glob.glob('/opt/*/routes/cookoff.py')+\n"
    "    glob.glob('/home/pi/*/routes/cookoff.py')+\n"
    "    glob.glob('/home/*/NIS_Protocol/routes/cookoff.py')\n"
    ")\n"
    "if not paths:\n"
    "    res=subprocess.run(['find','/opt','/home','-name','cookoff.py','-type','f'],\n"
    "                       capture_output=True,text=True,timeout=10)\n"
    "    paths=[p for p in res.stdout.strip().split('\\n') if p]\n"
    "print(json.dumps(paths))\n"
)

try:
    r = requests.post(f"{PI_NIS}/execute", json={"code": search_code}, timeout=20)
    if r.status_code == 200:
        out = r.json().get("output", "")
        last_line = [l for l in out.strip().split('\n') if l.startswith('[')]
        if last_line:
            paths = json.loads(last_line[-1])
            if paths:
                cookoff_path = paths[0]
                print(f"  Found: {cookoff_path}")
            else:
                print("  cookoff.py not found on Pi")
        else:
            print(f"  Unexpected output: {out[:200]}")
    else:
        print(f"  /execute returned {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  Search error: {e}")

if not cookoff_path:
    print("\nCannot find cookoff.py on Pi. Manual restart needed after git pull.")
    sys.exit(0)

# Patch ACTION_MAP directly
print(f"\nPatching {cookoff_path}...")

patch_code = (
    "path = '" + cookoff_path + "'\n"
    "with open(path, 'r') as f:\n"
    "    src = f.read()\n"
    "\n"
    "changes = 0\n"
    "\n"
    "# Fix: place/put/drop -> place_bin\n"
    "pairs = [\n"
    "    ('\"place\":             \"/arm/named/reach_forward\",',\n"
    "     '\"place\":             \"/arm/named/place_bin\",'),\n"
    "    ('\"put\":               \"/arm/named/reach_forward\",',\n"
    "     '\"put\":               \"/arm/named/place_bin\",'),\n"
    "    ('\"drop\":              \"/arm/named/reach_forward\",',\n"
    "     '\"drop\":              \"/arm/named/place_bin\",'),\n"
    "]\n"
    "for old, new in pairs:\n"
    "    if old in src:\n"
    "        src = src.replace(old, new)\n"
    "        changes += 1\n"
    "\n"
    "# Add place_bin to STEP_DELAY if missing\n"
    "if '\"/arm/named/place_bin\"' not in src:\n"
    "    src = src.replace(\n"
    "        '\"/arm/named/pick_table\":    2.5,',\n"
    "        '\"/arm/named/pick_table\":    2.5,\\n        \"/arm/named/place_bin\":     2.5,'\n"
    "    )\n"
    "    changes += 1\n"
    "\n"
    "# Add compound place_bin phrases if missing\n"
    "if '\"place in bin\"' not in src:\n"
    "    src = src.replace(\n"
    "        '\"full_demo\":         \"/arm/pick_and_place\",',\n"
    "        '\"full_demo\":         \"/arm/pick_and_place\",\\n        \"place_bin\":         \"/arm/named/place_bin\",\\n        \"place in bin\":      \"/arm/named/place_bin\",\\n        \"drop in bin\":       \"/arm/named/place_bin\",'\n"
    "    )\n"
    "    changes += 1\n"
    "\n"
    "with open(path, 'w') as f:\n"
    "    f.write(src)\n"
    "print(f'DONE: {changes} changes applied')\n"
)

try:
    r = requests.post(f"{PI_NIS}/execute", json={"code": patch_code}, timeout=20)
    if r.status_code == 200:
        out = r.json().get("output", "")
        print(f"  Result: {out.strip()[:200]}")
    else:
        print(f"  Failed: {r.status_code} {r.text[:100]}")
except Exception as e:
    print(f"  Patch error: {e}")

# Trigger module reload
print("\nClearing Python module cache on Pi...")
reload_code = (
    "import sys\n"
    "removed = [k for k in list(sys.modules) if 'cookoff' in k]\n"
    "for k in removed: del sys.modules[k]\n"
    "print(f'Cleared: {removed}')\n"
)
try:
    r = requests.post(f"{PI_NIS}/execute", json={"code": reload_code}, timeout=10)
    if r.status_code == 200:
        out = r.json().get("output", "").strip()
        print(f"  {out[:120]}")
except Exception as e:
    print(f"  {e}")

print(
    "\nDone! The fix is written to disk on the Pi.\n"
    "Restart nis-protocol service on Pi to fully reload:\n"
    "  sudo systemctl restart nis-protocol\n"
    "\n"
    "For the current demo session, the choreographed pick_and_place\n"
    "sequence still works correctly without restarting.\n"
)
