"""Restore arm to correct stored home position immediately."""
import json, time, urllib.request, sys, os
sys.path.insert(0, '.')

PI = 'http://192.168.1.163:8085'

def pi_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(PI + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=15)
    return json.loads(r.read())

def pi_get(path):
    r = urllib.request.urlopen(PI + path, timeout=10)
    return json.loads(r.read())

def snap(name):
    os.makedirs('data/calib_frames', exist_ok=True)
    r = urllib.request.urlopen(PI + '/camera/snapshot', timeout=15)
    p = 'data/calib_frames/' + name
    with open(p, 'wb') as f:
        f.write(r.read())
    print("  Saved " + p + " (" + str(os.path.getsize(p)) + " bytes)")
    return p

print("Reading arm memory for correct home...")
data = pi_get('/arm/touch_poses')
poses = data.get('touch_poses') or data.get('poses') or {}
print("  Poses in memory: " + str(list(poses.keys())))

if 'home' in poses:
    home = {str(k): int(v) for k, v in poses['home'].items()}
    print("  Stored home: " + ' '.join('S'+k+'='+str(v) for k,v in sorted(home.items())))
else:
    # fallback to previously known correct home
    home = {"1": 100, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350}
    print("  No home in memory, using known calibrated values")
    print("  Home: " + ' '.join('S'+k+'='+str(v) for k,v in sorted(home.items())))

print()
print("Opening gripper first...")
pi_post('/arm/group_move', {'positions': {'1': 100}, 'duration_ms': 500})
time.sleep(0.8)

print("Moving to CORRECT home (user-calibrated)...")
home['1'] = 100  # gripper open
r = pi_post('/arm/group_move', {'positions': home, 'duration_ms': 2000})
print("  Result: " + str(r.get('ok', '?')))
time.sleep(2.5)

snap('correct_home.jpg')
print()
print("DONE — arm is at correct stored home.")
print("This is the real home: arm forward-reaching, NOT upright.")
