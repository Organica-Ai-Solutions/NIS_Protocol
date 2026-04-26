"""Quick X-axis sweep to find lighter position."""
import json, time, urllib.request, sys, os
sys.path.insert(0, '.')
from src.kinematics.hiwonder_ik import ik_to_servos

PI = 'http://192.168.1.163:8085'
os.makedirs('data/calib_frames', exist_ok=True)


def pi_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(PI + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=15)
    return json.loads(r.read())


def snap(name):
    r = urllib.request.urlopen(PI + '/camera/snapshot', timeout=15)
    p = 'data/calib_frames/' + name
    with open(p, 'wb') as f:
        f.write(r.read())
    sz = os.path.getsize(p)
    print("  Saved " + name + " (" + str(sz) + " bytes)")
    return p


def ki_move(x, y, z, pitch, ms=1000, g=None):
    s = ik_to_servos(x, y, z, pitch)
    if g is not None:
        s['1'] = g
    r = pi_post('/arm/group_move', {'positions': s, 'duration_ms': ms})
    sg = ' '.join('S' + k + '=' + str(v) for k, v in sorted(s.items()))
    ok = str(r.get('ok', '?'))
    print("  ki(" + str(x) + "," + str(y) + "," + str(z) + "," + str(pitch) + ") " + sg + " -> " + ok)


print("=== X SWEEP CALIBRATION ===")
print("Moving arm to HOME first...")
ki_move(0, 17, 20.5, 0, ms=1500, g=100)
time.sleep(2.0)
snap('ik_home.jpg')
print()

print("Sweeping X=0 to X=12cm at z=13cm (inspect height):")
print("Watch the arm — note which X aligns gripper over lighter")
print()

for x in [0, 3, 6, 9, 12]:
    print("x=" + str(x) + "cm:")
    ki_move(x, 17, 13.0, 0, ms=1000)
    time.sleep(2.0)
    snap('sweep_x' + str(x).zfill(2) + '.jpg')
    print()

print("Returning to HOME...")
ki_move(0, 17, 20.5, 0, ms=1500, g=100)
time.sleep(1.5)
print()
print("SWEEP DONE — frames saved to data/calib_frames/")
print("Files: sweep_x00.jpg, sweep_x03.jpg, sweep_x06.jpg, sweep_x09.jpg, sweep_x12.jpg")
