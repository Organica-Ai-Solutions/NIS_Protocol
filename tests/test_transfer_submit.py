#!/usr/bin/env python3
import urllib.request, json, time

NIS = "http://localhost:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(NIS+path, data=data,
                                 headers={"Content-Type":"application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    return json.loads(r.read())

print("Submitting Transfer2.5 job...")
d = post("/cookoff/transfer", {"type": "edge", "strength": 0.7})
print("Submit:", json.dumps(d))

job_id = d.get("job_id")
if not job_id:
    print("No job_id returned")
else:
    print(f"Job {job_id} — polling...")
    for i in range(4):
        time.sleep(5)
        r2 = urllib.request.urlopen(NIS+"/cookoff/transfer/status/"+job_id, timeout=10)
        s = json.loads(r2.read())
        status = s.get("status")
        print(f"  Poll {i+1}: status={status} ok={s.get('ok')}")
        if status not in ("running", "submitted"):
            break
    print("Done.")
