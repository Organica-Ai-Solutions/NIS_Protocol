#!/usr/bin/env python3
"""
Deploy neurohub-ui Next.js build to Pi.
Pushes .next/ standalone build + server.js via chunked base64 over /system/shell.
Usage: python _deploy_neurohub.py
"""
import base64, json, pathlib, sys, time, urllib.request

NIS   = "http://192.168.1.163:8000"
AGENT = "http://192.168.1.163:8085"
ZIP   = pathlib.Path("C:/Users/DiegoTorres/Desktop/NIS_Protocol/neurohub_next.zip")
PI_DIR = "/opt/neurolinux/neurohub-ui"
CHUNK = 40_000

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def shell(cmd, timeout=40):
    body = json.dumps({"cmd": cmd}).encode()
    req  = urllib.request.Request(NIS + "/system/shell", data=body,
                                   headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(r.read())
        return (d.get("stdout") or d.get("output") or d.get("result") or "").strip(), d.get("returncode", 0)
    except Exception as e:
        return str(e), -1


def push_zip(local: pathlib.Path, remote: str):
    data   = local.read_bytes()
    chunks = [data[i:i+CHUNK] for i in range(0, len(data), CHUNK)]
    print(f"  Pushing {local.name} ({len(data)//1024}KB, {len(chunks)} chunks)...")
    for idx, chunk in enumerate(chunks):
        b64  = base64.b64encode(chunk).decode("ascii")
        mode = "wb" if idx == 0 else "ab"
        cmd  = (f"python3 -c \""
                f"import base64; "
                f"f=open('{remote}','{mode}'); "
                f"f.write(base64.b64decode('{b64}')); "
                f"f.close()\"")
        _, rc = shell(cmd, timeout=45)
        if rc != 0:
            print(f"  FAIL at chunk {idx}"); return False
        print(f"  chunk {idx+1}/{len(chunks)}", end="\r")
    print()
    return True


def main():
    print("\n=== Deploy neurohub-ui to Pi ===\n")

    # [1] Push zip
    print("[1] Pushing zip...")
    tmp_zip = "/tmp/neurohub_next.zip"
    if not push_zip(ZIP, tmp_zip):
        print("Push failed"); sys.exit(1)

    # Verify zip size
    out, _ = shell(f"wc -c {tmp_zip}")
    print(f"  Remote zip: {out}")

    # [2] Extract to neurohub-ui directory
    print("\n[2] Extracting...")
    out, rc = shell(
        f"cd {PI_DIR} && python3 -c \""
        f"import zipfile; "
        f"zipfile.ZipFile('{tmp_zip}').extractall('.')\" "
        f"&& echo extracted",
        timeout=60,
    )
    if "extracted" not in out:
        print(f"  Extract failed: {out}"); sys.exit(1)
    print(f"  {out}")

    # [3] Cleanup
    shell(f"rm -f {tmp_zip}")

    # [4] Restart neurohub-ui
    print("\n[3] Restarting neurohub-ui service...")
    # Use fuser since sudo not available
    out, _ = shell(
        "fuser -k 3000/tcp 2>/dev/null; sleep 1; "
        "python3 -c \""
        "import subprocess,os; "
        "p=subprocess.Popen("
        "['node','/opt/neurolinux/neurohub-ui/server.js'],"
        "env={**os.environ,'PORT':'3000','HOSTNAME':'0.0.0.0'},"
        "stdout=open('/tmp/neurohub.log','w'),"
        "stderr=subprocess.STDOUT,"
        "start_new_session=True,"
        "close_fds=True); "
        "print('pid='+str(p.pid))\"",
        timeout=15,
    )
    print(f"  {out or '(no output)'}")

    # [5] Wait and verify
    print("\n[4] Waiting 8s for startup...")
    for i in range(8):
        time.sleep(1); print(f"  {'.'*(i+1)}", end="\r")
    print()

    try:
        r = urllib.request.urlopen(AGENT.replace("8085", "3000") + "/cosmos", timeout=8)
        html = r.read(500).decode("utf-8", errors="replace")
        has_auto = "Autonomous" in html or "autonomous" in html
        print(f"  /cosmos: HTTP {r.status} — Autonomous tab: {'✓' if has_auto else '? (checking JS bundle)'}")
    except Exception as e:
        print(f"  /cosmos: {e}")

    print("\n=== Done ===")
    print(f"  Dashboard: http://192.168.1.163:3000/cosmos")


if __name__ == "__main__":
    main()
