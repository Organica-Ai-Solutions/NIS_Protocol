#!/usr/bin/env python3
"""
Patch cosmos_cookoff.py on Pi:
Replace the H100 direct Transfer2.5 fallback (172.16.1.83:8300) with
a call through NIS /cookoff/transfer — which NIS proxies to H100.
"""
import subprocess

COOKOFF = "/opt/neurolinux/cosmos_cookoff.py"

with open(COOKOFF) as f:
    src = f.read()

# Find and replace the H100 direct fallback block
OLD = '''\
    if result.get("error"):
        # H100 transfer direct fallback (port 8300) — async submit+poll to avoid timeout
        log.info("NIS transfer failed, trying H100 Transfer2.5 submit+poll at port 8300")
        try:
            ctrl = transfer_type if transfer_type in ("edge", "depth", "seg", "vis") else "edge"
            async with httpx.AsyncClient(timeout=30.0) as c:
                sr = await c.post(f"{H100_TRANSFER_URL}/transfer/submit", json={
                    "demo": "car_edge",
                    "control_type": ctrl,
                    "guidance": strength * 5.0,
                })
                if sr.status_code != 200:
                    raise Exception(f"submit HTTP {sr.status_code}")
                job_id = sr.json().get("job_id")
            log.info("Transfer2.5 job submitted: %s — polling...", job_id)
            # Poll up to 15 min
            for _ in range(45):
                await asyncio.sleep(20)
                async with httpx.AsyncClient(timeout=10.0) as c:
                    pr = await c.get(f"{H100_TRANSFER_URL}/transfer/status/{job_id}")
                    pd = pr.json()
                if pd.get("status") == "running":
                    continue
                if pd.get("video_b64"):
                    result = {
                        "source": "h100_transfer_direct",
                        "description": f"Cosmos Transfer2.5 {ctrl} transfer",
                        "transferred_image": pd.get("preview_b64", ""),
                        "video_base64": pd.get("video_b64", ""),
                        "all_videos": pd.get("all_videos", {}),
                        "status": "completed",
                    }
                    break
                raise Exception(f"Transfer2.5 job failed: {pd.get('error', pd)}")
            else:
                raise Exception("Transfer2.5 timed out after 15 min")'''

NEW = '''\
    if result.get("error"):
        # Route through NIS /cookoff/transfer — NIS proxies to H100 Transfer2.5 :8300
        log.info("NIS cosmos.transfer failed, routing through NIS /cookoff/transfer")
        try:
            ctrl = transfer_type if transfer_type in ("edge", "depth", "seg", "vis") else "edge"
            body = {
                "type": ctrl,
                "strength": strength,
            }
            if source_b64:
                body["source_image"] = source_b64
            if target_b64:
                body["target_image"] = target_b64
            # NIS /cookoff/transfer handles submit+poll internally (up to 15 min)
            async with httpx.AsyncClient(timeout=httpx.Timeout(
                connect=5.0, read=960.0, write=30.0, pool=5.0
            )) as c:
                r = await c.post(f"{NIS_URL}/cookoff/transfer", json=body)
                if r.status_code == 200:
                    result = r.json()
                    result.setdefault("source", "nis_transfer25")
                else:
                    raise Exception(f"NIS /cookoff/transfer HTTP {r.status_code}: {r.text[:80]}")'''

if OLD in src:
    src = src.replace(OLD, NEW)
    print("✅ Patch applied: H100 direct transfer → NIS /cookoff/transfer")
else:
    # Try a more targeted replacement of just the URL
    if 'f"{H100_TRANSFER_URL}/transfer/submit"' in src:
        print("⚠  Full pattern not found, trying targeted URL replacement...")
        src = src.replace(
            'f"{H100_TRANSFER_URL}/transfer/submit"',
            'f"{NIS_URL}/cookoff/transfer"  # routed through NIS'
        )
        print("  ✅ URL replaced")
    else:
        print("❌ Pattern not found — manual fix needed")

with open(COOKOFF, "w") as f:
    f.write(src)

r = subprocess.run(["python3", "-m", "py_compile", COOKOFF],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:300])
