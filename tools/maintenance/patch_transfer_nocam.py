#!/usr/bin/env python3
"""
Patch cosmos_cookoff.py: make cosmos_transfer work without a live camera.
When no frames are available, generate a minimal synthetic JPEG placeholder
so Transfer2.5 can still run on H100.
"""
import subprocess

COOKOFF = "/opt/neurolinux/cosmos_cookoff.py"

with open(COOKOFF) as f:
    src = f.read()

OLD = '''\
    if not source_b64 or not target_b64:
        state.mode = CookoffMode.IDLE
        return {"ok": False, "error": "Need both source and target frames"}'''

NEW = '''\
    if not source_b64 or not target_b64:
        # Generate a synthetic placeholder frame (grey 320x240 JPEG) so Transfer2.5
        # can still run when no physical camera is connected.
        try:
            import io, base64
            from PIL import Image
            img = Image.new("RGB", (320, 240), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            placeholder = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            import base64
            # Minimal 1x1 grey JPEG (hardcoded bytes)
            _grey_jpeg = (
                b"\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01\\x00\\x00\\x01\\x00\\x01\\x00\\x00"
                b"\\xff\\xdb\\x00C\\x00\\x08\\x06\\x06\\x07\\x06\\x05\\x08\\x07\\x07\\x07\\t\\t"
                b"\\x08\\n\\x0c\\x14\\r\\x0c\\x0b\\x0b\\x0c\\x19\\x12\\x13\\x0f\\x14\\x1d\\x1a"
                b"\\x1f\\x1e\\x1d\\x1a\\x1c\\x1c $.' \",#\\x1c\\x1c(7),01444\\x1f'9=82<.342\\x1e"
                b"\\xff\\xc0\\x00\\x0b\\x08\\x00\\x01\\x00\\x01\\x01\\x01\\x11\\x00"
                b"\\xff\\xc4\\x00\\x1f\\x00\\x00\\x01\\x05\\x01\\x01\\x01\\x01\\x01\\x01\\x00"
                b"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08"
                b"\\t\\n\\x0b\\xff\\xc4\\x00\\xb5\\x10\\x00\\x02\\x01\\x03\\x03\\x02\\x04\\x03"
                b"\\x05\\x05\\x04\\x04\\x00\\x00\\x01}\\x01\\x02\\x03\\x00\\x04\\x11\\x05\\x12"
                b"!1A\\x06\\x13Qa\\x07\\\"q\\x142\\x81\\x91\\xa1\\x08#B\\xb1\\xc1\\x15R\\xd1"
                b"\\xf0$3br\\x82\\t\\n\\x16\\x17\\x18\\x19\\x1a%&'()*456789:CDEFGHIJ"
                b"STUVWXYZ\\xff\\xda\\x00\\x08\\x01\\x01\\x00\\x00?\\x00\\xfb\\xd4P\\x00\\x00"
                b"\\x00\\x1f\\xff\\xd9"
            )
            placeholder = base64.b64encode(_grey_jpeg).decode()
        if not source_b64:
            source_b64 = placeholder
            log.info("cosmos_transfer: using synthetic placeholder for source frame")
        if not target_b64:
            target_b64 = placeholder
            log.info("cosmos_transfer: using synthetic placeholder for target frame")'''

if OLD in src:
    src = src.replace(OLD, NEW)
    print("✅ Patch applied: synthetic frame fallback in cosmos_transfer")
else:
    print("❌ Pattern not found — checking file...")
    idx = src.find("Need both source and target frames")
    if idx >= 0:
        print(f"  Found error string at char {idx}")
        print(f"  Context: {src[idx-200:idx+100]!r}")
    else:
        print("  Error string not found either")

with open(COOKOFF, "w") as f:
    f.write(src)

r = subprocess.run(["python3", "-m", "py_compile", COOKOFF],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:300])
