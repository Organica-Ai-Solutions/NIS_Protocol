#!/usr/bin/env python3
"""
Patch /home/nvidia/routes/cookoff.py — replace cookoff_autonomous function
with improved version:
  - Multi-scan YOLO (2 scans, union)
  - Budget by successful picks, not total attempts
  - Cosmos post-pick double-check verification
  - Final Cosmos table-clear confirmation
  - Max 5 sweeps to prevent infinite loops
  - Cosmos fallback when YOLO finds nothing

Usage: python _patch_autonomous.py
"""
import json, subprocess, sys

SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "awesome-gpu-name"]

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(cmd: str, timeout: int = 30):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip(), r.returncode


NEW_FUNCTION = r'''
@router.post("/autonomous")
async def cookoff_autonomous(request: AutonomousRequest):
    """
    FULL AUTONOMOUS lighter sweep — improved v2.

    Improvements over v1:
      - Multi-scan YOLO (2 scans, union merge) for complete detection
      - Budget counts successful picks, not total attempts
      - Cosmos fallback when YOLO finds nothing (confirms table clear)
      - Cosmos post-pick double-check when YOLO still sees object
      - Final Cosmos table-clear verification after all picks
      - Max 5 outer sweeps prevents infinite loop

    Monitor live via: GET /events/stream?topics=cookoff,arm,cosmos
    """
    import httpx
    t_start       = time.time()
    logs: List[str] = []
    all_picks: List[Dict] = []
    total_retries = 0
    LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask"}
    MAX_SWEEPS = 5

    _publish_cookoff("autonomous_start", {
        "task":        request.task,
        "max_picks":   request.max_picks,
        "max_retries": request.max_retries,
        "execute_arm": request.execute_arm,
        "conf":        request.conf,
    })
    logger.info("[autonomous] START task=%r execute=%s max_picks=%d",
                request.task, request.execute_arm, request.max_picks)

    async with httpx.AsyncClient(timeout=25.0) as c:

        # Camera warmup — 2 frames to stabilise auto-exposure
        for _ in range(2):
            try:
                await c.get(f"{AGENT_URL}/camera/snapshot", timeout=4.0)
                await asyncio.sleep(0.3)
            except Exception:
                pass

        successful_picks = 0
        total_attempts   = 0
        table_clear      = False
        sweep_count      = 0

        # OUTER LOOP — re-scan after each sweep until table clear or budget gone
        while successful_picks < request.max_picks and not table_clear and sweep_count < MAX_SWEEPS:
            sweep_count += 1
            _publish_cookoff("sweep_start", {
                "sweep":            sweep_count,
                "successful_picks": successful_picks,
            })

            # 1. MULTI-SCAN YOLO — 2 passes, merge unique detections ─────────
            scan1 = await _yolo_scan_nis(
                "lighter,bottle,cup,vase,flask,bin,bowl", conf=request.conf
            )
            await asyncio.sleep(0.4)
            scan2 = await _yolo_scan_nis(
                "lighter,bottle,cup,vase,flask,bin,bowl",
                conf=max(0.04, request.conf * 0.75),  # slightly lower conf on 2nd pass
            )

            dets1 = scan1.get("detections", [])
            dets2 = scan2.get("detections", [])
            frame_w   = scan1.get("frame_w", 1280)
            scene_ctx = scan1.get("scene_context", "")

            # Union: add det2 only if no similar label within 80px in dets1
            merged_dets = list(dets1)
            for d2 in dets2:
                if not any(
                    abs(d2["cx"] - d1["cx"]) < 80
                    and d2.get("label", "") == d1.get("label", "")
                    for d1 in dets1
                ):
                    merged_dets.append(d2)

            lighter_dets = sorted(
                [d for d in merged_dets
                 if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
                 and "bin" not in d.get("label", "").lower()],
                key=lambda d: d.get("cx", 640),
            )
            bin_det = next(
                (d for d in merged_dets if "bin" in d.get("label", "").lower()), None
            )

            _publish_cookoff("autonomous_scan", {
                "sweep":       sweep_count,
                "n_lighters":  len(lighter_dets),
                "n_total_det": len(merged_dets),
                "n_scan1":     len(dets1),
                "n_scan2":     len(dets2),
                "scene":       scene_ctx[:120],
            })
            logs.append(
                f"Sweep {sweep_count}: YOLO merged={len(lighter_dets)} lighters "
                f"(scan1={len(dets1)} scan2={len(dets2)}) | {scene_ctx[:80]}"
            )

            # If YOLO finds nothing — ask Cosmos to double-check ──────────────
            if not lighter_dets:
                snap_b64 = await _get_snap_b64(c)
                cosmos_check, cosmos_check_reasoning = await _cosmos_plan_positions(
                    c, snap_b64,
                    "Look very carefully at the table surface. "
                    "Are there any lighters, bottles, or small cylindrical objects still present? "
                    "List any remaining objects with pixel position [cx,cy]. "
                    "If nothing remains, say 'table is clear'.",
                )
                if cosmos_check_reasoning:
                    _publish_cookoff("reasoning_start", {"task": "table clear check"})
                    await _stream_reasoning(cosmos_check_reasoning)

                if not cosmos_check:
                    table_clear = True
                    logs.append(f"Sweep {sweep_count}: YOLO+Cosmos both confirm table clear")
                    _publish_cookoff("table_clear", {"sweep": sweep_count})
                    break
                else:
                    # Cosmos found objects YOLO missed — use them as targets
                    lighter_dets = [
                        {
                            "cx": p["cx"], "cy": p.get("cy", 360),
                            "label": p["label"], "conf": 0.50, "color": "",
                        }
                        for p in cosmos_check
                        if "bin" not in p["label"].lower()
                    ]
                    logs.append(
                        f"Sweep {sweep_count}: Cosmos found {len(lighter_dets)} "
                        "objects YOLO missed"
                    )
                    if not lighter_dets:
                        table_clear = True
                        break

            # 2. COSMOS REASONING — position planning ─────────────────────────
            snap_b64 = await _get_snap_b64(c)
            cosmos_positions, cosmos_reasoning = await _cosmos_plan_positions(
                c, snap_b64,
                f"There are approximately {len(lighter_dets)} lighters on the table. "
                "Identify each lighter and provide its pixel position [cx,cy]. "
                "Order them left-to-right for efficient sequential picking.",
            )
            _publish_cookoff("cosmos_positions", {
                "n":         len(cosmos_positions),
                "positions": cosmos_positions[:8],
            })
            if cosmos_reasoning:
                _publish_cookoff("reasoning_start", {"task": "pick planning"})
                await _stream_reasoning(cosmos_reasoning)
            logs.append(
                f"Cosmos: {len(cosmos_positions)} positions for "
                f"{len(lighter_dets)} YOLO detections"
            )

            # 3. MERGE YOLO + COSMOS ──────────────────────────────────────────
            merged: List[Dict] = []
            for ld in lighter_dets:
                entry = dict(ld)
                for cp in cosmos_positions:
                    if abs(cp["cx"] - ld["cx"]) < 120:
                        entry["cosmos_cx"] = cp["cx"]
                        entry["cosmos_cy"] = cp.get("cy", ld.get("cy", 360))
                        break
                merged.append(entry)

            # Add Cosmos-only positions not seen by YOLO (with hallucination guard)
            for cp in cosmos_positions:
                already = any(
                    abs(cp["cx"] - m.get("cosmos_cx", m["cx"])) < 100
                    for m in merged
                )
                if not already and 50 < cp["cx"] < frame_w - 50:
                    merged.append({
                        "label":     cp["label"],
                        "cx":        cp["cx"],
                        "cy":        cp.get("cy", 360),
                        "conf":      0.50,
                        "cosmos_cx": cp["cx"],
                        "cosmos_cy": cp.get("cy", 360),
                        "color":     "",
                    })

            merged = sorted(merged, key=lambda d: d.get("cosmos_cx", d["cx"]))

            # 4. PICK EACH LIGHTER ────────────────────────────────────────────
            any_picked_this_sweep = False

            for lighter in merged:
                if successful_picks >= request.max_picks:
                    break

                base_cx    = lighter.get("cosmos_cx", lighter["cx"])
                place_zone = _dest_to_place_zone("bin", bin_det, frame_w)

                pick_result: Dict = {
                    "lighter_idx": total_attempts,
                    "sweep":       sweep_count,
                    "label":       lighter.get("label", "lighter"),
                    "color":       lighter.get("color", ""),
                    "yolo_cx":     lighter["cx"],
                    "cosmos_cx":   lighter.get("cosmos_cx"),
                    "place_zone":  place_zone,
                    "attempts":    [],
                    "success":     False,
                }

                _publish_cookoff("picking", {
                    "pick_n":    successful_picks + 1,
                    "sweep":     sweep_count,
                    "yolo_cx":   lighter["cx"],
                    "cosmos_cx": lighter.get("cosmos_cx"),
                    "place":     place_zone,
                })
                logger.info("[autonomous] picking #%d label=%s cx=%d place=%s",
                            total_attempts + 1, lighter.get("label"), base_cx, place_zone)

                # RETRY LOOP ──────────────────────────────────────────────────
                S6_ADJUSTMENTS = [0, +15, -15]

                for attempt in range(request.max_retries):

                    # Re-scan on retry — get fresh position
                    if attempt > 0:
                        await asyncio.sleep(1.0)
                        rs = await _yolo_scan_nis(
                            "lighter,bottle,cup,vase,flask", conf=request.conf
                        )
                        frame_w = rs.get("frame_w", frame_w)
                        rs_lighters = sorted(
                            [d for d in rs.get("detections", [])
                             if any(a in d.get("label", "").lower()
                                    for a in LIGHTER_ALIASES)
                             and "bin" not in d.get("label", "").lower()],
                            key=lambda d: abs(d["cx"] - base_cx),
                        )
                        if rs_lighters:
                            base_cx = rs_lighters[0]["cx"]
                        total_retries += 1

                    adj = S6_ADJUSTMENTS[attempt % len(S6_ADJUSTMENTS)]
                    s6  = max(400, min(620, _cx_to_s6(base_cx, frame_w) + adj))

                    logger.info("[autonomous]   attempt %d: cx=%d s6=%d adj=%+d",
                                attempt, base_cx, s6, adj)

                    attempt_log: Dict = {
                        "attempt":     attempt,
                        "cx":          base_cx,
                        "s6":          s6,
                        "s6_adj":      adj,
                        "pick_ok":     False,
                        "disappeared": False,
                    }

                    if request.execute_arm:
                        pk = await _run_ik_pick(c, s6=s6, place=place_zone)
                        attempt_log["pick_ok"] = pk.get("ok", False)
                        attempt_log["steps"]   = pk.get("steps", [])

                        await asyncio.sleep(1.5)

                        # Primary verify: YOLO disappearance check
                        disappeared = await _verify_pick_disappeared(
                            base_cx, frame_w, request.conf
                        )

                        # Secondary verify: if YOLO still sees it, ask Cosmos
                        if not disappeared:
                            try:
                                snap_check = await _get_snap_b64(c)
                                cosmos_check, _ = await _cosmos_plan_positions(
                                    c, snap_check,
                                    f"Is there still a lighter at approximately "
                                    f"pixel x={base_cx} on the table? "
                                    "List remaining lighters with positions.",
                                )
                                # Cosmos confirms gone if no position near base_cx
                                if not any(
                                    abs(cp["cx"] - base_cx) < 100
                                    for cp in cosmos_check
                                ):
                                    disappeared = True
                                    attempt_log["cosmos_confirmed_gone"] = True
                            except Exception:
                                pass

                        attempt_log["disappeared"] = disappeared

                        _publish_cookoff("pick_attempt", {
                            "pick_n":      successful_picks + 1,
                            "attempt":     attempt,
                            "s6":          s6,
                            "pick_ok":     attempt_log["pick_ok"],
                            "disappeared": disappeared,
                        })

                        pick_result["attempts"].append(attempt_log)

                        if disappeared:
                            pick_result["success"] = True
                            logs.append(
                                f"  Pick #{total_attempts+1} SUCCESS "
                                f"attempt={attempt} s6={s6}"
                            )
                            break
                        else:
                            logs.append(
                                f"  Pick #{total_attempts+1} MISS "
                                f"attempt={attempt} s6={s6} — retrying"
                            )

                    else:
                        # Simulation — always succeeds
                        attempt_log["pick_ok"]     = True
                        attempt_log["disappeared"] = True
                        pick_result["attempts"].append(attempt_log)
                        pick_result["success"] = True
                        logs.append(
                            f"  [SIM] Pick #{total_attempts+1} s6={s6} cx={base_cx}"
                        )
                        break

                all_picks.append(pick_result)
                total_attempts += 1

                if pick_result["success"]:
                    successful_picks += 1
                    any_picked_this_sweep = True
                    _publish_cookoff("pick_success", {
                        "n_successful": successful_picks,
                        "remaining_budget": request.max_picks - successful_picks,
                    })

                # Brief settle between objects
                if lighter is not merged[-1]:
                    await asyncio.sleep(1.5)

            # If arm is enabled and nothing succeeded this sweep — break to
            # avoid infinite retrying when all picks consistently miss
            if request.execute_arm and not any_picked_this_sweep:
                logs.append(
                    f"Sweep {sweep_count}: 0 successful picks — stopping outer loop"
                )
                _publish_cookoff("no_progress", {"sweep": sweep_count})
                break

        # end while outer loop

    # 5. FINAL COSMOS TABLE-CLEAR VERIFICATION ────────────────────────────────
    _publish_cookoff("final_check", {"msg": "Final Cosmos table-clear verification…"})
    try:
        async with httpx.AsyncClient(timeout=15.0) as cv:
            snap_final = await _get_snap_b64(cv)
            final_pos, final_reasoning = await _cosmos_plan_positions(
                cv, snap_final,
                "Final check: look at the entire table surface carefully. "
                "Are there any lighters, bottles, or objects that have NOT been "
                "placed in the bin? List positions if any remain, or confirm clear.",
            )
            if final_reasoning:
                _publish_cookoff("reasoning_start", {"task": "final table-clear check"})
                await _stream_reasoning(final_reasoning)

            if not final_pos:
                table_clear = True
                logs.append("Final Cosmos check: table clear confirmed")
                _publish_cookoff("table_clear", {"source": "cosmos_final"})
            else:
                remaining_n = len([p for p in final_pos if "bin" not in p["label"].lower()])
                logs.append(
                    f"Final Cosmos: {remaining_n} object(s) may remain on table"
                )
    except Exception as e:
        logger.warning("[autonomous] final cosmos check failed: %s", e)
        # Fallback: infer from results
        if not table_clear:
            table_clear = (successful_picks > 0 and successful_picks == len(all_picks))

    latency_ms = round((time.time() - t_start) * 1000)
    successes  = sum(1 for p in all_picks if p["success"])

    _publish_cookoff("autonomous_done", {
        "table_clear":   table_clear,
        "picks_total":   total_attempts,
        "picks_success": successes,
        "total_retries": total_retries,
        "sweeps":        sweep_count,
        "latency_ms":    latency_ms,
    })
    logger.info(
        "[autonomous] DONE picks=%d/%d sweeps=%d retries=%d clear=%s lat=%dms",
        successes, total_attempts, sweep_count, total_retries, table_clear, latency_ms,
    )

    return {
        "ok":            successes > 0 or (total_attempts == 0 and table_clear),
        "task":          request.task,
        "table_clear":   table_clear,
        "picks_total":   total_attempts,
        "picks_success": successes,
        "sweeps":        sweep_count,
        "total_retries": total_retries,
        "all_picks":     all_picks,
        "logs":          logs,
        "latency_ms":    latency_ms,
        "timestamp":     time.time(),
    }
'''


def main():
    print("=== Patching cookoff_autonomous on H100 ===\n")

    import base64
    b64 = base64.b64encode(NEW_FUNCTION.encode()).decode()

    # Write a Python patch script to H100 via stdin pipe
    patch_code = (
        "import base64\n"
        "new_func = base64.b64decode('" + b64 + "').decode()\n"
        "with open('/home/nvidia/routes/cookoff.py', 'r') as f:\n"
        "    content = f.read()\n"
        "marker = '@router.post(\"/autonomous\")'\n"
        "idx = content.find(marker)\n"
        "if idx == -1:\n"
        "    print('ERROR: marker not found')\n"
        "    exit(1)\n"
        "new_content = content[:idx] + new_func.lstrip('\\n')\n"
        "with open('/home/nvidia/routes/cookoff.py', 'w') as f:\n"
        "    f.write(new_content)\n"
        "print('OK replaced cookoff_autonomous')\n"
        "print(f'File: {len(new_content)} chars')\n"
    )

    result = subprocess.run(
        SSH + ["python3"],
        input=patch_code,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (result.stdout + result.stderr).strip()
    print(f"  {out}")

    if "OK" not in out:
        print("  FAIL — check output above")
        return

    # Verify line count
    out2, _ = run("wc -l /home/nvidia/routes/cookoff.py")
    print(f"  Lines after patch: {out2}")

    # Quick syntax check
    out3, rc = run("/home/nvidia/organica-ai/venv/bin/python3 -m py_compile /home/nvidia/routes/cookoff.py 2>&1 && echo 'syntax OK'")
    print(f"  Syntax check: {out3}")
    if rc != 0:
        print("  SYNTAX ERROR — fix before restart")
        return

    # Restart NIS
    print("\n=== Restarting NIS ===")
    restart = (
        "tmux kill-session -t nis_h100 2>/dev/null; "
        "pkill -f 'uvicorn main:app.*8090' 2>/dev/null; "
        "sleep 2; "
        "tmux new-session -d -s nis_h100 "
        "'cd /home/nvidia && set -a && source /home/nvidia/.env.h100 && set +a && "
        "YOLO_MODEL=yolov8x.pt /home/nvidia/organica-ai/venv/bin/uvicorn main:app "
        "--host 0.0.0.0 --port 8090 --log-level info --workers 1 "
        "2>&1 | tee /tmp/nis_8090.log'"
    )
    out4, _ = run(restart, timeout=20)
    print(f"  Restart: {out4 or 'sent'}")

    import time
    print("  Waiting 8s for startup...")
    time.sleep(8)

    out5, _ = run("curl -sf http://localhost:8090/health 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f\"health ok v{d.get(chr(118),chr(63))}\")' 2>/dev/null || echo 'not ready'")
    print(f"  Health: {out5}")

    print("\n=== Done ===")
    print("  Improved autonomous pipeline deployed.")
    print("  Changes:")
    print("    - YOLO: 2 scans merged (union) for full detection")
    print("    - Budget: counts successful picks, not total attempts")
    print("    - Cosmos: fallback when YOLO finds nothing")
    print("    - Cosmos: post-pick double-check verification")
    print("    - Cosmos: final table-clear confirmation")
    print("    - Max 5 sweeps to prevent infinite loop")


if __name__ == "__main__":
    main()
