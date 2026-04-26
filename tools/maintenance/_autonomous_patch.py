

# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS COOKOFF — Full lighter sweep with Cosmos guidance + retry logic
# Picks ALL lighters regardless of position, drops each in bin. Retry 3×/pick.
# ══════════════════════════════════════════════════════════════════════════════

async def _get_snap_b64(client) -> str:
    """Take Pi camera snapshot and return base64 string."""
    try:
        r = await client.get(f"{AGENT_URL}/camera/snapshot", timeout=6.0)
        if r.status_code == 200:
            return r.json().get("image_base64", "")
    except Exception as e:
        logger.warning("_get_snap_b64 failed: %s", e)
    return ""


def _cosmos_extract_positions(reasoning: str) -> List[Dict]:
    """
    Parse Cosmos /robot-plan reasoning text for [cx,cy] lighter positions.
    Handles: word [x,y], word (x,y), bare [x,y], cx=X cy=Y patterns.
    Returns sorted list of {"cx":int,"cy":int,"label":str} left-to-right.
    """
    positions: List[Dict] = []
    seen: set = set()
    patterns = [
        (_re.compile(r"(\w+)\s*\[(\d+)\s*,\s*(\d+)\]", _re.I), 3),
        (_re.compile(r"(\w+)\s*\((\d+)\s*,\s*(\d+)\)",  _re.I), 3),
        (_re.compile(r"\[(\d+)\s*,\s*(\d+)\]"),                  2),
        (_re.compile(r"\((\d+)\s*,\s*(\d+)\)"),                  2),
    ]
    for pat, ngroups in patterns:
        for m in pat.finditer(reasoning):
            g = m.groups()
            if ngroups == 3:
                label, x, y = g[0].lower(), int(g[1]), int(g[2])
            else:
                label, x, y = "object", int(g[0]), int(g[1])
            if not (10 < x < 2000 and 10 < y < 2000):
                continue
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            positions.append({"cx": x, "cy": y, "label": label})
    return sorted(positions, key=lambda p: p["cx"])


async def _cosmos_plan_positions(client, snap_b64: str, task: str) -> List[Dict]:
    """
    Call Cosmos /robot-plan with current camera frame.
    Merge structured trajectory + reasoning-parsed positions.
    Returns list of {"cx":int,"cy":int,"label":str} sorted left-to-right.
    """
    if not snap_b64:
        return []
    try:
        r = await client.post(f"{H100_REASON_URL}/robot-plan", json={
            "command":      task,
            "image_base64": snap_b64,
            "robot_type":   "xarm",
        }, timeout=20.0)
        if r.status_code == 200:
            d = r.json()
            positions: List[Dict] = []
            # Structured trajectory from Cosmos
            for t in (d.get("trajectory") or []):
                pt = t.get("point_2d") or []
                if len(pt) >= 2:
                    positions.append({"cx": int(pt[0]), "cy": int(pt[1]),
                                      "label": t.get("label", "object").lower()})
            # Also parse reasoning/plan text for additional positions
            reasoning = d.get("reasoning") or d.get("plan") or ""
            for ep in _cosmos_extract_positions(reasoning):
                if not any(abs(ep["cx"] - p["cx"]) < 50 for p in positions):
                    positions.append(ep)
            return sorted(positions, key=lambda p: p["cx"])
    except Exception as e:
        logger.warning("_cosmos_plan_positions failed: %s", e)
    return []


async def _verify_pick_disappeared(expected_cx: int, frame_w: int,
                                   conf: float = 0.08) -> bool:
    """
    Post-pick YOLO rescan — check if a lighter disappeared near expected_cx.
    Returns True if lighter is gone from pick position (pick succeeded).
    """
    try:
        scan = await _yolo_scan_nis("lighter,bottle,cup,vase,flask", conf=conf)
        LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask"}
        lighters = [
            d for d in scan.get("detections", [])
            if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
            and "bin" not in d.get("label", "").lower()
        ]
        near = [d for d in lighters if abs(d.get("cx", 9999) - expected_cx) < 110]
        return len(near) == 0   # nothing at pick position → object was picked
    except Exception as e:
        logger.warning("_verify_pick_disappeared failed: %s", e)
        return False


class AutonomousRequest(BaseModel):
    task:        str   = Field(default="pick all lighters and put them in the bin",
                               description="High-level task description")
    max_picks:   int   = Field(default=10,   ge=1, le=20,
                               description="Max total picks before stopping")
    max_retries: int   = Field(default=3,    ge=1, le=5,
                               description="Max retry attempts per lighter")
    execute_arm: bool  = Field(default=True,
                               description="False = simulation/dry-run only")
    conf:        float = Field(default=0.08, ge=0.01, le=0.5,
                               description="YOLO detection confidence threshold")


@router.post("/autonomous")
async def cookoff_autonomous(request: AutonomousRequest):
    """
    FULL AUTONOMOUS lighter sweep.

    Flow per loop iteration:
      1. YOLO+GDINO scan  →  find all lighters on table
      2. Cosmos /robot-plan  →  semantic [cx,cy] for every lighter
      3. Merge YOLO + Cosmos  →  sorted left-to-right
      4. For each lighter: pick (retry x3) → verify (disappeared?) → bin drop
      5. Re-scan at top → keep going until table clear or max_picks reached

    Retry strategy (per lighter):
      attempt 0: S6 from YOLO/Cosmos cx
      attempt 1: S6 + 15  (shift right)
      attempt 2: S6 - 15  (shift left)

    Monitor live via: GET /events/stream?topics=cookoff,arm,cosmos
    """
    import httpx
    t_start       = time.time()
    logs: List[str] = []
    all_picks: List[Dict] = []
    total_retries = 0
    LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask"}

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

        pick_count  = 0
        table_clear = False

        # OUTER LOOP: re-scan after each complete sweep until table clear
        while pick_count < request.max_picks and not table_clear:

            # 1. SCAN ─────────────────────────────────────────────────────────
            scan    = await _yolo_scan_nis(
                "lighter,bottle,cup,vase,flask,bin,bowl", conf=request.conf
            )
            dets    = scan.get("detections", [])
            frame_w = scan.get("frame_w", 1280)
            scene_ctx = scan.get("scene_context", "")

            lighter_dets = sorted(
                [d for d in dets
                 if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
                 and "bin" not in d.get("label", "").lower()],
                key=lambda d: d.get("cx", 640)
            )
            bin_det = next(
                (d for d in dets if "bin" in d.get("label", "").lower()), None
            )

            _publish_cookoff("autonomous_scan", {
                "loop_pick":   pick_count,
                "n_lighters":  len(lighter_dets),
                "n_total_det": len(dets),
                "scene":       scene_ctx[:120],
            })
            logs.append(
                f"Loop {pick_count}: YOLO sees {len(lighter_dets)} lighters | {scene_ctx[:80]}"
            )

            if not lighter_dets:
                table_clear = True
                logs.append("Table clear — no lighters detected")
                break

            # 2. COSMOS POSITIONS ─────────────────────────────────────────────
            snap_b64 = await _get_snap_b64(c)
            cosmos_positions = await _cosmos_plan_positions(
                c, snap_b64,
                "Identify all lighters on the table. "
                "For each lighter provide its pixel position [cx,cy] in image coordinates. "
                "List them left-to-right."
            )
            logs.append(
                f"Cosmos: {len(cosmos_positions)} positions — {str(cosmos_positions[:4])}"
            )
            _publish_cookoff("cosmos_positions", {
                "n":         len(cosmos_positions),
                "positions": cosmos_positions[:5],
            })

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

            # Cosmos-only detections not seen by YOLO (hallucination guard: ≥50px from edge)
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
            for lighter in merged:
                if pick_count >= request.max_picks:
                    break

                base_cx    = lighter.get("cosmos_cx", lighter["cx"])
                place_zone = _dest_to_place_zone("bin", bin_det, frame_w)

                pick_result: Dict = {
                    "lighter_idx": pick_count,
                    "label":       lighter.get("label", "lighter"),
                    "color":       lighter.get("color", ""),
                    "yolo_cx":     lighter["cx"],
                    "cosmos_cx":   lighter.get("cosmos_cx"),
                    "place_zone":  place_zone,
                    "attempts":    [],
                    "success":     False,
                }

                _publish_cookoff("picking", {
                    "pick_n":    pick_count + 1,
                    "yolo_cx":  lighter["cx"],
                    "cosmos_cx": lighter.get("cosmos_cx"),
                    "place":    place_zone,
                })
                logger.info("[autonomous] picking #%d label=%s cx=%d place=%s",
                            pick_count + 1, lighter.get("label"), base_cx, place_zone)

                # RETRY LOOP ──────────────────────────────────────────────────
                S6_ADJUSTMENTS = [0, +15, -15]

                for attempt in range(request.max_retries):

                    # Re-scan on retry for fresh position
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
                            key=lambda d: abs(d["cx"] - base_cx)
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

                        await asyncio.sleep(1.5)   # let objects settle
                        disappeared = await _verify_pick_disappeared(
                            base_cx, frame_w, request.conf
                        )
                        attempt_log["disappeared"] = disappeared

                        _publish_cookoff("pick_attempt", {
                            "pick_n":      pick_count + 1,
                            "attempt":     attempt,
                            "s6":          s6,
                            "pick_ok":     attempt_log["pick_ok"],
                            "disappeared": disappeared,
                        })

                        pick_result["attempts"].append(attempt_log)

                        if disappeared:
                            pick_result["success"] = True
                            logs.append(
                                f"  Pick #{pick_count+1} SUCCESS "
                                f"attempt={attempt} s6={s6}"
                            )
                            break
                        else:
                            logs.append(
                                f"  Pick #{pick_count+1} MISS "
                                f"attempt={attempt} s6={s6} — retrying"
                            )
                    else:
                        # Simulation — always succeeds, no arm movement
                        attempt_log["pick_ok"]     = True
                        attempt_log["disappeared"] = True
                        pick_result["attempts"].append(attempt_log)
                        pick_result["success"] = True
                        logs.append(
                            f"  [SIM] Pick #{pick_count+1} s6={s6} cx={base_cx}"
                        )
                        break

                all_picks.append(pick_result)
                pick_count += 1

            # end for lighter in merged — outer while loop re-scans at top

        # end while

    latency_ms = round((time.time() - t_start) * 1000)
    successes  = sum(1 for p in all_picks if p["success"])

    _publish_cookoff("autonomous_done", {
        "table_clear":   table_clear,
        "picks_total":   pick_count,
        "picks_success": successes,
        "total_retries": total_retries,
        "latency_ms":    latency_ms,
    })
    logger.info("[autonomous] DONE picks=%d/%d retries=%d clear=%s lat=%dms",
                successes, pick_count, total_retries, table_clear, latency_ms)

    return {
        "ok":            successes > 0 or (pick_count == 0 and table_clear),
        "task":          request.task,
        "table_clear":   table_clear,
        "picks_total":   pick_count,
        "picks_success": successes,
        "total_retries": total_retries,
        "all_picks":     all_picks,
        "logs":          logs,
        "latency_ms":    latency_ms,
        "timestamp":     time.time(),
    }
