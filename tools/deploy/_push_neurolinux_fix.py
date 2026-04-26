#!/usr/bin/env python3
"""
Push NIS Protocol fixes to Pi.

Covers all changes from Feb 27-28 2026 sessions:
  - NeuroKernel core: drive_scheduler, audit_chain, neurokernel, loop_guard, skill_loader
  - Autonomous: agent_orchestrator, edge_ai_operating_system
  - Routes: neurokernel, events (SSE), cookoff
  - App: main.py, nis_console.py
  - Agents: robotics-arm, cosmos-reason2, nis-orchestrator (agent.toml + SKILL.md)
  - NeuroLinux: __init__, robot_abstraction, edge_deployment

Usage:
  python _push_neurolinux_fix.py          # push all
  python _push_neurolinux_fix.py --core   # push only core files (faster)
"""

import base64, json, pathlib, sys, urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NIS     = 'http://192.168.1.163:8000'
LOCAL   = pathlib.Path(r'C:\Users\DiegoTorres\Desktop\NIS_Protocol')
PI_BASE = '/opt/nis-protocol'

# ── Core files (always push these) ────────────────────────────────────────────
CORE_FILES = [
    # NeuroKernel core — all fixed + circuit-broken + persistent state
    (LOCAL / 'src' / 'core' / 'drive_scheduler.py',          f'{PI_BASE}/src/core/drive_scheduler.py'),
    (LOCAL / 'src' / 'core' / 'audit_chain.py',              f'{PI_BASE}/src/core/audit_chain.py'),
    (LOCAL / 'src' / 'core' / 'neurokernel.py',              f'{PI_BASE}/src/core/neurokernel.py'),
    (LOCAL / 'src' / 'core' / 'loop_guard.py',               f'{PI_BASE}/src/core/loop_guard.py'),
    (LOCAL / 'src' / 'core' / 'skill_loader.py',             f'{PI_BASE}/src/core/skill_loader.py'),
    # Autonomous agents — task failure callbacks added
    (LOCAL / 'src' / 'core' / 'agent_orchestrator.py',       f'{PI_BASE}/src/core/agent_orchestrator.py'),
    (LOCAL / 'src' / 'core' / 'edge_ai_operating_system.py', f'{PI_BASE}/src/core/edge_ai_operating_system.py'),
    # Routes — SSE channel + cookoff arm SSE + neurokernel
    (LOCAL / 'routes' / 'events.py',                         f'{PI_BASE}/routes/events.py'),
    (LOCAL / 'routes' / 'cookoff.py',                        f'{PI_BASE}/routes/cookoff.py'),
    (LOCAL / 'routes' / 'neurokernel.py',                    f'{PI_BASE}/routes/neurokernel.py'),
    (LOCAL / 'routes' / '__init__.py',                       f'{PI_BASE}/routes/__init__.py'),
    # App entry point — EdgeAIOS wired, startup task callbacks
    (LOCAL / 'main.py',                                       f'{PI_BASE}/main.py'),
    # Pi-specific entry point — 4 stale H100 IPs fixed (192.168.1.160 -> 172.16.1.83)
    (LOCAL / 'main_pi.py',                                    f'{PI_BASE}/main_pi.py'),
    # Console — daemon mode, --list-agents, agents intent
    (LOCAL / 'nis_console.py',                                f'{PI_BASE}/nis_console.py'),
    # Cookoff demo day scripts
    (LOCAL / 'cookoff_day.py',                                f'{PI_BASE}/cookoff_day.py'),
    (LOCAL / 'cosmos_cookoff_demo.py',                        f'{PI_BASE}/cosmos_cookoff_demo.py'),
    (LOCAL / 'test_full_cookoff_demo.py',                     f'{PI_BASE}/test_full_cookoff_demo.py'),
    (LOCAL / 'pi_status.py',                                  f'{PI_BASE}/pi_status.py'),
]

# ── Agent manifests (push always — small text files) ──────────────────────────
AGENT_FILES = [
    (LOCAL / 'agents' / 'robotics-arm'   / 'agent.toml', f'{PI_BASE}/agents/robotics-arm/agent.toml'),
    (LOCAL / 'agents' / 'robotics-arm'   / 'SKILL.md',   f'{PI_BASE}/agents/robotics-arm/SKILL.md'),
    (LOCAL / 'agents' / 'cosmos-reason2' / 'agent.toml', f'{PI_BASE}/agents/cosmos-reason2/agent.toml'),
    (LOCAL / 'agents' / 'cosmos-reason2' / 'SKILL.md',   f'{PI_BASE}/agents/cosmos-reason2/SKILL.md'),
    (LOCAL / 'agents' / 'nis-orchestrator' / 'agent.toml', f'{PI_BASE}/agents/nis-orchestrator/agent.toml'),
    (LOCAL / 'agents' / 'nis-orchestrator' / 'SKILL.md',   f'{PI_BASE}/agents/nis-orchestrator/SKILL.md'),
]

# ── NeuroLinux panel (push always) ────────────────────────────────────────────
NEUROLINUX_FILES = [
    (LOCAL / 'src' / 'neurolinux' / '__init__.py',           f'{PI_BASE}/src/neurolinux/__init__.py'),
    (LOCAL / 'src' / 'neurolinux' / 'robot_abstraction.py',  f'{PI_BASE}/src/neurolinux/robot_abstraction.py'),
    (LOCAL / 'src' / 'neurolinux' / 'edge_deployment.py',    f'{PI_BASE}/src/neurolinux/edge_deployment.py'),
]

FILES = CORE_FILES + AGENT_FILES + NEUROLINUX_FILES


def nis_alive():
    try:
        urllib.request.urlopen(NIS + '/health', timeout=4)
        return True
    except Exception:
        return False


def shell(cmd, timeout=30):
    d = json.dumps({'cmd': cmd}).encode()
    r = urllib.request.Request(NIS + '/system/shell', data=d,
                               headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    return (resp.get('stdout') or resp.get('output') or resp.get('result') or '').strip()


def push_file(local: pathlib.Path, remote: str) -> bool:
    data = local.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    # backup first
    shell(f'cp {remote} {remote}.bak 2>/dev/null || true')
    cmd = (f"python3 -c \"import base64; "
           f"open('{remote}','wb').write(base64.b64decode('{b64}'))\"")
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(NIS + '/system/shell',
                               data=json.dumps({'cmd': cmd}).encode(),
                               headers={'Content-Type': 'application/json'}),
        timeout=40).read())
    ok = r.get('returncode', 1) == 0
    size = shell(f'wc -c {remote} 2>/dev/null')
    print(f"  {'OK' if ok else 'FAIL'}: {local.name} -> {remote}  ({size})")
    return ok


def main():
    core_only = '--core' in sys.argv
    file_set = CORE_FILES if core_only else FILES
    label = 'core files only' if core_only else f'all {len(file_set)} files'

    if not nis_alive():
        print("NIS is DOWN at", NIS)
        print("Start it first: sudo systemctl restart nis-protocol")
        sys.exit(1)

    # Ensure new agent directories exist on Pi
    for d in ('agents/cosmos-reason2', 'agents/nis-orchestrator'):
        shell(f'mkdir -p {PI_BASE}/{d}', timeout=10)

    print(f"\n=== Pushing NIS Protocol fixes to Pi ({label}) ===\n")
    all_ok = True
    for local, remote in file_set:
        if not local.exists():
            print(f"  SKIP (not found locally): {local.name}")
            continue
        ok = push_file(local, remote)
        all_ok = all_ok and ok

    if all_ok:
        print("\n[restart] Restarting nis-protocol...")
        r = shell('sudo systemctl restart nis-protocol 2>&1 || echo restart_failed', timeout=20)
        print(f"  {r or '(sent)'}")
        print("\nDone. Verify: http://192.168.1.163:8000/neurokernel/status")
    else:
        print("\nSome files failed — check NIS /system/shell endpoint.")


if __name__ == '__main__':
    main()
