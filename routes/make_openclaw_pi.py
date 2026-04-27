"""
Build openclaw_pi.py — Pi-edge-only version of the openclaw bridge.
Run from the NIS_Protocol directory.
"""
import re

with open(r"routes/openclaw.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def remove_ranges(lines, ranges):
    skip_set = set()
    for s, e in ranges:
        for i in range(s, e+1):
            skip_set.add(i)
    return [line for i, line in enumerate(lines, 1) if i not in skip_set]

# Remove Windows-only handler functions (1-indexed inclusive)
ranges_to_remove = [
    (450, 460),   # Windows service BASE URL constants
    (466, 506),   # _handle_alphacortex
    (507, 546),   # _handle_arbitrage
    (887, 942),   # _handle_organica
    (943, 986),   # _handle_orion
    (987, 1036),  # _handle_portfolio
    (1037, 1112), # _handle_mexpay
    (1113, 1173), # _handle_hub
    (1174, 1248), # _handle_moe
    (1249, 1320), # _handle_auto
    (1321, 1386), # _handle_organica_web
    (1387, 1491), # _handle_cryptobot
]

result = remove_ranges(lines, ranges_to_remove)
content = "".join(result)

# ── Docstring: remove Windows tool lines ──────────────────────────────────────
for line in [
    "  nis_alphacortex   — Query AlphaCortex: positions, account, analysis, orders\n",
    "  nis_arbitrage     — Query ArbitrageMachine: status, opportunities, metrics\n",
    "  nis_organica      — Route to any of the 39 Organica Framework specialized agents\n",
    "  nis_orion         — Send a code/build/debug question to Orion coding AI (:8080)\n",
    "  nis_portfolio     — Query SmartPortfolio: account, optimization status, summary\n",
    "  nis_mexpay        — Mexican fintech: SPEI real-time payments, ISO 20022 messaging (:3010)\n",
    "  nis_hub           — NIS-HUB: central orchestration, node management, fleet, missions (:8003)\n",
    "  nis_moe           — NIS MoE: semantic embeddings and similarity search (:8004)\n",
    "  nis_auto          — NIS-AUTO: automotive/OBD-II AGI agent (:8005)\n",
    "  nis_organica_web  — OrganicaAI website backend: Gemini chat, user auth (:5001)\n",
    "  nis_cryptobot     — CryptoBot: Alpaca crypto trading, strategies, backtesting (:5002)\n",
]:
    content = content.replace(line, "")

# ── Pi-only _handle_stack ─────────────────────────────────────────────────────
pi_stack = '''async def _handle_stack(args):
    """Pi-native service health snapshot."""
    import httpx, asyncio as _aio
    async def ping(url):
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(url)
                return r.status_code < 500
        except Exception:
            return False
    xarm_ok, yolo_ok, nk_ok, of_ok = await _aio.gather(
        ping("http://localhost:8085/health"),
        ping("http://localhost:8000/yolo/status"),
        ping("http://localhost:8000/neurokernel/health"),
        ping("http://localhost:8000/openfang/status"),
    )
    return {
        "stack": {
            "xarm_agent :8085": bool(xarm_ok),
            "yolo :8000": bool(yolo_ok),
            "neurokernel :8000": bool(nk_ok),
            "openfang :8000": bool(of_ok),
        },
        "node": "pi-edge",
        "timestamp": time.time(),
    }

'''
idx_start = content.find("async def _handle_stack(args: Dict[str, Any]) -> Dict[str, Any]:")
idx_end = content.find("\nasync def _dispatch_tool")
content = content[:idx_start] + pi_stack + content[idx_end+1:]

# ── Strip Windows dispatch routes ─────────────────────────────────────────────
for line in [
    '    if t in ("alphacortex", "alpha", "trading"):\n        return await _handle_alphacortex(args)\n',
    '    if t in ("arbitrage", "arb", "crypto"):\n        return await _handle_arbitrage(args)\n',
    '    if t in ("organica", "agents", "agent_framework"):\n        return await _handle_organica(args)\n',
    '    if t in ("orion", "code", "coding"):\n        return await _handle_orion(args)\n',
    '    if t in ("portfolio", "smartportfolio", "sp"):\n        return await _handle_portfolio(args)\n',
    '    if t in ("mexpay", "spei", "iso20022", "payments", "fintech"):\n        return await _handle_mexpay(args)\n',
    '    if t in ("hub", "nis_hub", "orchestration", "fleet"):\n        return await _handle_hub(args)\n',
    '    if t in ("moe", "embed", "embeddings", "semantic"):\n        return await _handle_moe(args)\n',
    '    if t in ("auto", "automotive", "obd", "vehicle"):\n        return await _handle_auto(args)\n',
    '    if t in ("organica_web", "orgweb", "organicaweb", "gemini_chat"):\n        return await _handle_organica_web(args)\n',
    '    if t in ("cryptobot", "crypto_bot", "cryptotrading", "trading_bot"):\n        return await _handle_cryptobot(args)\n',
]:
    content = content.replace(line, "")

# ── Strip Windows tools from agent TOOL_DESCRIPTIONS ─────────────────────────
for desc in [
    "        \"nis_alphacortex: US equity trading. Args: {action: 'status'|'positions'|'account'|'analyze', symbol?: str}\\n\"\n",
    "        \"nis_arbitrage: Crypto arbitrage. Args: {action: 'status'|'opportunities'|'metrics'}\\n\"\n",
    "        \"nis_organica: Route to one of 39 specialized Organica agents. Args: {action: 'route'|'list'|'call', message?: str, agent_id?: str}\\n\"\n",
    "        \"nis_orion: Orion TypeScript coding AI. Args: {message: str} \u2014 use for code, build, debug questions\\n\"\n",
    "        \"nis_portfolio: SmartPortfolio service. Args: {action: 'status'|'account'|'health'}\\n\"\n",
    "        \"nis_mexpay: Mexican fintech \u2014 SPEI payments, ISO 20022, banks. Args: {action: 'status'|'transfer'|'history'|'banks'|'validate', from_clabe?: str, to_clabe?: str, amount?: number, concept?: str, clabe?: str}\\n\"\n",
    "        \"nis_hub: NIS-HUB orchestration \u2014 nodes, fleet, missions. Args: {action: 'status'|'nodes'|'fleet'|'missions'|'health'}\\n\"\n",
    "        \"nis_moe: NIS MoE semantic embeddings. Args: {action: 'embed'|'similarity'|'batch'|'info'|'health', text?: str, text1?: str, text2?: str, texts?: list}\\n\"\n",
    "        \"nis_auto: NIS-AUTO automotive AGI. Args: {action: 'status'|'chat'|'consciousness'|'agents'|'process', message?: str, text?: str}\\n\"\n",
    "        \"nis_organica_web: OrganicaAI Gemini chat. Args: {action: 'health'|'login'|'chat', email?: str, password?: str, token?: str, message?: str}\\n\"\n",
    "        \"nis_cryptobot: CryptoBot Alpaca trading. Args: {action: 'status'|'health'|'account'|'positions'|'trades'|'market'|'strategies'|'start'|'stop'|'backtest', symbol?: str}\\n\"\n",
]:
    content = content.replace(desc, "")

# ── Pi-only /status return ────────────────────────────────────────────────────
old_gather = (
    "    (\n"
    "        alpha_ok, arb_ok, sp_ok, nk_ok, of_ok, org_ok, orion_ok,\n"
    "        mexpay_ok, hub_ok, moe_ok, auto_ok, orgweb_ok, cryptobot_ok,\n"
    "    ) = await asyncio.gather(\n"
    "        _ping(f\"{_ALPHA_BASE}/health\"),\n"
    "        _ping(f\"{_ARB_BASE}/api/health\"),\n"
    "        _ping(f\"{_SP_BASE}/health\"),\n"
    "        _ping(f\"{_PI_NIS_BASE}/neurokernel/health\"),\n"
    "        _ping(f\"{_PI_NIS_BASE}/openfang/status\"),\n"
    "        _ping(f\"{_ORG_BASE}/health\"),\n"
    "        _ping(f\"{_ORION_BASE}\"),\n"
    "        _ping(f\"{_MEXPAY_BASE}/api/health\"),\n"
    "        _ping(f\"{_HUB_BASE}/health\"),\n"
    "        _ping(f\"{_MOE_BASE}/health\"),\n"
    "        _ping(f\"{_AUTO_BASE}/health\"),\n"
    "        _ping(f\"{_ORGWEB_BASE}/\"),\n"
    "        _ping(f\"{_CRYPTOBOT_BASE}/api/health\"),\n"
    "        return_exceptions=False,\n"
    "    )"
)
new_gather = (
    "    (\n"
    "        nk_ok, of_ok,\n"
    "    ) = await asyncio.gather(\n"
    "        _ping(\"http://localhost:8000/neurokernel/health\"),\n"
    "        _ping(\"http://localhost:8000/openfang/status\"),\n"
    "        return_exceptions=False,\n"
    "    )"
)
content = content.replace(old_gather, new_gather)

old_return = (
    "    return {\n"
    "        \"status\": \"ok\",\n"
    "        \"capabilities\": {\n"
    "            \"nis_chat\": llm_ok,\n"
    "            \"nis_cosmos_plan\": cosmos_ok,\n"
    "            \"nis_xarm\": xarm_ok,\n"
    "            \"nis_skills\": skills_count > 0,\n"
    "            \"nis_alphacortex\": bool(alpha_ok),\n"
    "            \"nis_arbitrage\": bool(arb_ok),\n"
    "            \"nis_stack\": True,\n"
    "            \"nis_neurokernel\": bool(nk_ok),\n"
    "            \"nis_openfang\": bool(of_ok),\n"
    "            \"nis_organica\": bool(org_ok),\n"
    "            \"nis_orion\": bool(orion_ok),\n"
    "            \"nis_portfolio\": bool(sp_ok),\n"
    "            \"nis_mexpay\": bool(mexpay_ok),\n"
    "            \"nis_hub\": bool(hub_ok),\n"
    "            \"nis_moe\": bool(moe_ok),\n"
    "            \"nis_auto\": bool(auto_ok),\n"
    "            \"nis_organica_web\": bool(orgweb_ok),\n"
    "            \"nis_cryptobot\": bool(cryptobot_ok),\n"
    "            \"nis_agent\": llm_ok,\n"
    "        },\n"
    "        \"services\": {\n"
    "            \"alphacortex\": bool(alpha_ok),\n"
    "            \"arbitrage\": bool(arb_ok),\n"
    "            \"smartportfolio\": bool(sp_ok),\n"
    "            \"organica\": bool(org_ok),\n"
    "            \"orion\": bool(orion_ok),\n"
    "            \"neurokernel_pi\": bool(nk_ok),\n"
    "            \"openfang_pi\": bool(of_ok),\n"
    "            \"mexpay\": bool(mexpay_ok),\n"
    "            \"nis_hub\": bool(hub_ok),\n"
    "            \"nis_moe\": bool(moe_ok),\n"
    "            \"nis_auto\": bool(auto_ok),\n"
    "            \"organica_web\": bool(orgweb_ok),\n"
    "            \"cryptobot\": bool(cryptobot_ok),\n"
    "        },\n"
    "        \"skills_loaded\": skills_count,\n"
    "        \"bridge_version\": \"1.7\",\n"
    "    }"
)
new_return = (
    "    return {\n"
    "        \"status\": \"ok\",\n"
    "        \"node\": \"pi-edge\",\n"
    "        \"capabilities\": {\n"
    "            \"nis_chat\": llm_ok,\n"
    "            \"nis_cosmos_plan\": cosmos_ok,\n"
    "            \"nis_xarm\": xarm_ok,\n"
    "            \"nis_skills\": skills_count > 0,\n"
    "            \"nis_neurokernel\": bool(nk_ok),\n"
    "            \"nis_openfang\": bool(of_ok),\n"
    "            \"nis_stack\": True,\n"
    "            \"nis_agent\": llm_ok,\n"
    "        },\n"
    "        \"services\": {\n"
    "            \"xarm_agent\": xarm_ok,\n"
    "            \"neurokernel\": bool(nk_ok),\n"
    "            \"openfang\": bool(of_ok),\n"
    "        },\n"
    "        \"skills_loaded\": skills_count,\n"
    "        \"bridge_version\": \"1.7-pi\",\n"
    "    }"
)
content = content.replace(old_return, new_return)

with open(r"routes/openclaw_pi.py", "w", encoding="utf-8") as f:
    f.write(content)

lines_out = len(content.splitlines())
print(f"openclaw_pi.py written: {lines_out} lines (was 1889)")
