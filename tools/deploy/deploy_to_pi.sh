#!/bin/bash
# Deploy updated NeuroLinux + Cosmos Cookoff files to Pi
# Run from Windows: bash deploy_to_pi.sh
# Or: ssh pi@neurolinux.local "bash -s" < deploy_to_pi.sh

PI="pi@neurolinux.local"
PI_OPT="/opt/neurolinux"
NIS_DIR="C:/Users/DiegoTorres/Desktop/NIS_Protocol"
NL_DIR="C:/Users/DiegoTorres/Desktop/NeuroLinux/neurolinux-os/buildroot/board/neurolinux/overlay/opt/neurolinux"

echo "=== Deploying to Pi: $PI ==="

# 1. Deploy cosmos_cookoff.py (H100 IP fix + async transfer submit+poll)
echo "  [1/3] cosmos_cookoff.py..."
scp "$NL_DIR/cosmos_cookoff.py" "$PI:$PI_OPT/cosmos_cookoff.py"

# 2. Deploy index.html (Transfer2.5 video player + cookoffTransfer JS)
echo "  [2/3] index.html..."
scp "$NL_DIR/static/index.html" "$PI:$PI_OPT/static/index.html"

# 3. Restart neurolinux-agent service
echo "  [3/3] Restarting neurolinux-agent..."
ssh "$PI" "sudo systemctl restart neurolinux-agent && sleep 2 && systemctl is-active neurolinux-agent"

echo ""
echo "=== Verifying Pi agent health ==="
sleep 3
curl -s http://neurolinux.local:8085/health | python3 -m json.tool 2>/dev/null || echo "  (agent not yet ready)"

echo ""
echo "=== Testing cookoff/status from Pi -> H100 ==="
curl -s http://neurolinux.local:8085/cookoff/status | python3 -m json.tool 2>/dev/null || echo "  (cookoff not available)"

echo ""
echo "=== Deploy complete ==="
