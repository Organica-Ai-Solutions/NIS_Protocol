# start_tunnel.ps1
# Keeps SSH tunnel to H100 alive (auto-restarts on disconnect)
# Run once: powershell -ExecutionPolicy Bypass -File start_tunnel.ps1
# Tunnels H100 Cosmos ports to localhost so NIS on PC can reach them

Write-Host "Starting persistent SSH tunnel to H100 (Cosmos ports 8100/8200/8300)..."
Write-Host "Press Ctrl+C to stop."

while ($true) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Connecting tunnel..."
    ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
        -N `
        -L 8100:localhost:8100 `
        -L 8200:localhost:8200 `
        -L 8300:localhost:8300 `
        awesome-gpu-name
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Tunnel dropped — reconnecting in 5s..."
    Start-Sleep -Seconds 5
}
