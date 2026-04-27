# Open SSH tunnels to H100 for all Cosmos stack ports
# Forwards: 8000 (NIS), 8100 (Reason2), 8200 (Predict2.5), 8300 (Transfer2.5), 8400 (Demo)
# Run this in a terminal and keep it open while working with the Cosmos stack

Write-Host "Opening SSH tunnels to H100 (awesome-gpu-name)..." -ForegroundColor Cyan
Write-Host "Ports: 8000=NIS  8100=Reason2  8200=Predict2.5  8300=Transfer2.5  8400=Demo" -ForegroundColor Yellow
Write-Host "Keep this window open. Ctrl+C to close tunnels." -ForegroundColor Green
Write-Host ""

ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=no `
    -L 8000:localhost:8000 `
    -L 8100:localhost:8100 `
    -L 8200:localhost:8200 `
    -L 8300:localhost:8300 `
    -L 8400:localhost:8400 `
    awesome-gpu-name
