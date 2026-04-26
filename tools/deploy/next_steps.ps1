# next_steps.ps1 — Run the 3 next steps for NIS / H100
# Usage: .\next_steps.ps1
# Or: .\next_steps.ps1 -RestartPi -LaunchHeavy -SetEnv

param(
    [switch]$RestartPi,
    [switch]$LaunchHeavy,
    [switch]$SetEnv,
    [switch]$All
)

$PI_HOST = "neurolinux@192.168.1.163"
$H100_ALIAS = "awesome-gpu-name"

Write-Host ""
Write-Host "=== NIS / H100 NEXT STEPS ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Restart Pi NIS
if ($RestartPi -or $All) {
    Write-Host "[1/3] Restarting Pi NIS via SSH..." -ForegroundColor Yellow
    ssh $PI_HOST "sudo systemctl restart nis-protocol"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK — NIS restarted. Run 'python pi_status.py --ping' to verify." -ForegroundColor Green
    } else {
        Write-Host "  Failed — you may need to run manually on the Pi:" -ForegroundColor Red
        Write-Host "    ssh $PI_HOST"
        Write-Host "    sudo systemctl restart nis-protocol"
    }
} else {
    Write-Host "[1/3] Restart Pi NIS (manual if SSH fails):" -ForegroundColor Gray
    Write-Host "  ssh $PI_HOST"
    Write-Host "  sudo systemctl restart nis-protocol"
    Write-Host "  Or run: .\next_steps.ps1 -RestartPi" -ForegroundColor DarkGray
}
Write-Host ""

# Step 2: Launch heavy training on H100
if ($LaunchHeavy -or $All) {
    Write-Host "[2/3] Launching heavy training on H100..." -ForegroundColor Yellow
    ssh $H100_ALIAS "cd ~/organica-ai 2>/dev/null || cd /data/organica-ai 2>/dev/null || cd ~ ; bash scripts/h100_heavy/launch_heavy.sh 2>/dev/null || echo 'launch_heavy.sh not found - check path'"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK — Heavy jobs launched. Check: ssh $H100_ALIAS 'nvidia-smi'" -ForegroundColor Green
    } else {
        Write-Host "  Run manually:" -ForegroundColor Red
        Write-Host "    ssh $H100_ALIAS"
        Write-Host "    cd ~/organica-ai && bash scripts/h100_heavy/launch_heavy.sh"
    }
} else {
    Write-Host "[2/3] Launch heavy training on H100 (credit-efficient):" -ForegroundColor Gray
    Write-Host "  ssh $H100_ALIAS"
    Write-Host "  cd ~/organica-ai && bash scripts/h100_heavy/launch_heavy.sh"
    Write-Host "  Or run: .\next_steps.ps1 -LaunchHeavy" -ForegroundColor DarkGray
}
Write-Host ""

# Step 3: Set H100_REASON_URL for tunnels
if ($SetEnv -or $All) {
    Write-Host "[3/3] Setting H100_REASON_URL in .env..." -ForegroundColor Yellow
    $envFile = ".env"
    if (-not (Test-Path $envFile)) {
        Copy-Item ".env.example" $envFile
        Write-Host "  Created .env from .env.example"
    }
    $content = Get-Content $envFile -Raw
    if ($content -notmatch "H100_REASON_URL") {
        Add-Content $envFile "`n# H100 Cosmos (tunnel)`nH100_REASON_URL=http://localhost:8100"
        Write-Host "  Added H100_REASON_URL=http://localhost:8100" -ForegroundColor Green
    } else {
        Write-Host "  H100_REASON_URL already in .env" -ForegroundColor Green
    }
    Write-Host "  Start tunnel first: .\open_tunnels.ps1  (keep open)" -ForegroundColor Cyan
} else {
    Write-Host "[3/3] Set H100_REASON_URL (for SSH tunnel):" -ForegroundColor Gray
    Write-Host "  Add to .env: H100_REASON_URL=http://localhost:8100"
    Write-Host "  Then run: .\open_tunnels.ps1  (keep window open)" -ForegroundColor DarkGray
    Write-Host "  Or run: .\next_steps.ps1 -SetEnv" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "  pi_status:  python pi_status.py" -ForegroundColor DarkGray
Write-Host "  H100 info:  docs/H100_CREDIT_AUDIT.md" -ForegroundColor DarkGray
Write-Host ""
