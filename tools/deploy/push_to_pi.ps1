# push_to_pi.ps1 — Push all NIS + NeuroHub changes to Pi
# Usage: .\push_to_pi.ps1
# You will be prompted for Pi password (neurolinux)

$PI = "neurolinux@192.168.1.163"
$NIS = "C:\Users\DiegoTorres\Desktop\NIS_Protocol"
$UI = "C:\Users\DiegoTorres\Desktop\NeuroLinux\phase4-distributed\neurohub-ui"

Write-Host ""
Write-Host "=== Pushing changes to Pi ===" -ForegroundColor Cyan
Write-Host ""

# 1. NIS Protocol — yolo_vision, docs
Write-Host "[1/4] Pushing NIS routes + docs..." -ForegroundColor Yellow
scp "$NIS\routes\yolo_vision.py" "${PI}:/opt/nis-protocol/routes/"
scp "$NIS\docs\fix_microphone.md" "$NIS\docs\YOLO_ACCURACY.md" "${PI}:/opt/nis-protocol/docs/"
if ($LASTEXITCODE -eq 0) { Write-Host "  OK" -ForegroundColor Green } else { Write-Host "  FAILED" -ForegroundColor Red }
Write-Host ""

# 2. Build NeuroHub UI
Write-Host "[2/4] Building NeuroHub UI..." -ForegroundColor Yellow
Set-Location $UI
npm run build 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "  OK" -ForegroundColor Green } else { Write-Host "  FAILED (npm run build)" -ForegroundColor Red }
Write-Host ""

# 3. Push built UI pages (cosmos, voice, autonomy)
Write-Host "[3/4] Pushing built UI to Pi..." -ForegroundColor Yellow
$pages = @("cosmos", "voice", "autonomy")
foreach ($p in $pages) {
    $src = "$UI\.next\server\app\$p\page.js"
    if (Test-Path $src) {
        scp $src "${PI}:/opt/neurolinux/neurohub-ui/.next/server/app/$p/page.js"
        scp "$UI\.next\server\app\$p\page.js.map" "${PI}:/opt/neurolinux/neurohub-ui/.next/server/app/$p/page.js.map" 2>$null
        Write-Host "  $p OK" -ForegroundColor Green
    }
}
Write-Host ""

# 4. Restart services
Write-Host "[4/4] Restarting NIS + neurohub-ui..." -ForegroundColor Yellow
ssh $PI "echo neurolinux | sudo -S systemctl restart nis-protocol neurohub-ui"
if ($LASTEXITCODE -eq 0) { Write-Host "  OK" -ForegroundColor Green } else { Write-Host "  Run manually: ssh $PI 'sudo systemctl restart nis-protocol neurohub-ui'" -ForegroundColor Red }
Write-Host ""

Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "  NIS:      http://192.168.1.163:8000/health" -ForegroundColor DarkGray
Write-Host "  UI:       http://192.168.1.163:3000" -ForegroundColor DarkGray
Write-Host ""

Set-Location $NIS
