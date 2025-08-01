
Write-Host "Running NVIDIA API Test..." -ForegroundColor Cyan
& 'C:\Users\penti\OneDrive\Desktop\NIS_Protocol\.venv\Scripts\python.exe' "$PSScriptRoot\quick_nvidia_test.py"
Write-Host ""
Write-Host "Test complete! Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
