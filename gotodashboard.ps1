$ErrorActionPreference = "SilentlyContinue"
$root = $PSScriptRoot
$dashboardUrl = "http://127.0.0.1:8401/?v=3#telemetry"
$apiUrl = "http://127.0.0.1:8200/health"
$python = Join-Path $root ".venv\Scripts\python.exe"

function Test-Url($url) {
    try {
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Host "SIH26145 dashboard launcher" -ForegroundColor Cyan

if (-not (Test-Url "http://127.0.0.1:8401/")) {
    Write-Host "Starting dashboard server on port 8401..." -ForegroundColor Yellow
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "http.server", "8401", "--bind", "127.0.0.1", "--directory", "dashboard" `
        -WorkingDirectory $root -WindowStyle Minimized
}

if (-not (Test-Url $apiUrl)) {
    if (Test-Path $python) {
        Write-Host "Starting FastAPI service on port 8200..." -ForegroundColor Yellow
        Start-Process -FilePath $python `
            -ArgumentList "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", "8200" `
            -WorkingDirectory $root -WindowStyle Minimized
    } else {
        Write-Host "Windows virtual environment not found. Dashboard will open without API." -ForegroundColor Red
    }
}

for ($i = 0; $i -lt 15; $i++) {
    if (Test-Url "http://127.0.0.1:8401/") { break }
    Start-Sleep -Milliseconds 500
}

Write-Host "Opening $dashboardUrl" -ForegroundColor Green
Start-Process $dashboardUrl
Write-Host "Use ALERT FEED -> TEST ATTACK for the prototype demo." -ForegroundColor Cyan
