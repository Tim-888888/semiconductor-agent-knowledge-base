$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\developTools\anaconda3\envs\dl\python.exe"

if (-not (Test-Path $python)) {
    throw "The expected dl Python interpreter was not found: $python"
}

Set-Location $projectRoot
Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "semikb.api.main:app", "--host", "127.0.0.1", "--port", "8000" -WorkingDirectory $projectRoot -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173" -WorkingDirectory (Join-Path $projectRoot "web") -WindowStyle Hidden

Write-Host "API: http://127.0.0.1:8000/docs"
Write-Host "Web: http://127.0.0.1:5173"
