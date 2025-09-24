# Industrial Safety Q&A System - Startup Script

Write-Host "Starting QA System..." -ForegroundColor Green

# Check prerequisites
if (-not (Test-Path ".\venv")) {
    Write-Host "Error: Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".\data\qa_service.db")) {
    Write-Host "Error: Database not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Initializing web server on http://localhost:5000"
Write-Host "Press Ctrl+C to stop"

# Launch application
& .\venv\Scripts\python.exe start_clean.py
