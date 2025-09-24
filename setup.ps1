# QA Service Setup Script for Windows
# This script sets up the complete environment for the Q&A service

Write-Host "Setting up QA Service..." -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists, removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating configuration file..." -ForegroundColor Yellow
    $envContent = @"
# Database Configuration
DATABASE_PATH=data/qa_service.db

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search Configuration  
VECTOR_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
TOP_K_RESULTS=20
CONFIDENCE_THRESHOLD=0.3

# API Configuration
FLASK_HOST=localhost
FLASK_PORT=5000
FLASK_DEBUG=True

# Logging
LOG_LEVEL=INFO
"@
    $envContent | Out-File -FilePath ".env" -Encoding utf8
}

# Create __init__.py files
Write-Host "Creating Python package files..." -ForegroundColor Yellow
$initDirs = @("src", "src/api", "src/ingest", "src/search", "src/utils", "tests")
foreach ($dir in $initDirs) {
    New-Item -Path "$dir/__init__.py" -ItemType File -Force | Out-Null
}

# Create sources.json template
if (-not (Test-Path "data/sources.json")) {
    Write-Host "Creating sources template..." -ForegroundColor Yellow
    $sourcesContent = @"
{
    "documents": {
        "example.pdf": {
            "title": "Example Document",
            "url": "https://example.com/document.pdf",
            "description": "An example PDF document"
        }
    }
}
"@
    $sourcesContent | Out-File -FilePath "data/sources.json" -Encoding utf8
}

Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Put your 20 PDF files in the 'pdfs/' folder" -ForegroundColor Cyan
Write-Host "Run 'python -m src.ingest.pdf_processor' to process PDFs" -ForegroundColor Cyan
Write-Host "Run 'python -m src.api.app' to start the web service" -ForegroundColor Cyan
Write-Host ""
Write-Host "Don't forget to activate the virtual environment with:" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor White
