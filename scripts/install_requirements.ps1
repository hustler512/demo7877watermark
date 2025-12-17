# Install Python dependencies (PowerShell)
Set-StrictMode -Version Latest
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$reqPath = Join-Path $scriptDir "..\requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r $reqPath
Write-Host "Dependencies installed."
