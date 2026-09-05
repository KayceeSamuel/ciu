# CIU installer for Windows.
#
#   irm https://raw.githubusercontent.com/KayceeSamuel/ciu/main/install.ps1 | iex
#
# Installs into %USERPROFILE%\.ciu and touches nothing else. Python packages
# go in a private virtualenv so this cannot break an existing Python or conda
# setup. To remove everything: Remove-Item -Recurse $HOME\.ciu

$ErrorActionPreference = 'Stop'

$CiuRepo  = 'KayceeSamuel/ciu'
$ForkRepo = 'KayceeSamuel/llama.cpp'
$Prefix   = Join-Path $HOME '.ciu'

function Step($m) { Write-Host $m -ForegroundColor White }
function Dim($m)  { Write-Host $m -ForegroundColor DarkGray }
function Die($m)  { Write-Host $m -ForegroundColor Red; exit 1 }

# ----------------------------------------------------------------- platform

if ([Environment]::Is64BitOperatingSystem -eq $false) {
    Die 'CIU needs 64-bit Windows.'
}

# nvidia-smi ships with the driver, so its presence is a reliable signal that
# a usable NVIDIA GPU is installed.
$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

if ($hasNvidia) {
    $asset  = 'llama-server-windows-x64-cuda.zip'
    $backend = 'CUDA'
} else {
    $asset  = 'llama-server-windows-x64-cpu.zip'
    $backend = 'CPU'
    Write-Host ''
    Write-Host 'No NVIDIA GPU found.' -ForegroundColor Yellow
    Write-Host 'CIU has CUDA and Metal kernels but no Vulkan kernel yet, so AMD and'
    Write-Host 'Intel GPUs fall back to the CPU. That works but generates at well'
    Write-Host 'under one token a second, which is not usable for real work.'
    Write-Host ''
    $reply = Read-Host 'Install anyway? [y/N]'
    if ($reply -notmatch '^[Yy]') { Die 'Stopped.' }
}

# Find a Python new enough to run CIU. The Windows launcher (py.exe) is the
# most reliable route when several versions are installed.
$py = $null
foreach ($cand in @('py -3.12', 'py -3.11', 'py -3', 'python3', 'python')) {
    $parts = $cand.Split(' ')
    $exe = Get-Command $parts[0] -ErrorAction SilentlyContinue
    if (-not $exe) { continue }
    try {
        $args = @()
        if ($parts.Count -gt 1) { $args += $parts[1] }
        $args += @('-c', 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)')
        & $parts[0] @args 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    } catch { }
}
if (-not $py) {
    Die 'Python 3.10 or newer is required. Install it from python.org and run this again.'
}

Write-Host ''
Step "Installing CIU"
Dim  "Windows x64, $backend backend, into $Prefix"
Write-Host ''

New-Item -ItemType Directory -Force -Path "$Prefix\bin", "$Prefix\models" | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ciu-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    # ------------------------------------------------------- llama-server
    # The NF4DQ fork, prebuilt. Building it from source needs Visual Studio
    # and the CUDA toolkit, which is the step this exists to remove.

    Step '1/3  Fetching the inference engine'
    $url = "https://github.com/$ForkRepo/releases/latest/download/$asset"
    try {
        Invoke-WebRequest -Uri $url -OutFile "$tmp\engine.zip" -UseBasicParsing
    } catch {
        Write-Host ''
        Write-Host "Could not download $asset."
        Write-Host "Check https://github.com/$ForkRepo/releases for what is published."
        Die 'Stopped.'
    }
    Expand-Archive -Path "$tmp\engine.zip" -DestinationPath "$Prefix\bin" -Force

    # Windows marks anything downloaded from the internet and SmartScreen may
    # refuse to run it. Clearing the mark here saves the user a dialog they
    # have no way to interpret.
    Get-ChildItem "$Prefix\bin" -Recurse |
        Unblock-File -ErrorAction SilentlyContinue

    if (-not (Test-Path "$Prefix\bin\llama-server.exe")) {
        Die 'The download did not contain llama-server.exe.'
    }

    # --------------------------------------------------------------- CIU

    Step '2/3  Fetching CIU'
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://github.com/$CiuRepo/archive/refs/heads/main.zip" `
        -OutFile "$tmp\ciu.zip"
    if (Test-Path "$Prefix\app") { Remove-Item -Recurse -Force "$Prefix\app" }
    Expand-Archive -Path "$tmp\ciu.zip" -DestinationPath $tmp -Force
    Move-Item (Join-Path $tmp 'ciu-main') "$Prefix\app"

    # ---------------------------------------------------------- packages
    # A private virtualenv. Installing into the user's own Python is how you
    # break someone's environment and get blamed for it.

    Step '3/3  Installing Python packages'
    $pyParts = $py.Split(' ')
    $venvArgs = @()
    if ($pyParts.Count -gt 1) { $venvArgs += $pyParts[1] }
    $venvArgs += @('-m', 'venv', "$Prefix\venv")
    & $pyParts[0] @venvArgs
    if ($LASTEXITCODE -ne 0) { Die 'Could not create a virtualenv.' }

    & "$Prefix\venv\Scripts\python.exe" -m pip install --quiet --upgrade pip 2>$null
    & "$Prefix\venv\Scripts\python.exe" -m pip install --quiet `
        fastapi uvicorn httpx huggingface_hub gguf
    if ($LASTEXITCODE -ne 0) { Die 'Could not install Python packages.' }

    # --------------------------------------------------------- launcher

    $launcher = @"
@echo off
REM Starts CIU and opens the page.
set "CIU_LLAMA_SERVER=$Prefix\bin\llama-server.exe"
cd /d "$Prefix\app"
start "" http://127.0.0.1:8674
"$Prefix\venv\Scripts\python.exe" run.py
"@
    Set-Content -Path "$Prefix\bin\ciu.cmd" -Value $launcher -Encoding ASCII

    # Put it on PATH for this user so `ciu` works from any new terminal.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$Prefix\bin*") {
        [Environment]::SetEnvironmentVariable(
            'Path', "$userPath;$Prefix\bin", 'User')
        $pathAdded = $true
    }

    Write-Host ''
    Step 'Done.'
    Write-Host ''
    Write-Host '  Start it with:   ' -NoNewline
    Write-Host 'ciu' -ForegroundColor Cyan
    if ($pathAdded) {
        Dim '  (Open a new terminal first, so it picks up the new PATH.)'
    }
    Write-Host ''
    Write-Host '  Your browser opens at http://127.0.0.1:8674, where you pick a model'
    Write-Host '  that fits your machine. Everything runs locally.'
    Write-Host ''
    Dim  "  To remove: Remove-Item -Recurse $Prefix"
    Write-Host ''
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
